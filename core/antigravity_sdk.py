"""Antigravity SDK entry point, executed only inside the existing robot sandbox."""
import argparse
import asyncio
import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

STATE = Path('/tmp/home/.gemini/antigravity-cli')
CONFIG = Path('/tmp/home/.gemini/config/roboarcade-sdk.json')


def allowed_tool(name, server_name, path=None):
    if server_name:
        return server_name == 'roboarcade'
    if name != 'view_file' or not path:
        return False
    if str(path).startswith('file:'):
        parsed = urlparse(str(path))
        if parsed.netloc not in ('', 'localhost'):
            return False
        path = unquote(parsed.path)
    target = Path(path).resolve()
    return any(target.is_relative_to(STATE / part) for part in ('brain', 'mcp'))


def tool_name(name):
    return getattr(name, 'value', name)


def tool_path(call):
    # Native view_file uses AbsolutePath, which SDK 0.1.22 does not normalize.
    return call.canonical_path or call.args.get('AbsolutePath')


def usage_record(usage):
    cached = usage.get('cached_content_token_count') or 0
    thinking = usage.get('thoughts_token_count') or 0
    return {'input_tokens': max(0, (usage.get('prompt_token_count') or 0) - cached),
            'cache_read_tokens': cached, 'thinking_tokens': thinking,
            'output_tokens': (usage.get('candidates_token_count') or 0) + thinking}


def emit(event, **data):
    print(json.dumps({'event': event, 'timestamp': datetime.now(timezone.utc).isoformat(),
                      **data}), flush=True)


async def run(args):
    from google.antigravity import Agent, LocalAgentConfig
    from google.antigravity import hooks
    from google.antigravity.models import ModelTarget, GeminiAPIEndpoint, GeminiModelOptions
    from google.antigravity.types import (CapabilitiesConfig, CompactionConfig,
        CustomSystemInstructions, McpStdioServer, BuiltinTools, HookResult,
        SessionContinuationMode)

    config = json.loads(CONFIG.read_text())
    if config['model'] != args.model:
        raise ValueError('Model does not match the read-only run configuration')
    calls = {}
    finished_tools = set()

    @hooks.pre_tool_call_decide
    def guard(call):
        calls[call.step_id or call.id] = call
        allowed = allowed_tool(tool_name(call.name), call.server_name, tool_path(call))
        emit('tool_permission', tool=call.model_dump(mode='json'), allowed=allowed)
        return HookResult(allow=allowed,
                          message='Only robot MCP tools and saved observations are allowed.')

    @hooks.post_tool_call
    def tool_result(result):
        identity = result.step_id or result.id
        if identity in finished_tools:
            return
        finished_tools.add(identity)
        call = calls.get(identity)
        cid, _, index = (identity or ':0').rpartition(':')
        info = {'name': tool_name(result.name), 'parameters': call.args if call else {},
                'output': result.result}
        if result.server_name:
            info.update(name='call_mcp_tool', parameters={
                'ServerName': result.server_name, 'ToolName': tool_name(result.name),
                'Arguments': call.args if call else {}})
        if result.error:
            info['error'] = {'message': result.error}
        emit('step_update', step_update={'conversation_id': cid, 'step_index': int(index),
             'step_type': 'tool', 'state': 'ERROR' if result.error else 'DONE', 'tool_info': info})

    servers = [McpStdioServer(name=name, command=spec['command'], args=spec.get('args', []),
               env=spec.get('env', {}), force_all_tools_eager=True)
               for name, spec in config['mcp'].items()]
    cfg = LocalAgentConfig(
        model=ModelTarget(name=args.model, endpoint=GeminiAPIEndpoint(
            options=GeminiModelOptions(max_output_tokens=65536))),
        system_instructions=CustomSystemInstructions(text=config['instructions']),
        capabilities=CapabilitiesConfig(enabled_tools=[BuiltinTools.VIEW_FILE], enable_subagents=False),
        compaction_config=CompactionConfig(token_threshold=config['compact_threshold']),
        mcp_servers=servers, policies=[], hooks=[guard, tool_result],
        workspaces=['/tmp/home/workspace'], app_data_dir=str(STATE), save_dir=str(STATE/'sessions'),
        conversation_id=args.conversation,
        session_continuation_mode=SessionContinuationMode.RESUME if args.conversation else None)
    async with Agent(cfg) as agent:
        initialized = False
        responses = {s.id for s in agent.conversation.history if s.source.value == 'MODEL'
                     and s.status.value == 'DONE'}

        def snapshot():
            nonlocal initialized
            if not agent.conversation_id:
                return
            # A resumed history can arrive after __aenter__, during send().
            responses.update(s.id for s in agent.conversation.history
                             if s.source.value == 'MODEL' and s.status.value == 'DONE')
            if not initialized:
                emit('init', conversation_id=agent.conversation_id, runtime='antigravity_sdk',
                     sdk_version='0.1.22', compact_threshold=config['compact_threshold'])
                initialized = True
            usage = usage_record(agent.conversation.total_usage.model_dump())
            emit('sdk_usage', conversation_id=agent.conversation_id, usage=usage,
                 completed_model_steps=len(responses))

        async def heartbeat():
            while True:
                snapshot()
                await asyncio.sleep(2)

        beat = asyncio.create_task(heartbeat())
        completed = set()
        raw_path = STATE/'sdk_steps.jsonl'
        try:
            await agent.conversation.send(args.prompt)
            with raw_path.open('a') as raw:
                async for step in agent.conversation.receive_steps():
                    if not initialized:
                        snapshot()
                    raw.write(step.model_dump_json() + '\n')
                    raw.flush()
                    if step.source.value == 'MODEL' and step.status.value == 'DONE':
                        responses.add(step.id)
                    if step.type.value == 'COMPACTION' and step.status.value == 'DONE':
                        if step.id not in completed:
                            completed.add(step.id)
                            emit('compaction', conversation_id=step.trajectory_id,
                                 step_index=step.step_index, threshold=config['compact_threshold'])
                    if step.source.value == 'MODEL' and step.type.value in ('TEXT_RESPONSE', 'THINKING'):
                        emit('step_update', step_update={
                            'conversation_id': step.trajectory_id, 'step_index': step.step_index,
                            'step_type': 'agent_response', 'state': step.status.value,
                            'text_delta': step.content_delta})
                    if step.error and step.type.value != 'TOOL_CALL':
                        raise RuntimeError(step.error)
            snapshot()
            emit('result', result={'status': 'SUCCESS', 'conversation_id': agent.conversation_id})
        except Exception as exc:
            snapshot()
            emit('result', result={'status': 'ERROR', 'conversation_id': agent.conversation_id,
                                  'error': str(exc)})
            raise
        finally:
            beat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--conversation')
    parser.add_argument('-p', dest='prompt', required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
