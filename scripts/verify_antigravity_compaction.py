"""Live API test: checkpoint threshold, marker retention, and process resume.

Run with the isolated google-antigravity==0.1.22 Python. Uses real API tokens.
The threshold counts native trajectory estimates, not Gemini API prompt tokens.
"""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import (
    CapabilitiesConfig, CompactionConfig, CustomSystemInstructions, SessionContinuationMode,
)


async def probe(args, threshold, key):
    root = args.output / str(threshold)
    root.mkdir(parents=True, exist_ok=False)
    config = LocalAgentConfig(
        model=args.model, api_key=key,
        capabilities=CapabilitiesConfig(enabled_tools=[], enable_subagents=False),
        compaction_config=CompactionConfig(token_threshold=threshold),
        system_instructions=CustomSystemInstructions(text=
            'Remember any marker given by the user. Reply OK when given padding; never '
            'analyze or reproduce the padding. Return the marker only when asked.'),
        workspaces=[str(root)], app_data_dir=str(root/'state'), save_dir=str(root/'saved'))
    report = {'threshold': threshold, 'token_accounting': 'native_estimate',
              'model': args.model, 'sdk_version': importlib.metadata.version('google-antigravity'),
              'turns': []}
    async with Agent(config) as agent:
        for turn in range(5):
            if turn < 4:
                prompt = ('Remember this marker for later: AMBER-9462. ' if turn == 0 else '')
                prompt += 'Reply only OK. Inert padding: ' + 'padding '*30000
            else:
                prompt = 'What marker did I give you earlier? Reply with it only.'
            response = await agent.chat(prompt)
            answer = await response.text()
            checkpoints = {s.id for s in agent.conversation.history
                           if s.type.value == 'COMPACTION' and s.status.value == 'DONE'}
            report['turns'].append({'turn': turn, 'answer': answer,
                'usage': response.usage_metadata.model_dump(mode='json'),
                'completed_checkpoints': sorted(checkpoints)})
            (root/'report.json').write_text(json.dumps(report, indent=2))
        cid = agent.conversation_id
        (root/'steps.json').write_text(json.dumps(
            [s.model_dump(mode='json') for s in agent.conversation.history]))
    config = config.model_copy(update={'conversation_id': cid,
        'session_continuation_mode': SessionContinuationMode.RESUME})
    async with Agent(config) as agent:
        response = await agent.chat('What marker did I give you earlier? Reply with it only.')
        report['resume_answer'] = await response.text()
        report['conversation_id'] = agent.conversation_id
    report['passed'] = (report['turns'][-1]['answer'].strip() == 'AMBER-9462'
                        and report['resume_answer'].strip() == 'AMBER-9462'
                        and report['conversation_id'] == cid
                        and bool(report['turns'][-1]['completed_checkpoints']) == (threshold == 245000))
    (root/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'threshold': threshold, 'passed': report['passed'], 'report': str(root/'report.json')}), flush=True)
    return report['passed']


async def verify(args):
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not key:
        key = Path(os.environ.get('ROBOARCADE_GEMINI_KEY_FILE', 'secrets/gemini_key.txt')).read_text().strip()
    if not key:
        raise RuntimeError('Gemini API key is empty')
    results = await asyncio.gather(*(probe(args, n, key) for n in (245000, 350000)))
    if not all(results):
        raise RuntimeError('Compaction verification failed; inspect the reports')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='gemini-4-argon')
    parser.add_argument('--output', type=Path, required=True, help='New directory for reports and saved sessions')
    arguments = parser.parse_args()
    arguments.output = arguments.output.resolve()
    if importlib.metadata.version('google-antigravity') != '0.1.22':
        raise RuntimeError('This verification is pinned to google-antigravity==0.1.22')
    asyncio.run(asyncio.wait_for(verify(arguments), timeout=900))
