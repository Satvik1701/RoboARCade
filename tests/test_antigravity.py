"""Run with .venv/bin/python -m pytest tests/test_antigravity.py."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import antigravity as agy
from core.agent_sandbox import model_runtime
from core import run_seed as seed


def test_api_key_is_read_without_entering_config(tmp_path, monkeypatch):
    secret = tmp_path / 'key.txt'
    secret.write_text('test-key\n')
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('ROBOARCADE_GEMINI_KEY_FILE', str(secret))
    assert agy.api_key() == 'test-key'
    monkeypatch.setenv('GEMINI_API_KEY', 'environment-key')
    assert agy.api_key() == 'environment-key'
    config = tmp_path / 'mcp.json'
    config.write_text(json.dumps({'mcpServers': {'roboarcade': {'command': '.venv/bin/python',
                                                              'args': ['core/mcp_server.py']}}}))
    guide = tmp_path / 'guide.md'
    guide.write_text('Only the robot interface')
    root = Path(agy.prepare(tmp_path, 'gemini-4-argon', config, 8100,
                           {'GEMINI_API_KEY': 'environment-key', 'ROBOARCADE_INTERFACE': '/iface'}, [guide]))
    text = '\n'.join(p.read_text() for p in (root/'config').iterdir())
    assert 'test-key' not in text and 'environment-key' not in text
    settings = json.loads((root/'config/settings.json').read_text())
    assert settings['permissions']['allow'] == [
        'mcp(roboarcade/*)', 'read_file(/tmp/home/.gemini/antigravity-cli/brain)',
        'read_file(/tmp/home/.gemini/antigravity-cli/mcp/roboarcade)']
    assert 'command(*)' in settings['permissions']['deny']
    assert 'read_file(/proc)' in settings['permissions']['deny']
    mcp = json.loads((root/'config/mcp_config.json').read_text())
    assert Path(mcp['mcpServers']['roboarcade']['command']).is_absolute()
    assert mcp['mcpServers']['roboarcade']['env']['ROBOARCADE_BASE_PORT'] == '8100'


def test_native_runtimes_are_preserved_and_gemini_always_uses_box():
    assert model_runtime('gemini-4-argon') == 'antigravity'
    assert model_runtime('gpt-6-astra') == 'codex'
    assert model_runtime('claude-fable-5-1') == 'claude'
    cmd = agy.command(prompt='task', model='gemini-4-argon', session_id='session',
                      base_port=8100, pkg='/iface', resume=True)
    assert cmd[0].endswith('agent_sandbox.sh')
    assert cmd[cmd.index('--conversation')+1] == 'session'
    assert '--dangerously-skip-permissions' not in cmd
    assert '--effort' not in cmd


def test_provider_failure_is_not_scored_as_a_robot_failure(tmp_path, monkeypatch):
    log = tmp_path / 'agent.jsonl'
    log.write_text(json.dumps({'event': 'result', 'result': {'status':'ERROR', 'error':'quota exhausted'}})+'\n')
    monkeypatch.setattr(seed, 'run_agent', lambda *a, **k: 'agent_error')
    result = {}
    status, error = seed.drive_antigravity(seed.Experiment(task='bowl_on_plate', language=''),
        'test', 'prompt', str(log), {'ROBOARCADE_BASE_PORT':'8100'}, [], [], None,
        str(tmp_path), 0, result, {})
    assert status == 'agent_error' and error == 'quota exhausted'
    assert result['agent_error'] == 'quota exhausted'


def test_resume_preserves_conversation(tmp_path, monkeypatch):
    log = tmp_path / 'agent.jsonl'
    log.write_text(json.dumps({'event':'result','result': {'status':'SUCCESS', 'conversation_id':'same'}})+'\n')
    calls = []
    def run(*args):
        calls.append(args)
        return 'ok'
    monkeypatch.setattr(seed, 'run_agent', run)
    monkeypatch.setattr(seed, 'episode_over', lambda *a: len(calls) == 2)
    monkeypatch.setattr(seed, 'mcp_ui_unattached', lambda *a: False)
    result = {}
    seed.drive_antigravity(seed.Experiment(task='bowl_on_plate',language=''), 'test', 'prompt',str(log),
        {'ROBOARCADE_BASE_PORT':'8100'}, [], [], SimpleNamespace(poll=lambda: None),str(tmp_path),0,result,{})
    assert len(calls) == 2
    assert calls[1][7] == 'same'
    assert calls[1][10] is True
    state = Path(calls[0][6]['ROBOARCADE_AGY_STATE'])
    assert state.is_relative_to('/tmp')
    assert calls[1][6]['ROBOARCADE_AGY_STATE'] == str(state)
    assert not state.exists()
    assert (tmp_path/'antigravity/config/mcp_config.json').is_file()


def test_stream_deltas_and_tool_updates_are_not_duplicated(tmp_path, monkeypatch):
    from core import agent_log
    monkeypatch.setattr(agent_log, 'SHOT_CACHE', tmp_path/'cache')
    path = tmp_path/'antigravity.jsonl'
    def row(index, state, kind, **kwargs):
        return {'event':'step_update','step_update':dict(conversation_id='s',step_index=index,
                    state=state,step_type=kind,**kwargs)}
    tool = {'name':'call_mcp_tool','parameters':{'ServerName':'roboarcade',
            'ToolName':'execute_waypoint','Arguments':{'x':0.1}}}
    rows = [row(1,'ACTIVE','agent_response',text_delta='I will '),
            row(1,'DONE','agent_response',text_delta='move.'),
            row(2,'ACTIVE','tool',tool_info=tool),
            row(2,'DONE','tool',tool_info={**tool,'output':'Moved.'})]
    path.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    parsed = agent_log.transcript(path)
    assert [e['text'] for e in parsed['events'] if e['kind']=='says'] == ['I will move.']
    assert [e['tool'] for e in parsed['events'] if e['kind']=='does'] == ['execute_waypoint']
    assert parsed['n_wp'] == 1
    assert agent_log.transcript_file(tmp_path) == path


def test_missing_result_and_missing_tools_fail_closed(tmp_path):
    log = tmp_path/'agent.jsonl'
    log.write_text(json.dumps({'event': 'init', 'conversation_id': 's'})+'\n')
    assert agy.outcome(log) == ('s', 'Antigravity exited without a result')
    error = {'event': 'step_update', 'step_update': {
        'step_type': 'tool', 'state': 'ERROR', 'tool_info': {
            'name': 'call_mcp_tool', 'error': {'message': 'tool screenshot is not enabled for server roboarcade'}}}}
    with log.open('a') as stream:
        for _ in range(3):
            stream.write(json.dumps(error)+'\n')
    assert 'three rejected calls' in agy.startup_error(log)
    error['step_update']['state'] = 'DONE'
    error['step_update']['tool_info'].pop('error')
    with log.open('a') as stream:
        stream.write(json.dumps(error)+'\n')
    assert agy.startup_error(log) is None


def test_offloaded_images_stay_in_episode_archive(tmp_path, monkeypatch):
    from core import agent_log
    monkeypatch.setattr(agent_log, 'SHOT_CACHE', tmp_path/'cache')
    path = tmp_path/'antigravity.jsonl'
    media = tmp_path/'antigravity/state/brain/session/.system_generated/steps/2/media_0.jpg'
    media.parent.mkdir(parents=True)
    media.write_bytes(b'image bytes')
    def event(relative):
        return {'event': 'step_update', 'step_update': {
            'conversation_id': 's', 'step_index': 2, 'step_type': 'tool', 'state': 'DONE',
            'tool_info': {'name': 'call_mcp_tool', 'parameters': {'ToolName': 'screenshot'},
                          'output': '[Resource offloaded to file:///tmp/home/.gemini/antigravity-cli/'
                                    + relative + ']'}}}
    path.write_text(json.dumps(event('brain/session/.system_generated/steps/2/media_0.jpg'))+'\n')
    assert agent_log.shot_bytes(path, 0) == ('image/jpeg', b'image bytes')
    outside = tmp_path/'private.jpg'
    outside.write_bytes(b'private')
    path.write_text(json.dumps(event('brain/../../../private.jpg'))+'\n')
    assert agent_log.shot_bytes(path, 0) is None


def test_native_usage_survives_missing_final_result(tmp_path):
    path = tmp_path/'agent.jsonl'
    path.write_text('')
    transcript = tmp_path/'brain/session/.system_generated/logs/transcript_full.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({'type': 'PLANNER_RESPONSE', 'input_tokens': 100,
                                      'output_tokens': 20, 'cache_read_tokens': 50})+'\n')
    result = agy.metrics(path, 10, tmp_path)
    assert result['tokens_in'] == 100
    assert result['tokens_out'] == 20
    assert result['cache_read'] == 50
    assert result['num_turns'] == 1
    assert result['cost_usd'] is None


def test_run_summary_does_not_report_unknown_api_cost_as_free(tmp_path):
    from core.run_eval import summarize_and_write
    result = {'success': None, 'metrics': {'tokens_budget': 100, 'cost_usd': None,
                                         'billed_to': 'api_key'}}
    exp = seed.Experiment(task='bowl_on_plate', language='', model='gemini-4-argon')
    summarize_and_write([result], exp, 'smoke', str(tmp_path), False, '/iface')
    meta = json.loads((tmp_path/'meta.json').read_text())
    assert meta['totals']['cost_usd'] is None
    assert meta['totals']['on_api_key'] == 1
    assert meta['effort'] is None
    assert meta['auto_compact_window'] is None


def test_thinking_usage_survives_timeout_without_double_counting(tmp_path):
    path = tmp_path/'agent.jsonl'
    transcript = tmp_path/'brain/session/.system_generated/logs/transcript_full.jsonl'
    transcript.parent.mkdir(parents=True)
    transcript.write_text(json.dumps({'type': 'PLANNER_RESPONSE', 'input_tokens': 100,
                                      'output_tokens': 20, 'cache_read_tokens': 50})+'\n')
    step = {'event': 'step_update', 'step_update': {
        'conversation_id': 'session', 'step_index': 1, 'step_type': 'agent_response',
        'state': 'DONE', 'usage': {'thinking_tokens': 18}}}
    path.write_text((json.dumps(step)+'\n')*2)
    result = agy.metrics(path, 10, tmp_path)
    assert result['tokens']['thinking'] == 18
    assert result['usage']['thinking_tokens'] == 18
    assert result['tokens_out'] == 20
    assert result['tokens_budget'] == 120
    with transcript.open('a') as stream:
        stream.write(json.dumps({'type': 'PLANNER_RESPONSE', 'output_tokens': 10})+'\n')
    assert agy.metrics(path, 10, tmp_path)['tokens']['thinking'] is None


def test_sdk_command_and_metadata_are_explicit(monkeypatch):
    monkeypatch.setenv('ROBOARCADE_ANTIGRAVITY_BACKEND', 'sdk')
    cmd = agy.command(prompt='task', model='gemini-4-argon', session_id='same',
                      base_port=8100, pkg='/iface', resume=True)
    assert 'agy-sdk' in cmd and 'agy' not in cmd
    assert cmd[cmd.index('--conversation')+1] == 'same'
    meta = agy.runtime_metadata()
    assert meta['antigravity_compact_threshold'] == 245000
    assert meta['compaction_token_accounting'] == 'native_estimate'
    monkeypatch.setenv('ROBOARCADE_ANTIGRAVITY_BACKEND', 'cli')
    assert agy.runtime_metadata()['antigravity_compact_threshold'] is None
    monkeypatch.setenv('ROBOARCADE_ANTIGRAVITY_BACKEND', 'typo')
    with pytest.raises(ValueError):
        agy.backend()


def test_sdk_tool_guard_rejects_private_paths_and_other_capabilities(tmp_path, monkeypatch):
    from enum import Enum
    from core import antigravity_sdk as sdk
    monkeypatch.setattr(sdk, 'STATE', tmp_path)
    class Builtin(str, Enum):
        VIEW_FILE = 'view_file'
    assert sdk.tool_name(Builtin.VIEW_FILE) == 'view_file'
    assert sdk.allowed_tool('screenshot', 'roboarcade')
    assert not sdk.allowed_tool('screenshot', 'other')
    assert not sdk.allowed_tool('run_command', None)
    assert sdk.allowed_tool('view_file', None, tmp_path/'brain/session/image.jpg')
    call = SimpleNamespace(canonical_path=None, args={'AbsolutePath': (tmp_path/'brain/image.jpg').as_uri()})
    assert sdk.allowed_tool('view_file', None, sdk.tool_path(call))
    assert not sdk.allowed_tool('view_file', None, 'file://foreign/tmp/image.jpg')
    assert not sdk.allowed_tool('view_file', None, tmp_path/'brain/../sessions/history.db')
    assert not sdk.allowed_tool('view_file', None, '/proc/self/environ')
    (tmp_path/'brain').mkdir()
    (tmp_path/'brain/escape').symlink_to(tmp_path/'sessions')
    assert not sdk.allowed_tool('view_file', None, tmp_path/'brain/escape/private')


def test_sdk_snapshot_resume_and_timeout_accounting(tmp_path):
    from core.antigravity_sdk import usage_record
    usage = usage_record({'prompt_token_count': 100, 'cached_content_token_count': 70,
                          'thoughts_token_count': 20, 'candidates_token_count': 5})
    assert usage == {'input_tokens': 30, 'cache_read_tokens': 70,
                     'thinking_tokens': 20, 'output_tokens': 25}
    path = tmp_path/'agent.jsonl'
    def snapshot(usage, turns):
        return {'event': 'sdk_usage', 'conversation_id': 'same',
                'usage': usage, 'completed_model_steps': turns}
    checkpoint = {'event': 'compaction', 'conversation_id': 'same', 'step_index': 9,
                  'threshold': 245000}
    rows = [snapshot(usage, 1), snapshot(usage, 1),
            {'event':'init', 'conversation_id':'same'}, snapshot(usage, 1),
            checkpoint, checkpoint,
            snapshot({k:v*2 for k,v in usage.items()}, 2)]
    path.write_text('\n'.join(map(json.dumps, rows))+'\n')
    result = agy.metrics(path, 3600)
    assert result['tokens_in'] == 60 and result['tokens_out'] == 50
    assert result['cache_read'] == 140 and result['tokens']['thinking'] == 40
    assert result['num_turns'] is None and result['tokens_budget'] == 110
    assert result['completed_model_steps'] == 2
    assert len(result['compactions']) == 1
    assert agy.outcome(path)[1] == 'Antigravity exited without a result'


def test_sdk_missing_installation_fails_before_simulator(monkeypatch):
    monkeypatch.setenv('ROBOARCADE_ANTIGRAVITY_BACKEND', 'sdk')
    monkeypatch.delenv('ROBOARCADE_AGY_PYTHON', raising=False)
    with pytest.raises(RuntimeError, match='ROBOARCADE_AGY_PYTHON'):
        agy.validate_runtime()


def test_sandbox_waits_for_delayed_host_bridges(tmp_path, monkeypatch):
    import importlib
    import socket
    import threading
    monkeypatch.setenv('ROBOARCADE_EGRESS_HOSTS', 'generativelanguage.googleapis.com')
    bridge = importlib.import_module('core.agent_sandbox.connect_sandbox_to_sim_and_network')
    path = str(tmp_path/'proxy.sock')
    listener = socket.socket(socket.AF_UNIX)
    def bind():
        listener.bind(path)
        listener.listen(1)
    timer = threading.Timer(0.05, bind)
    timer.start()
    try:
        bridge.wait_for_host_sockets([(1, path)], timeout=1)
        timer.join()
        assert Path(path).is_socket()
        with pytest.raises(TimeoutError, match='did not become ready'):
            bridge.wait_for_host_sockets([(1, str(tmp_path/'missing.sock'))], timeout=0.02)
    finally:
        timer.join()
        listener.close()


@pytest.mark.parametrize('error,expected', [
    ('Error 503: service is currently unavailable', True),
    ('Error 500: Internal error encountered', True),
    ('error sending request: http2: server sent GOAWAY', True),
    ('model output must contain either output text or tool calls', True),
    ('Error 403: PERMISSION_DENIED', False),
    ('response blocked by content safety filters: OTHER', False),
    ('quota exhausted', False),
    ('unknown driver failure', False),
])
def test_provider_retry_classification(error, expected):
    assert agy.retryable_error(error) is expected


def test_provider_retry_preserves_state_and_episode_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(seed.time, 'time', lambda: clock[0])
    monkeypatch.setattr(seed.SHUTDOWN, 'wait', lambda delay: clock.__setitem__(0, clock[0]+delay))
    monkeypatch.setattr(seed, 'episode_over', lambda *a: False)
    log = tmp_path/'agent.jsonl'
    calls = []
    def run(*args):
        calls.append(args)
        log.write_text(json.dumps({'event':'result','result': {
            'status':'ERROR','conversation_id':'same', 'error':'Error 503: unavailable'}})+'\n')
        return 'agent_error' if len(calls) == 1 else 'timeout'
    monkeypatch.setattr(seed, 'run_agent', run)
    result = {}
    status, error = seed.drive_antigravity(seed.Experiment(task='bowl_on_plate', language='', episode_timeout=60),
        'test', 'prompt',str(log),{'ROBOARCADE_BASE_PORT':'8100'},[],[],
        SimpleNamespace(poll=lambda: None),str(tmp_path),0,result,{})
    assert status == 'timeout' and error is None
    assert len(calls) == 2
    assert calls[1][7] == 'same' and calls[1][10] is True
    assert calls[1][6]['ROBOARCADE_AGY_STATE'] == calls[0][6]['ROBOARCADE_AGY_STATE']
    assert calls[0][4] == 60 and calls[1][4] == 45
    assert len(result['provider_retries']) == 1


def test_provider_retry_stops_after_three_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr(seed.SHUTDOWN, 'wait', lambda delay: False)
    monkeypatch.setattr(seed, 'episode_over', lambda *a: False)
    log = tmp_path/'agent.jsonl'
    log.write_text(json.dumps({'event':'result','result': {
        'status':'ERROR','conversation_id':'same','error':'Error 500: Internal'}})+'\n')
    calls = []
    monkeypatch.setattr(seed, 'run_agent', lambda *a: calls.append(a) or 'agent_error')
    result = {}
    status, error = seed.drive_antigravity(seed.Experiment(task='bowl_on_plate', language=''),
        'test','prompt',str(log),{'ROBOARCADE_BASE_PORT':'8100'},[],[],
        SimpleNamespace(poll=lambda: None),str(tmp_path),0,result,{})
    assert status == 'agent_error' and error == 'Error 500: Internal'
    assert len(calls) == 4 and len(result['provider_retries']) == 3
