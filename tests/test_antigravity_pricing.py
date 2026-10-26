import json
import pytest
from core.antigravity_pricing import cost_usd, cost_basis
from core.antigravity import metrics


def test_prices_cached_and_uncached_input_separately():
    assert cost_usd(1_000_000, 1_000_000, 1_000_000) == 12.1
    assert cost_usd(0, 1_000_000, 0) == .1
    assert cost_usd(0, 0, 0) == 0
    assert cost_basis()['reference_date'] == '2026-10-26'


@pytest.mark.parametrize('value', [None, -1, float('nan'), float('inf'), True, '100'])
def test_unknown_or_invalid_usage_is_not_free(value):
    assert cost_usd(value, 0, 0) is None
    assert cost_usd(0, value, 0) is None
    assert cost_usd(0, 0, value) is None
    assert cost_usd(1, 2, 3, model='another-model') is None


def test_sdk_repeated_usage_and_thinking_are_not_charged_twice(tmp_path):
    path=tmp_path/'antigravity.jsonl'
    rows=[{'event':'sdk_usage','conversation_id':'s','usage':{'input_tokens':1_000_000,
           'cache_read_tokens':2_000_000,'output_tokens':100_000,'thinking_tokens':80_000}}]*3
    path.write_text('\n'.join(json.dumps(row) for row in rows))
    actual=metrics(path,10,model='gemini-4-argon')
    assert actual['cost_usd']==3.2
    assert actual['cost_estimated']
    assert actual['tokens_out']==100_000


def test_native_fallback_includes_compaction_without_counting_it_as_a_model_turn(tmp_path):
    path=tmp_path/'antigravity.jsonl';path.write_text('')
    native=tmp_path/'brain/session/.system_generated/logs/transcript_full.jsonl'
    native.parent.mkdir(parents=True)
    native.write_text('\n'.join(json.dumps({'type':kind,'input_tokens':1_000_000,
                      'cache_read_tokens':0,'output_tokens':100_000})
                      for kind in ['PLANNER_RESPONSE','CHECKPOINT']))
    actual=metrics(path,10,tmp_path,model='gemini-4-argon')
    assert actual['cost_usd']==6
    assert actual['num_turns']==1
