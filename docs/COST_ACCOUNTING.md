# Cost accounting

The dashboard reports the mean cost per graded trial. The **Success only** columns
select trials passing the same binary success and invariant checks as the success
column. **All** includes successful and failed graded trials, including timeouts.
Missing measurements are excluded from the mean and their coverage is shown.

Astra and Sol dollars are **Standard API-equivalent estimates**, not subscription
payments. The calculation uses the recorded per-request token counts and the
repository's pinned rates. It does not estimate a fraction of the plan price or
apply the API Fast-mode surcharge.

Rates verified against [OpenAI's pricing page](https://developers.openai.com/api/docs/pricing)
and [Astra's model page](https://developers.openai.com/api/docs/models/gpt-6-astra)
on 2026-09-22:

| Astra tokens | Standard dollars per million |
|---|---:|
| Uncached input | 10 |
| Cached input | 1 |
| Cache writes, when reported | 12.50 |
| Output | 50 |

For each request, let `I` be total input tokens, `R` cached input tokens, `W` cache
write tokens and `O` output tokens. Set `U = max(0, I - R - W)`.

```
request cost = [input_multiplier × (10U + R + 12.5W)
                + output_multiplier × 50O] / 1,000,000
trial cost = sum(request cost)
```

The multipliers are 1 and 1 when `I <= 272000`. For `I > 272000`, they are 2 and
1.5 respectively, applying to the full request. This threshold is per request,
not the sum of tokens across a trial. Reasoning tokens are already included in
reported output tokens and are not added again.

For Beam seed 1000005 in `beam_rerun_5x_astra/20260914_180900`, no request crosses
the threshold. Distinct native usage records give 774,091 uncached input,
41,492,864 cached input, zero cache writes, and 67,069 output tokens:

```
(10 × 774091 + 1 × 41492864 + 12.5 × 0 + 50 × 67069) / 1000000
= $52.587224 ≈ $52.59
```

The original saved estimate was $52.7990 because a quota notification repeated a
previous request's usage. The shared parser now ignores repeated cumulative
usage counters; a genuine new request with identical token counts still counts
when the cumulative counter advances.

The audited Codex costs are written into each affected episode's episode.json (metrics.cost_usd); the original file is
kept beside it as episode.json.before_migration.

Claude costs retain their recorded native estimates. Legacy runs without
per-trial costs may use a run-total allocation; the table labels those allocations.
