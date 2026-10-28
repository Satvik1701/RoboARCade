"""API-equivalent benchmark costs at the documented October 26, 2026 rates.

These are standard Gemini Developer API rates, not Batch/Flex/Priority prices
or an invoice. Keep the reference date with exported estimates. Introductory
Argon rates end November 30, 2026; a later benchmark needs a new price snapshot.
"""
import math

REFERENCE_DATE = "2026-10-26"
SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-4-argon"
ARGON_RATES = {"input": 2.0, "cached_input": 0.10, "output": 10.0}


def cost_usd(input_tokens, cached_input_tokens, output_tokens, *, model="gemini-4-argon"):
    """Input excludes cached tokens; output already includes thinking tokens."""
    counts = (input_tokens, cached_input_tokens, output_tokens)
    if model != "gemini-4-argon" or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
        for v in counts
    ):
        return None
    return sum(n * ARGON_RATES[k] for n, k in zip(counts, ("input", "cached_input", "output"))) / 1e6


def cost_basis():
    return {"source": SOURCE, "reference_date": REFERENCE_DATE,
            "usd_per_million_tokens": dict(ARGON_RATES), "tier": "standard",
            "introductory_rates_through": "2026-11-30", "estimated": True,
            "input_excludes_cached": True, "output_includes_thinking": True}
