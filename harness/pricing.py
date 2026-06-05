"""
FinCodeBench pricing
Per-model token pricing and USD cost estimation, so a run can report what it cost.

Prices are USD per 1,000,000 tokens, (input, output). Sourced from Anthropic's
published model pricing. Cache tokens are priced relative to the base input rate
(writes ~1.25x, reads ~0.1x); this suite doesn't use prompt caching today, but the
math is here so the numbers stay correct if it ever does.
"""

# Exact model id → (input_per_mtok, output_per_mtok) in USD.
MODEL_PRICING = {
    "claude-opus-4-8":   (5.0, 25.0),
    "claude-opus-4-7":   (5.0, 25.0),
    "claude-opus-4-6":   (5.0, 25.0),
    "claude-opus-4-5":   (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5":  (1.0, 5.0),
}

# Fallback by family when an exact id isn't listed (e.g. a dated snapshot).
_FAMILY_PRICING = {
    "opus":   (5.0, 25.0),
    "sonnet": (3.0, 15.0),
    "haiku":  (1.0, 5.0),
}

_CACHE_WRITE_MULT = 1.25   # cache_creation_input_tokens, relative to input rate
_CACHE_READ_MULT = 0.1     # cache_read_input_tokens, relative to input rate


def price_for(model: str):
    """Return (input_per_mtok, output_per_mtok) for a model id, or None if unknown."""
    if not model:
        return None
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]
    for family, price in _FAMILY_PRICING.items():
        if family in model:
            return price
    return None


def compute_cost(model: str, input_tokens: int = 0, output_tokens: int = 0,
                 cache_creation_tokens: int = 0, cache_read_tokens: int = 0):
    """
    Estimate the USD cost of a set of token counts for a model.
    Returns a float rounded to 6 dp, or None when the model's price is unknown
    (so callers can render "—" rather than a misleading $0.00).
    """
    price = price_for(model)
    if price is None:
        return None
    in_rate, out_rate = price
    cost = (
        input_tokens * in_rate
        + output_tokens * out_rate
        + cache_creation_tokens * in_rate * _CACHE_WRITE_MULT
        + cache_read_tokens * in_rate * _CACHE_READ_MULT
    ) / 1_000_000
    return round(cost, 6)
