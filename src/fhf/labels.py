"""Exact competition labels and provider identities; never infer historical labels."""
MODEL_TO_FAMILY = {
    "gpt-5.5":"OpenAI", "gpt-5.6-luna":"OpenAI", "gpt-5.6-sol":"OpenAI",
    "gpt-5.6-terra":"OpenAI", "gpt-6-astra":"OpenAI",
    "gemini-3.1-pro-preview":"Google", "gemini-3.6-flash":"Google",
    "gemini-3.7-flash":"Google", "gemini-3.8-flash":"Google",
    "claude-fable-5":"Anthropic", "claude-opus-4-6":"Anthropic",
    "claude-opus-4-8":"Anthropic", "claude-opus-5":"Anthropic",
    "claude-sonnet-5":"Anthropic", "qwen3.8-max":"Qwen",
    "glm-5.3":"ZAI", "glm-5.3-flash":"ZAI", "kimi-k3":"Moonshot AI",
}
MODEL_NAMES = list(MODEL_TO_FAMILY)
FAMILY_NAMES = list(dict.fromkeys(MODEL_TO_FAMILY.values()))
MODEL_TO_ID = {name: i for i, name in enumerate(MODEL_NAMES)}
FAMILY_TO_ID = {name: i for i, name in enumerate(FAMILY_NAMES)}
MODEL_FAMILY_IDS = [FAMILY_TO_ID[MODEL_TO_FAMILY[k]] for k in MODEL_NAMES]

def check_labels(values):
    """Fail loudly on labels outside the official exact 18-class vocabulary."""
    extra = set(values) - set(MODEL_NAMES)
    if extra:
        raise ValueError(f"Unknown model labels: {sorted(extra)}")
