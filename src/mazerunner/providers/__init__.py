"""Provider registry: the two dispatch tables the runner resolves against.

`PROVIDERS` maps a config's `type` to its adapter class; `ENV_KEYS` gives the
default environment variable each first-party provider reads its key from. Keys
are named here and read from the environment, never stored in a config file —
which is why a config can be committed and a key cannot.

Any OpenAI-compatible gateway is reachable without a new adapter: set
`type: "openai_compat"` with a `base_url`.
"""

from . import anthropic_provider, gemini_provider, openai_compat, openai_provider

PROVIDERS = {
    openai_provider.NAME: openai_provider.OpenAIProvider,
    anthropic_provider.NAME: anthropic_provider.AnthropicProvider,
    gemini_provider.NAME: gemini_provider.GeminiProvider,
    openai_compat.TYPE_NAME: openai_compat.OpenAICompatProvider,
}

ENV_KEYS = {
    openai_provider.NAME: openai_provider.ENV_KEY,
    anthropic_provider.NAME: anthropic_provider.ENV_KEY,
    gemini_provider.NAME: gemini_provider.ENV_KEY,
}
