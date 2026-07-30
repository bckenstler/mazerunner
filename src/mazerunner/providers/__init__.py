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
