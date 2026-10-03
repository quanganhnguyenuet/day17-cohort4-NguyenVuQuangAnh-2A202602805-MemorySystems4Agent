from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower().replace("-", "_")
    aliases = {"anthorpic": "anthropic", "google": "gemini", "google_genai": "gemini", "local": "ollama"}
    provider = aliases.get(provider, provider)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if provider not in supported:
        raise ValueError(f"Unsupported LLM provider {value!r}; choose one of {', '.join(sorted(supported))}.")
    return provider


def build_chat_model(config: ProviderConfig):
    """Create a LangChain chat model lazily, with a focused missing-package error."""
    provider = normalize_provider(config.provider)
    options = {"model": config.model_name, "temperature": config.temperature}
    if provider in {"openai", "custom"}:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-openai to use the OpenAI or custom provider.") from exc
        if config.api_key:
            options["api_key"] = config.api_key
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOpenAI(**options)
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install langchain-google-genai to use the Gemini provider.") from exc
        options["model"] = config.model_name
        if config.api_key:
            options["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**options)
    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install langchain-anthropic to use the Anthropic provider.") from exc
        options["model"] = config.model_name
        if config.api_key:
            options["api_key"] = config.api_key
        return ChatAnthropic(**options)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise RuntimeError("Install langchain-ollama to use the Ollama provider.") from exc
        options["model"] = config.model_name
        if config.base_url:
            options["base_url"] = config.base_url
        return ChatOllama(**options)
    try:
        from langchain_openrouter import ChatOpenRouter
    except ImportError as exc:
        raise RuntimeError("Install langchain-openrouter to use the OpenRouter provider.") from exc
    options["model"] = config.model_name
    if config.api_key:
        options["api_key"] = config.api_key
    if config.base_url:
        options["base_url"] = config.base_url
    return ChatOpenRouter(**options)
