from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _provider_config(prefix: str, default_provider: str, default_model: str) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    key_names = {
        "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", "openrouter": "OPENROUTER_API_KEY",
        "custom": "CUSTOM_API_KEY",
    }
    url_names = {"ollama": "OLLAMA_BASE_URL", "custom": "CUSTOM_BASE_URL", "openrouter": "OPENROUTER_BASE_URL"}
    key = os.getenv(f"{prefix}_API_KEY") or os.getenv(key_names.get(provider, "")) or None
    url = os.getenv(f"{prefix}_BASE_URL") or os.getenv(url_names.get(provider, "")) or None
    return ProviderConfig(
        provider=provider,
        model_name=os.getenv(f"{prefix}_MODEL", default_model),
        temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0")),
        api_key=key,
        base_url=url,
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass
    state_dir = Path(os.getenv("LAB_STATE_DIR", root / "state")).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(
        base_dir=root,
        data_dir=Path(os.getenv("LAB_DATA_DIR", root / "data")).resolve(),
        state_dir=state_dir,
        compact_threshold_tokens=max(1, int(os.getenv("COMPACT_THRESHOLD_TOKENS", "900"))),
        compact_keep_messages=max(2, int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))),
        model=_provider_config("LLM", "openai", "gpt-4o-mini"),
        judge_model=_provider_config("JUDGE", "openai", "gpt-4o-mini"),
    )
