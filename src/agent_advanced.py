from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Offline agent with per-thread compact memory and persistent user profiles."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        answer = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", answer)
        generated = estimate_tokens(answer)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + generated
        return {"answer": answer, "response": answer, "token_usage": generated,
                "prompt_tokens_processed": self.thread_prompt_tokens[thread_id]}

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        profile = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context["summary"])
        messages = context["messages"]
        recent = "\n".join(item["content"] for item in messages)  # type: ignore[index]
        return estimate_tokens(profile) + estimate_tokens(summary) + estimate_tokens(recent)

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        facts = self.profile_store.facts(user_id)
        q = message.casefold()
        requested: list[tuple[str, str]] = []
        requests = [
            (("tên", "là ai", "biết "), "name", "Tên"),
            (("nghề", "công việc", "làm gì"), "profession", "Nghề nghiệp hiện tại"),
            (("ở đâu", "nơi ở", "đang ở", "còn ở", "chuyển đi"), "location", "Nơi ở hiện tại"),
            (("style", "phong cách", "trả lời"), "response style", "Style trả lời"),
            (("đồ uống", "uống", "cà phê"), "favorite drink", "Đồ uống yêu thích"),
            (("món ăn", "món yêu thích", "ăn gì"), "favorite food", "Món ăn yêu thích"),
            (("con gì", "thú cưng", "nuôi"), "pet", "Thú cưng"),
            (("quan tâm", "sở thích", "thích gì"), "interests", "Mối quan tâm"),
        ]
        for triggers, key, label in requests:
            if any(trigger in q for trigger in triggers) and facts.get(key):
                requested.append((label, facts[key]))
        if requested:
            return "\n".join(f"- {label}: {value}" for label, value in requested)

        context = self.compact_memory.context(thread_id)
        summary = str(context["summary"])
        if any(term in q for term in ("artemis", "x-59", "el nino", "wmo", "british columbia", "điện sạch")):
            snippets = [line.removeprefix("User: ") for line in summary.splitlines()
                        if any(term in line.casefold() for term in ("artemis", "x-59", "el nino", "wmo", "british columbia", "điện sạch"))]
            if snippets:
                return "- Mình nhớ các mốc chính: " + " ".join(snippets[:3])
        if facts:
            return "- Mình đã lưu hồ sơ và có thể nhắc lại tên, nghề nghiệp, nơi ở hoặc sở thích khi bạn hỏi cụ thể."
        return "- Mình chưa có đủ thông tin để trả lời câu này."

    def _maybe_build_langchain_agent(self):
        """Optional live provider factory; deterministic lab runs stay offline."""
        if self.force_offline:
            return None
        return build_chat_model(self.config.model)
