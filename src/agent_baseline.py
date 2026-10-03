from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Deterministic offline baseline with memory limited to each thread."""

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})
        session.prompt_tokens_processed += sum(estimate_tokens(m["content"]) for m in session.messages)
        facts: dict[str, str] = {}
        for item in session.messages:
            if item["role"] == "user":
                facts.update(extract_profile_updates(item["content"]))
        answer = self._answer_from_thread(facts, message)
        session.messages.append({"role": "assistant", "content": answer})
        generated = estimate_tokens(answer)
        session.token_usage += generated
        return {"answer": answer, "response": answer, "token_usage": generated,
                "prompt_tokens_processed": session.prompt_tokens_processed}

    @staticmethod
    def _answer_from_thread(facts: dict[str, str], message: str) -> str:
        q = message.casefold()
        wanted: list[tuple[str, str]] = []
        if any(k in q for k in ("tên", "là ai", "biết ")) and facts.get("name"):
            wanted.append(("Tên", facts["name"]))
        if any(k in q for k in ("nghề", "công việc", "làm gì")) and facts.get("profession"):
            wanted.append(("Nghề nghiệp", facts["profession"]))
        if any(k in q for k in ("ở đâu", "nơi ở", "đang ở")) and facts.get("location"):
            wanted.append(("Nơi ở", facts["location"]))
        if any(k in q for k in ("style", "phong cách", "trả lời")) and facts.get("response style"):
            wanted.append(("Style", facts["response style"]))
        if any(k in q for k in ("uống", "đồ uống")) and facts.get("favorite drink"):
            wanted.append(("Đồ uống yêu thích", facts["favorite drink"]))
        if any(k in q for k in ("món ăn", "ăn gì")) and facts.get("favorite food"):
            wanted.append(("Món ăn yêu thích", facts["favorite food"]))
        if any(k in q for k in ("con gì", "nuôi")) and facts.get("pet"):
            wanted.append(("Thú cưng", facts["pet"]))
        if wanted:
            return "Mình nhớ trong cuộc trò chuyện này: " + "; ".join(f"{key}: {value}" for key, value in wanted) + "."
        if "nhớ" in q and not facts:
            return "Mình sẽ giữ thông tin trong cuộc trò chuyện hiện tại."
        return "Mình đã ghi nhận. Bạn muốn mình giúp gì tiếp theo?"

    def _maybe_build_langchain_agent(self):
        """Optional live model factory; the benchmark uses offline mode by default."""
        if self.force_offline:
            return None
        return build_chat_model(self.config.model)
