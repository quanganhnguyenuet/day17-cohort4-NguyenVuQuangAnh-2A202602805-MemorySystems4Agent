from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Return a stable, deliberately simple estimate for English and Vietnamese text."""
    cleaned = text.strip()
    return 0 if not cleaned else max(1, (len(cleaned) + 3) // 4)


@dataclass
class UserProfileStore:
    """Store one structured Markdown profile per user."""

    root_dir: Path

    @staticmethod
    def _slug(user_id: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(user_id)).encode("ascii", "ignore").decode()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", normalized).strip("_.-")
        return slug[:80] or "default"

    def path_for(self, user_id: str) -> Path:
        return self.root_dir / self._slug(user_id) / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return "# User profile\n\n"
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"^-\s+([^:]+):\s*(.*)$", line)
            if match:
                facts[match.group(1).strip()] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        facts = self.facts(user_id)
        if key == "interests" and facts.get(key) and value.casefold() not in facts[key].casefold():
            value = (facts[key] + "; " + value)[:400]
        if facts.get(key) == value:
            return
        facts[key] = value
        lines = ["# User profile", "", "## Stable facts"]
        lines.extend(f"- {name}: {facts[name]}" for name in sorted(facts))
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Extract high-confidence facts from common Vietnamese self-description patterns."""
    text = re.sub(r"\s+", " ", message).strip()
    if not text or "?" in text or "？" in text or re.search(r"\b(?:là gì|ở đâu|con gì|bao nhiêu|đâu mới là)\b", text, re.IGNORECASE):
        return {}
    updates: dict[str, str] = {}

    patterns: dict[str, list[str]] = {
        "name": [r"\bmình\s+tên\s+là\s+([^,.!?;]+)", r"\btên\s+mình\s+là\s+([^,.!?;]+)"],
        "location": [
            r"\b(?:giờ|hiện tại|hiện)\s+(?:mình\s+)?(?:đang\s+)?(?:ở|sống\s+ở|làm việc ở)\s+([^,.!?;]+)",
            r"\bmình\s+(?:đang\s+)?(?:ở|sống\s+ở|làm việc ở)\s+([^,.!?;]+)",
        ],
        "profession": [
            r"\b(?:giờ|hiện tại|hiện)\s+(?:mình\s+)?(?:chuyển sang|làm|đang làm)\s+([^,.!?;]+)",
            r"\bmình\s+(?:đang làm|làm nghề|làm)\s+([^,.!?;]+)",
            r"\b(?:nghề nghiệp|công việc)(?:\s+hiện tại)?\s+(?:thì\s+)?(?:vẫn\s+)?là\s+([^,.!?;]+)",
        ],
        "favorite drink": [r"\b(?:đồ uống yêu thích|món uống yêu thích)\s+(?:của mình\s+)?(?:là|mình thích)\s+([^,.!?;]+)"],
        "favorite food": [r"\b(?:món ăn yêu thích|món khoái khẩu)\s+(?:của mình\s+)?(?:là|mình thích)\s+([^,.!?;]+)"],
        "response style": [
            r"\b(?:trả lời|giải thích)\s+(?:thành\s+)?(?:theo\s+)?(3\s+bullet[^.!?;]*|ngắn gọn[^.!?;]*|thành bullet[^.!?;]*)",
            r"\bstyle trả lời[^.!?;]*?(ngắn gọn[^.!?;]*)",
        ],
        "interests": [
            r"\bmình\s+(?:đang\s+)?quan tâm\s+(?:nhiều\s+)?(?:đến|tới)\s+([^.!?;]+)",
            r"\bmình\s+thích\s+((?:Python|AI|MLOps|RAG|async Python)[^.!?;]*)",
        ],
    }
    for key, expressions in patterns.items():
        candidates: list[tuple[int, str]] = []
        for expression in expressions:
            candidates.extend((match.start(), match.group(1)) for match in re.finditer(expression, text, flags=re.IGNORECASE))
        # Corrections in a message can match a different expression from the old
        # fact.  Use the latest mention in the text, not the first pattern listed.
        for _, raw_value in sorted(candidates, reverse=True):
            value = raw_value.strip(" ,")
            value = re.split(r"\s+(?:chứ|nhưng|vì|để|còn|và|dù)\s+", value, maxsplit=1, flags=re.IGNORECASE)[0].strip()
            if value.casefold() in {"gì", "ai", "đâu", "nào"}:
                continue
            # Explicit joke/travel mentions are not facts about a current job or home.
            if key == "profession" and any(x in value.casefold() for x in ("hay là", "đùa", "đùa rằng")):
                continue
            if key == "profession" and value.casefold().startswith(("việc ở ", "việc tại ")):
                continue
            if key == "location" and any(x in value.casefold() for x in ("chỉ là", "vừa bay", "đi họp", "đi du lịch")):
                continue
            if key == "response style" and "ngắn" in value.casefold() and "ngắn gọn" not in value.casefold():
                value = "ngắn gọn, " + value
            if value:
                updates[key] = value
                break

    # Explicit pet statements are stable facts and useful for recall questions.
    pet = re.search(r"\bmình\s+nuôi\s+(?:một\s+)?(?:bé\s+)?([^,.!?;]+)", text, flags=re.IGNORECASE)
    if pet:
        updates["pet"] = pet.group(1).strip()
    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Keep compact, bounded snippets from older turns for later follow-up."""
    snippets: list[str] = []
    for item in messages:
        content = re.sub(r"\s+", " ", item.get("content", "")).strip()
        if not content:
            continue
        prefix = "User" if item.get("role") == "user" else "Assistant"
        if len(content) > 230:
            content = content[:227].rsplit(" ", 1)[0] + "..."
        snippets.append(f"{prefix}: {content}")
    # A summary is a rolling note, so cap its size even after repeated compactions.
    return "\n".join(snippets[-max_items:])[:1800]


@dataclass
class CompactMemoryManager:
    """Keep recent messages verbatim and roll older messages into a bounded summary."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})
        summary = str(thread.get("summary", ""))
        total = estimate_tokens(summary) + sum(estimate_tokens(m["content"]) for m in messages)
        if total > max(1, self.threshold_tokens) and len(messages) > max(1, self.keep_messages):
            keep = max(1, self.keep_messages)
            older = messages[:-keep]
            previous = ([{"role": "assistant", "content": summary}] if summary else [])
            thread["summary"] = summarize_messages(previous + older, max_items=8)
            thread["messages"] = messages[-keep:]
            thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        thread = self.state.get(thread_id)
        if thread is None:
            return {"messages": [], "summary": "", "compactions": 0}
        return {"messages": list(thread["messages"]), "summary": thread["summary"], "compactions": thread["compactions"]}

    def compaction_count(self, thread_id: str) -> int:
        return int(self.state.get(thread_id, {}).get("compactions", 0))
