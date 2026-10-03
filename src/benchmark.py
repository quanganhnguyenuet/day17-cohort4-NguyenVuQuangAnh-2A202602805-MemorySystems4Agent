from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list of conversations in {path}")
    return value


def recall_points(answer: str, expected: list[str]) -> float:
    if not expected:
        return 1.0
    normalized = answer.casefold()
    matched = sum(1 for item in expected if item.casefold() in normalized)
    ratio = matched / len(expected)
    return 1.0 if ratio == 1 else (0.5 if ratio >= 0.5 else 0.0)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline proxy: reward a substantive, relevant answer without an LLM judge."""
    if not answer.strip():
        return 0.0
    coverage = sum(1 for item in expected if item.casefold() in answer.casefold()) / max(1, len(expected))
    substance = min(1.0, len(answer.strip()) / 45)
    return round(0.8 * coverage + 0.2 * substance, 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    users = {str(item["user_id"]) for item in conversations}
    profile_store = getattr(agent, "profile_store", None)
    before_sizes = {user: profile_store.file_size(user) for user in users} if profile_store else {}
    recall_scores: list[float] = []
    quality_scores: list[float] = []
    all_threads: list[str] = []

    for conversation in conversations:
        user_id = str(conversation["user_id"])
        thread_id = f"{conversation['id']}-main"
        all_threads.append(thread_id)
        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, str(turn))
        for index, item in enumerate(conversation.get("recall_questions", [])):
            recall_thread = f"{conversation['id']}-recall-{index}"
            all_threads.append(recall_thread)
            answer = str(agent.reply(user_id, recall_thread, item["question"])["answer"])
            expected = list(item.get("expected_contains", []))
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    tokens = sum(agent.token_usage(thread) for thread in all_threads)
    prompt_tokens = sum(agent.prompt_token_usage(thread) for thread in all_threads)
    compactions = sum(agent.compaction_count(thread) for thread in all_threads)
    growth = sum(profile_store.file_size(user) - before_sizes[user] for user in users) if profile_store else 0
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        response_quality=sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    headers = ["Agent", "Agent tokens only", "Prompt tokens processed", "Cross-session recall",
               "Response quality", "Memory growth (bytes)", "Compactions"]
    body = [[row.agent_name, str(row.agent_tokens_only), str(row.prompt_tokens_processed),
             f"{row.recall_score:.1%}", f"{row.response_quality:.1%}", str(row.memory_growth_bytes), str(row.compactions)]
            for row in rows]
    widths = [max(len(headers[i]), *(len(line[i]) for line in body)) for i in range(len(headers))]
    render = lambda line: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(line)) + " |"
    return "\n".join([render(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(render(line) for line in body)])


def _run_suite(title: str, conversations: list[dict[str, Any]], config) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for name, agent_type in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        with tempfile.TemporaryDirectory(prefix="day17-benchmark-") as temporary_state:
            isolated_config = replace(config, state_dir=Path(temporary_state))
            agent = agent_type(isolated_config, force_offline=True)
            rows.append(run_agent_benchmark(name, agent, conversations, isolated_config))
    print(f"\n## {title}\n")
    print(format_rows(rows))
    print("\nResponse quality is a deterministic heuristic based on expected-fact coverage and answer substance.")
    return rows


def main() -> None:
    config = load_config(Path(__file__).resolve().parent.parent)
    standard = load_conversations(config.data_dir / "conversations.json")
    stress = load_conversations(config.data_dir / "advanced_long_context.json")
    _run_suite("Standard Benchmark", standard, config)
    _run_suite("Long-Context Stress Benchmark", stress, config)


if __name__ == "__main__":
    main()
