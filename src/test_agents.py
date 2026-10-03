from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import extract_profile_updates


def make_config(tmp_path: Path) -> LabConfig:
    defaults = load_config(tmp_path)
    defaults.compact_threshold_tokens = 100
    defaults.compact_keep_messages = 2
    return defaults


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    store = agent.profile_store
    assert store.read_text("student") == "# User profile\n\n"
    store.write_text("student", "# User profile\n\n- city: Huế")
    assert store.edit_text("student", "Huế", "Đà Nẵng")
    assert "Đà Nẵng" in store.read_text("student")
    assert store.file_size("student") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for index in range(8):
        agent.reply("student", "long-thread", f"Thông tin dài lượt {index}: " + "chi tiết " * 30)
    assert agent.compaction_count("long-thread") > 0
    context = agent.compact_memory.context("long-thread")
    assert context["summary"]
    assert len(context["messages"]) <= 2


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    advanced.reply("student", "session-a", "Mình tên là Lan và mình ở Huế.")
    baseline.reply("student", "session-a", "Mình tên là Lan và mình ở Huế.")
    advanced_answer = advanced.reply("student", "session-b", "Mình tên gì và đang ở đâu?")["answer"]
    baseline_answer = baseline.reply("student", "session-b", "Mình tên gì và đang ở đâu?")["answer"]
    assert "Lan" in advanced_answer and "Huế" in advanced_answer
    assert "Lan" not in baseline_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    advanced = AdvancedAgent(config, force_offline=True)
    baseline = BaselineAgent(config, force_offline=True)
    for index in range(12):
        message = f"Lượt {index}: " + "nội dung benchmark có nhiều chi tiết " * 24
        advanced.reply("student", "same-thread", message)
        baseline.reply("student", "same-thread", message)
    assert advanced.compaction_count("same-thread") > 0
    assert advanced.prompt_token_usage("same-thread") < baseline.prompt_token_usage("same-thread")


def test_correction_uses_latest_location_in_same_message() -> None:
    message = "Lúc đầu mình nói hiện ở Huế, nhưng giờ mình đang ở Đà Nẵng."
    assert extract_profile_updates(message)["location"] == "Đà Nẵng"


def test_correction_replaces_old_persisted_fact(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("student", "session-a", "Mình đang ở Huế.")
    agent.reply("student", "session-b", "Giờ mình đang ở Đà Nẵng.")
    profile = agent.profile_store.read_text("student")
    assert "- location: Đà Nẵng" in profile
    assert "Huế" not in profile


def test_unrelated_action_does_not_replace_profession(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("student", "session-a", "Mình đang làm MLOps engineer.")
    agent.reply("student", "session-a", "Nếu compact đang làm đúng việc của nó thì ổn.")
    assert agent.profile_store.facts("student")["profession"] == "MLOps engineer"
    assert "MLOps engineer" in agent.reply("student", "session-b", "Mình làm nghề gì?")["answer"]


def test_question_does_not_create_profession(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    agent.reply("student", "session-a", "Mình làm product manager à?")
    assert "profession" not in agent.profile_store.facts("student")
