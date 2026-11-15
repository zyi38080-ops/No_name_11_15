import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import noname.strategy_memory as strategy_memory
from noname.planner import StrategyPlanner
from noname.schemas import ConversationAnalysis, ConversationStage, MIStrategy, SessionState
from noname.storage import SQLiteSessionStore
from noname.strategy_memory import (
    DEFAULT_CASES,
    load_strategy_memory,
    retrieve_strategy_memories,
    score_case,
)


def analysis(**kwargs) -> ConversationAnalysis:
    kwargs.setdefault("summary", "测试分析")
    return ConversationAnalysis(**kwargs)


# --- loading ----------------------------------------------------------------


def test_seed_memory_covers_every_ordinary_stage() -> None:
    stages = {case.stage for case in DEFAULT_CASES}

    assert ConversationStage.ENGAGE in stages
    assert ConversationStage.FOCUS in stages
    assert ConversationStage.EVOKE in stages
    assert ConversationStage.PLAN in stages
    assert ConversationStage.REVIEW in stages
    assert len(DEFAULT_CASES) >= 15


def test_shipped_json_matches_the_built_in_seed() -> None:
    """The JSON is meant to be a hand-editable mirror of the seed cases."""

    path = strategy_memory.strategy_memory_file()
    if not path.exists():  # pragma: no cover - file is part of the repo
        return

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert len(payload["cases"]) == len(DEFAULT_CASES)
    assert {case["id"] for case in payload["cases"]} == {case.id for case in DEFAULT_CASES}


def test_loading_is_cached_and_returns_seed_cases() -> None:
    load_strategy_memory.cache_clear()

    assert load_strategy_memory() is load_strategy_memory()
    assert load_strategy_memory()


def test_malformed_memory_file_degrades_to_built_in_seed(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "strategy_memory.json"
    broken.write_text("{ this is not valid json", encoding="utf-8")
    monkeypatch.setattr(strategy_memory, "strategy_memory_file", lambda: broken)
    load_strategy_memory.cache_clear()
    try:
        assert len(load_strategy_memory()) == len(DEFAULT_CASES)
    finally:
        load_strategy_memory.cache_clear()


def test_a_single_invalid_case_does_not_disable_the_memory(tmp_path, monkeypatch) -> None:
    path = tmp_path / "strategy_memory.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "cases": [
                    {"id": "good_case", "preferred_strategies": ["SUMMARY"]},
                    {"id": "bad_case", "preferred_strategies": ["NOT_A_STRATEGY"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(strategy_memory, "strategy_memory_file", lambda: path)
    load_strategy_memory.cache_clear()
    try:
        loaded = load_strategy_memory()
        assert [case.id for case in loaded] == ["good_case"]
    finally:
        load_strategy_memory.cache_clear()


# --- matching ---------------------------------------------------------------


def test_ambivalence_turn_matches_the_ambivalence_case_first() -> None:
    matches = retrieve_strategy_memories(
        SessionState(session_id="s"),
        analysis(
            stage=ConversationStage.EVOKE,
            change_talk=["希望能早点停"],
            sustain_talk=["但也不想少玩"],
        ),
    )

    assert matches
    assert matches[0].case_id == "evoke_ambivalence_01"
    assert MIStrategy.DOUBLE_SIDED_REFLECTION in matches[0].preferred_strategies
    assert MIStrategy.ACTION_PLANNING in matches[0].avoid_strategies


def test_matching_is_reported_with_an_explicit_score() -> None:
    matches = retrieve_strategy_memories(
        SessionState(session_id="s"),
        analysis(stage=ConversationStage.ENGAGE),
    )

    assert all(0.0 <= match.score <= 1.0 for match in matches)
    assert all(match.reason for match in matches)


def test_generic_cases_score_max_match_but_low_specificity() -> None:
    generic = next(case for case in DEFAULT_CASES if case.id == "repeated_questioning_01")
    specific = next(case for case in DEFAULT_CASES if case.id == "evoke_ambivalence_01")

    assert score_case(generic, analysis(stage=ConversationStage.EVOKE)) > 0.9
    assert strategy_memory.case_specificity(generic) == 0.0
    assert strategy_memory.case_specificity(specific) > 0.5


def test_no_match_returns_an_empty_list() -> None:
    assert retrieve_strategy_memories(
        SessionState(session_id="s"),
        analysis(stage=ConversationStage.SAFETY),
        cases=[],
    ) == []


# --- persistence compatibility ----------------------------------------------


def test_legacy_session_json_without_new_fields_still_loads(tmp_path) -> None:
    """Sessions written before the planner existed must keep deserialising."""

    database = tmp_path / "legacy.sqlite3"
    SQLiteSessionStore(database).close()

    legacy_payload = {
        "session_id": "legacy-demo",
        "user_turn_count": 2,
        "primary_focus_topic": "sleep",
        "messages": [
            {
                "role": "user",
                "content": "我最近总是玩到很晚",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        ],
        "updated_at": datetime.now(timezone.utc).isoformat(),
        # deliberately no recent_strategies / planner fields
    }
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO sessions (session_id, state_json, updated_at) VALUES (?, ?, ?)",
        ("legacy-demo", json.dumps(legacy_payload), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    connection.close()

    store = SQLiteSessionStore(database)
    restored = store.get_or_create("legacy-demo")

    assert restored.session_id == "legacy-demo"
    assert restored.primary_focus_topic == "sleep"
    assert restored.recent_strategies == []
    store.close()


def test_planner_runs_on_a_restored_legacy_session(tmp_path) -> None:
    database = tmp_path / "legacy-planner.sqlite3"
    SQLiteSessionStore(database).close()

    legacy_payload = {
        "session_id": "legacy-planner",
        "user_turn_count": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    connection = sqlite3.connect(database)
    connection.execute(
        "INSERT INTO sessions (session_id, state_json, updated_at) VALUES (?, ?, ?)",
        ("legacy-planner", json.dumps(legacy_payload), datetime.now(timezone.utc).isoformat()),
    )
    connection.commit()
    connection.close()

    store = SQLiteSessionStore(database)
    restored = store.get_or_create("legacy-planner")
    decision = StrategyPlanner().plan(
        restored,
        analysis(stage=ConversationStage.EVOKE, change_talk=["想改"]),
    )

    assert decision.selected_strategy is not None
    assert decision.mode in {"fast", "slow"}
    store.close()


def test_memory_file_path_is_repo_relative() -> None:
    path = strategy_memory.strategy_memory_file()

    assert isinstance(path, Path)
    assert path.name == "strategy_memory.json"
