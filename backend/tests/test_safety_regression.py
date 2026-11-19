"""Regression tests for context-aware safety scanning.

These cases were found by probing the rule layer during the competition
hardening pass. They protect the two failure directions that matter most:
missing a real disclosure, and turning ordinary gaming language into a crisis.
"""

import pytest

from noname.safety import assess_rule_risk
from noname.schemas import RiskLevel


@pytest.mark.parametrize(
    "text",
    [
        "我在游戏里被打爆了",
        "这局一直被对面打，太难受了",
        "我在游戏里被队友打",
        "游戏里我死了好多次",
    ],
)
def test_game_context_is_not_treated_as_real_violence(text: str) -> None:
    result = assess_rule_risk(text)

    assert result.level is RiskLevel.LOW
    assert "violence_or_abuse" not in result.signals


@pytest.mark.parametrize(
    "text",
    [
        "我不想死",
        "我不想伤害自己",
        "我不会自杀",
        "我不打算自杀",
        "我从没想过自杀",
        "我没有想死，只是最近真的很累",
    ],
)
def test_explicit_denial_is_not_escalated(text: str) -> None:
    result = assess_rule_risk(text)

    assert result.level is RiskLevel.LOW
    assert "self_harm_intent" not in result.signals


def test_affirmative_wish_to_die_is_still_high_risk() -> None:
    result = assess_rule_risk("我不想活了")

    assert result.level is RiskLevel.HIGH
    assert "self_harm_intent" in result.signals


@pytest.mark.parametrize(
    "text",
    [
        "我被人打了",
        "我被我爸打了",
        "我打游戏被我爸打了",
        "爸妈打我了",
        "队友打我",
        "对面打我",
    ],
)
def test_physical_abuse_variants_are_detected(text: str) -> None:
    result = assess_rule_risk(text)

    assert result.level is RiskLevel.HIGH
    assert "violence_or_abuse" in result.signals


@pytest.mark.parametrize(
    "text",
    [
        "我想一了百了",
        "我想消失，撑不下去了",
        "我觉得结束这一切算了",
        "我不想醒来",
    ],
)
def test_euphemistic_self_harm_language_is_detected(text: str) -> None:
    result = assess_rule_risk(text)

    assert result.level is RiskLevel.HIGH
    assert "self_harm_intent" in result.signals


def test_hopelessness_without_immediate_intent_stays_concern() -> None:
    result = assess_rule_risk("我觉得活着好累")

    assert result.level is RiskLevel.CONCERN
    assert "severe_hopelessness" in result.signals


def test_third_party_hostility_is_concern_not_self_harm() -> None:
    result = assess_rule_risk("队友说让我去死")

    assert result.level is RiskLevel.CONCERN
    assert "hostile_language" in result.signals
    assert "self_harm_intent" not in result.signals


def test_third_party_hostility_does_not_mask_self_directed_intent() -> None:
    result = assess_rule_risk("队友说让我去死，我也想死")

    assert result.level is RiskLevel.HIGH
    assert "self_harm_intent" in result.signals


def test_third_party_hostility_does_not_mask_intent_without_punctuation() -> None:
    result = assess_rule_risk("队友说让我去死我也想死")

    assert result.level is RiskLevel.HIGH
    assert "self_harm_intent" in result.signals


def test_negation_does_not_mask_a_later_disclosure() -> None:
    result = assess_rule_risk("我没有想死，但是我想消失")

    assert result.level is RiskLevel.CONCERN
    assert "severe_hopelessness" in result.signals
