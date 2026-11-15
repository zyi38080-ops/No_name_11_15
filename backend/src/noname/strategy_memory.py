"""Transparent Strategy Memory for the dual-process strategy planner.

This is deliberately **not** a vector store, and matching is **not** embedding
similarity. Each case is a hand-written MI experience record, and matching is a
documented additive rule score over interpretable signals (see ``score_case``).

The memory only supplies a *strategy prior*. It never decides the final action
on its own; the planner combines it with stage, goal, motivation, need and
repetition signals before selecting anything.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator

from .paths import strategy_memory_file
from .schemas import ConversationAnalysis, ConversationStage, MIStrategy, SessionState

logger = logging.getLogger(__name__)

ANY = "*"

# Additive match components. They sum to 1.0 so a perfect match scores 1.0.
MATCH_STAGE_WEIGHT = 0.35
MATCH_FOCUS_WEIGHT = 0.25
MATCH_NEED_WEIGHT = 0.20
MATCH_MOTIVATION_WEIGHT = 0.10
MATCH_CHANGE_TALK_WEIGHT = 0.05
MATCH_SUSTAIN_TALK_WEIGHT = 0.05


class StrategyMemoryCase(BaseModel):
    """One hand-written MI experience record.

    ``None`` on ``change_talk`` / ``sustain_talk`` means "don't care"; ``True``
    requires the signal to be present and ``False`` requires it to be absent.
    """

    id: str = Field(min_length=1)
    stage: ConversationStage | None = None
    focus_topics: list[str] = Field(default_factory=lambda: [ANY])
    needs: list[str] = Field(default_factory=lambda: [ANY])
    motivation_range: list[int] = Field(default_factory=lambda: [0, 10])
    change_talk: bool | None = None
    sustain_talk: bool | None = None
    preferred_strategies: list[MIStrategy] = Field(default_factory=list)
    avoid_strategies: list[MIStrategy] = Field(default_factory=list)
    reason: str = ""
    weight: float = Field(default=1.0, gt=0, le=2.0)

    @field_validator("motivation_range")
    @classmethod
    def validate_motivation_range(cls, value: list[int]) -> list[int]:
        if len(value) != 2:
            raise ValueError("motivation_range must contain exactly [low, high]")
        low, high = sorted(int(item) for item in value)
        return [max(0, low), min(10, high)]


class StrategyMemoryMatch(BaseModel):
    case_id: str
    score: float = Field(ge=0, le=1)
    specificity: float = Field(default=0.0, ge=0, le=1)
    preferred_strategies: list[MIStrategy] = Field(default_factory=list)
    avoid_strategies: list[MIStrategy] = Field(default_factory=list)
    reason: str = ""


def _needs_of(analysis: ConversationAnalysis) -> set[str]:
    return {item.name for item in analysis.psychological_needs}


def _motivation_of(analysis: ConversationAnalysis) -> int | None:
    return analysis.motivation.importance


def _signal_matches(required: bool | None, present: bool) -> float:
    """None = wildcard, otherwise require the signal to match the case."""

    if required is None:
        return 1.0
    return 1.0 if required == present else 0.0


def score_case(case: StrategyMemoryCase, analysis: ConversationAnalysis) -> float:
    """Transparent additive match score in [0, 1]. No hidden weighting."""

    score = 0.0

    if case.stage is None or case.stage is analysis.stage:
        score += MATCH_STAGE_WEIGHT

    if ANY in case.focus_topics or (analysis.focus_topic and analysis.focus_topic in case.focus_topics):
        score += MATCH_FOCUS_WEIGHT

    if ANY in case.needs or (_needs_of(analysis) & set(case.needs)):
        score += MATCH_NEED_WEIGHT

    importance = _motivation_of(analysis)
    low, high = case.motivation_range
    if importance is None:
        # Unknown motivation is a partial match rather than a mismatch, because
        # the caller genuinely does not know yet.
        score += MATCH_MOTIVATION_WEIGHT * 0.5
    elif low <= importance <= high:
        score += MATCH_MOTIVATION_WEIGHT

    score += MATCH_CHANGE_TALK_WEIGHT * _signal_matches(case.change_talk, bool(analysis.change_talk))
    score += MATCH_SUSTAIN_TALK_WEIGHT * _signal_matches(case.sustain_talk, bool(analysis.sustain_talk))

    return min(1.0, score * case.weight)


def case_specificity(case: StrategyMemoryCase) -> float:
    """Fraction of the six signals on which the case states a real constraint.

    A case that only says "any stage / any topic" matched nothing in particular,
    so it should not make the current situation look *familiar*. Keeping this
    separate from ``score_case`` is what stops generic cross-cutting rules from
    inflating the router's familiarity signal.
    """

    stated = [
        case.stage is not None,
        ANY not in case.focus_topics,
        ANY not in case.needs,
        case.motivation_range != [0, 10],
        case.change_talk is not None,
        case.sustain_talk is not None,
    ]
    return round(sum(stated) / len(stated), 4)


def retrieve_strategy_memories(
    state: SessionState,
    analysis: ConversationAnalysis,
    *,
    top_k: int = 3,
    min_score: float = 0.5,
    cases: list[StrategyMemoryCase] | None = None,
) -> list[StrategyMemoryMatch]:
    """Return the best-matching experience cases, strongest first.

    ``state`` is accepted for interface stability and future context-aware
    matching; the first version scores on the current turn analysis only.
    """

    del state  # reserved for future context-aware matching
    pool = cases if cases is not None else load_strategy_memory()

    scored = [(case, score_case(case, analysis)) for case in pool]
    scored = [(case, score) for case, score in scored if score >= min_score]
    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        StrategyMemoryMatch(
            case_id=case.id,
            score=round(score, 4),
            specificity=case_specificity(case),
            preferred_strategies=list(case.preferred_strategies),
            avoid_strategies=list(case.avoid_strategies),
            reason=case.reason,
        )
        for case, score in scored[:top_k]
    ]


# --------------------------------------------------------------------------
# Seed cases. Kept in code so the planner still works when the JSON file is
# absent (fresh checkout, packaging, or a hand-edit mistake).
# --------------------------------------------------------------------------

DEFAULT_CASES: tuple[StrategyMemoryCase, ...] = (
    # --- ENGAGE -----------------------------------------------------------
    StrategyMemoryCase(
        id="engage_interest_01",
        stage=ConversationStage.ENGAGE,
        focus_topics=[ANY],
        needs=[ANY],
        motivation_range=[0, 10],
        change_talk=False,
        sustain_talk=False,
        preferred_strategies=[MIStrategy.SIMPLE_REFLECTION, MIStrategy.OPEN_QUESTION],
        reason="用户只表达了对游戏的兴趣，先反映具体体验，不急于推进改变。",
    ),
    StrategyMemoryCase(
        id="engage_denial_01",
        stage=ConversationStage.ENGAGE,
        needs=["autonomy"],
        motivation_range=[0, 3],
        change_talk=False,
        sustain_talk=True,
        preferred_strategies=[MIStrategy.AUTONOMY_SUPPORT, MIStrategy.SIMPLE_REFLECTION],
        avoid_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.ELICIT_CHANGE_TALK],
        reason="用户明确表示不需要改变，推进计划或诱导改变理由只会增加对抗。",
    ),
    StrategyMemoryCase(
        id="engage_topic_unclear_01",
        stage=ConversationStage.ENGAGE,
        focus_topics=[ANY],
        motivation_range=[0, 10],
        preferred_strategies=[MIStrategy.OPEN_QUESTION, MIStrategy.SIMPLE_REFLECTION],
        reason="尚未形成明确关注点，先收集具体信息。",
    ),
    # --- FOCUS ------------------------------------------------------------
    StrategyMemoryCase(
        id="focus_topic_clear_01",
        stage=ConversationStage.FOCUS,
        focus_topics=["sleep", "stopping", "school", "family", "emotion", "social"],
        change_talk=False,
        preferred_strategies=[MIStrategy.SUMMARY, MIStrategy.OPEN_QUESTION],
        reason="关注点已经清楚，先总结再与用户共同确认优先方向。",
    ),
    StrategyMemoryCase(
        id="focus_ambiguous_01",
        stage=ConversationStage.FOCUS,
        motivation_range=[0, 5],
        preferred_strategies=[MIStrategy.SUMMARY, MIStrategy.COMPLEX_REFLECTION],
        avoid_strategies=[MIStrategy.ACTION_PLANNING],
        reason="关注点仍在摇摆，先做复杂反映而不是收缩到行动。",
    ),
    # --- EVOKE ------------------------------------------------------------
    StrategyMemoryCase(
        id="evoke_ambivalence_01",
        stage=ConversationStage.EVOKE,
        focus_topics=["sleep", "stopping", ANY],
        needs=["belonging", ANY],
        motivation_range=[3, 7],
        change_talk=True,
        sustain_talk=True,
        preferred_strategies=[MIStrategy.DOUBLE_SIDED_REFLECTION, MIStrategy.ELICIT_CHANGE_TALK],
        avoid_strategies=[MIStrategy.ACTION_PLANNING],
        reason="用户同时存在改变理由和维持现状理由，不宜过早推进计划。",
    ),
    StrategyMemoryCase(
        id="evoke_change_talk_only_01",
        stage=ConversationStage.EVOKE,
        change_talk=True,
        sustain_talk=False,
        preferred_strategies=[
            MIStrategy.ELICIT_CHANGE_TALK,
            MIStrategy.AFFIRMATION,
            MIStrategy.COMPLEX_REFLECTION,
        ],
        reason="用户已经在说自己的改变理由，应放大并肯定，而不是替他总结。",
    ),
    StrategyMemoryCase(
        id="evoke_sustain_talk_only_01",
        stage=ConversationStage.EVOKE,
        motivation_range=[0, 4],
        change_talk=False,
        sustain_talk=True,
        preferred_strategies=[MIStrategy.AUTONOMY_SUPPORT, MIStrategy.COMPLEX_REFLECTION],
        avoid_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.ASK_PERMISSION],
        reason="用户当前只有维持现状的理由，需要先支持自主性而不是推动改变。",
    ),
    StrategyMemoryCase(
        id="evoke_low_motivation_01",
        stage=ConversationStage.EVOKE,
        motivation_range=[0, 3],
        preferred_strategies=[MIStrategy.AUTONOMY_SUPPORT, MIStrategy.READINESS_RULER],
        avoid_strategies=[MIStrategy.ACTION_PLANNING],
        reason="改变意愿很低时，强行规划会破坏关系和自主感。",
    ),
    StrategyMemoryCase(
        id="evoke_readiness_check_01",
        stage=ConversationStage.EVOKE,
        motivation_range=[4, 7],
        preferred_strategies=[MIStrategy.READINESS_RULER, MIStrategy.ELICIT_CHANGE_TALK],
        reason="意愿处于中间区间，适合先量化意愿再决定是否收缩到计划。",
    ),
    StrategyMemoryCase(
        id="evoke_belonging_need_01",
        stage=ConversationStage.EVOKE,
        needs=["belonging", "connection"],
        preferred_strategies=[
            MIStrategy.DOUBLE_SIDED_REFLECTION,
            MIStrategy.COMPLEX_REFLECTION,
            MIStrategy.AFFIRMATION,
        ],
        reason="队友和朋友是需要而不是问题主题，双面反映可以同时承认真实价值与影响。",
    ),
    StrategyMemoryCase(
        id="evoke_autonomy_need_01",
        stage=ConversationStage.EVOKE,
        needs=["autonomy"],
        preferred_strategies=[MIStrategy.AUTONOMY_SUPPORT, MIStrategy.COMPLEX_REFLECTION],
        avoid_strategies=[MIStrategy.ACTION_PLANNING],
        reason="自主需要高时，先明确决定权在用户手里。",
    ),
    # --- PLAN -------------------------------------------------------------
    StrategyMemoryCase(
        id="plan_ready_01",
        stage=ConversationStage.PLAN,
        change_talk=True,
        motivation_range=[6, 10],
        preferred_strategies=[MIStrategy.ASK_PERMISSION, MIStrategy.ACTION_PLANNING],
        reason="用户已有明确行动意图，先征得同意再形成微行动。",
    ),
    StrategyMemoryCase(
        id="plan_confidence_low_01",
        stage=ConversationStage.PLAN,
        motivation_range=[3, 10],
        preferred_strategies=[MIStrategy.CONFIDENCE_RULER, MIStrategy.ASK_PERMISSION],
        reason="计划阶段需要区分「有多重要」和「有多把握」，把握不足时先降低目标难度。",
    ),
    StrategyMemoryCase(
        id="plan_sleep_01",
        stage=ConversationStage.PLAN,
        focus_topics=["sleep"],
        preferred_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.ASK_PERMISSION],
        reason="睡眠主题的行动实验通常足够小，可以直接形成三天微实验。",
    ),
    StrategyMemoryCase(
        id="plan_stopping_01",
        stage=ConversationStage.PLAN,
        focus_topics=["stopping"],
        preferred_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.ASK_PERMISSION],
        reason="停止困难适合用「最后一局」这类可观察的结束点做实验。",
    ),
    StrategyMemoryCase(
        id="plan_school_01",
        stage=ConversationStage.PLAN,
        focus_topics=["school"],
        preferred_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.ASK_PERMISSION],
        reason="学习启动类目标要足够小，先做五分钟再决定是否继续。",
    ),
    StrategyMemoryCase(
        id="plan_family_01",
        stage=ConversationStage.PLAN,
        focus_topics=["family"],
        preferred_strategies=[MIStrategy.ASK_PERMISSION, MIStrategy.ACTION_PLANNING],
        reason="家庭冲突主题的行动由用户自己提出，避免替用户安排沟通。",
    ),
    StrategyMemoryCase(
        id="plan_emotion_01",
        stage=ConversationStage.PLAN,
        focus_topics=["emotion"],
        preferred_strategies=[MIStrategy.ACTION_PLANNING, MIStrategy.AFFIRMATION],
        reason="情绪主题优先用观察记录类实验，而不是直接要求减少游戏。",
    ),
    StrategyMemoryCase(
        id="plan_social_01",
        stage=ConversationStage.PLAN,
        focus_topics=["social"],
        needs=["belonging"],
        preferred_strategies=[
            MIStrategy.ACTION_PLANNING,
            MIStrategy.ASK_PERMISSION,
            MIStrategy.AUTONOMY_SUPPORT,
        ],
        reason="社交主题的计划要同时保留下线边界和队友关系。",
    ),
    # --- REVIEW -----------------------------------------------------------
    StrategyMemoryCase(
        id="review_success_01",
        stage=ConversationStage.REVIEW,
        change_talk=True,
        preferred_strategies=[MIStrategy.AFFIRMATION, MIStrategy.REVIEW_AND_ADJUST],
        reason="用户观察到积极变化时，先肯定再讨论是否调整难度。",
    ),
    StrategyMemoryCase(
        id="review_failed_01",
        stage=ConversationStage.REVIEW,
        motivation_range=[0, 5],
        sustain_talk=True,
        preferred_strategies=[
            MIStrategy.AFFIRMATION,
            MIStrategy.COMPLEX_REFLECTION,
            MIStrategy.REVIEW_AND_ADJUST,
        ],
        avoid_strategies=[MIStrategy.ACTION_PLANNING],
        reason="计划未完成时不追加新任务，先复盘条件和难度。",
    ),
    StrategyMemoryCase(
        id="review_adjust_01",
        stage=ConversationStage.REVIEW,
        preferred_strategies=[MIStrategy.REVIEW_AND_ADJUST, MIStrategy.COMPLEX_REFLECTION],
        reason="复盘的重点是有效条件和目标难度，而不是成功或失败本身。",
    ),
    # --- Cross-cutting ----------------------------------------------------
    StrategyMemoryCase(
        id="repeated_questioning_01",
        stage=None,
        focus_topics=[ANY],
        needs=[ANY],
        motivation_range=[0, 10],
        preferred_strategies=[MIStrategy.SIMPLE_REFLECTION, MIStrategy.SUMMARY],
        avoid_strategies=[MIStrategy.OPEN_QUESTION],
        reason="连续多轮以问句收尾会加重负担，本轮应改用陈述式反映或总结。",
    ),
)


def _load_from_file() -> list[StrategyMemoryCase] | None:
    path = strategy_memory_file()
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Strategy memory file unreadable (%s); using built-in seed", exc)
        return None

    cases_raw = raw.get("cases") if isinstance(raw, dict) else raw
    if not isinstance(cases_raw, list):
        logger.warning("Strategy memory file has no case list; using built-in seed")
        return None

    cases: list[StrategyMemoryCase] = []
    for item in cases_raw:
        try:
            cases.append(StrategyMemoryCase.model_validate(item))
        except Exception as exc:  # a single bad record must not disable the memory
            logger.warning("Skipping invalid strategy memory case: %s", exc)
    return cases or None


@lru_cache(maxsize=1)
def load_strategy_memory() -> list[StrategyMemoryCase]:
    """Load the strategy memory, falling back to the built-in seed cases."""

    return _load_from_file() or list(DEFAULT_CASES)
