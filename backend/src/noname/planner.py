"""DPDP-inspired dual-process strategy planning.

This module upgrades *how the next MI strategy is chosen* for ordinary
(non-safety) turns. It is a **DPDP-inspired dual-process strategy planner**:

* System-1 flavour — a fast planner that scores a small candidate set with a
  transparent utility function augmented by hand-written Strategy Memory.
* System-2 flavour — a slow planner that runs a bounded **depth-2 look-ahead**
  over abstract planning transitions when the fast path is not confident.

It is explicitly **not** a reproduction of the DPDP paper, **not** reinforcement
learning, and **not** MCTS. There is no UCT, no selection/expansion/simulation/
backup loop, no learned value function and no training. Every weight is a
hand-set, documented business default.

Safety is never a term in these utility functions. High-risk and active-safety
turns bypass this module entirely (``mode="bypassed"``) and keep running through
the existing safety state machine.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from .schemas import (
    ConversationAnalysis,
    ConversationStage,
    LookaheadPath,
    MIStrategy,
    PlannerDecision,
    SessionState,
    StrategyCandidate,
)
from .strategy_memory import StrategyMemoryMatch, retrieve_strategy_memories

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Strategy groupings used by several rules below.
# --------------------------------------------------------------------------

# Strategies that move the conversation toward committing to change.
ADVANCING_STRATEGIES: frozenset[MIStrategy] = frozenset(
    {
        MIStrategy.ACTION_PLANNING,
        MIStrategy.ASK_PERMISSION,
        MIStrategy.CONFIDENCE_RULER,
        MIStrategy.ELICIT_CHANGE_TALK,
    }
)

# Strategies that hold space, reflect, or protect autonomy.
SUPPORTIVE_STRATEGIES: frozenset[MIStrategy] = frozenset(
    {
        MIStrategy.AUTONOMY_SUPPORT,
        MIStrategy.SIMPLE_REFLECTION,
        MIStrategy.COMPLEX_REFLECTION,
        MIStrategy.AFFIRMATION,
        MIStrategy.SUMMARY,
        MIStrategy.DOUBLE_SIDED_REFLECTION,
        MIStrategy.READINESS_RULER,
        MIStrategy.OPEN_QUESTION,
    }
)

# Initial candidate mapping per stage. This is a *prior*, not a hard rule: the
# heuristic result from ``mi.py`` and the Strategy Memory both widen it.
STAGE_CANDIDATES: dict[ConversationStage, tuple[MIStrategy, ...]] = {
    ConversationStage.ENGAGE: (
        MIStrategy.SIMPLE_REFLECTION,
        MIStrategy.OPEN_QUESTION,
        MIStrategy.AFFIRMATION,
    ),
    ConversationStage.FOCUS: (
        MIStrategy.SUMMARY,
        MIStrategy.OPEN_QUESTION,
        MIStrategy.COMPLEX_REFLECTION,
    ),
    ConversationStage.EVOKE: (
        MIStrategy.COMPLEX_REFLECTION,
        MIStrategy.DOUBLE_SIDED_REFLECTION,
        MIStrategy.AUTONOMY_SUPPORT,
        MIStrategy.ELICIT_CHANGE_TALK,
        MIStrategy.READINESS_RULER,
    ),
    ConversationStage.PLAN: (
        MIStrategy.ASK_PERMISSION,
        MIStrategy.ACTION_PLANNING,
        MIStrategy.CONFIDENCE_RULER,
        MIStrategy.AUTONOMY_SUPPORT,
    ),
    ConversationStage.REVIEW: (
        MIStrategy.AFFIRMATION,
        MIStrategy.REVIEW_AND_ADJUST,
        MIStrategy.COMPLEX_REFLECTION,
        MIStrategy.ACTION_PLANNING,
    ),
    ConversationStage.SAFETY: (),
}

NEIGHBOUR_STAGES: dict[ConversationStage, tuple[ConversationStage, ...]] = {
    ConversationStage.ENGAGE: (ConversationStage.FOCUS,),
    ConversationStage.FOCUS: (ConversationStage.ENGAGE, ConversationStage.EVOKE),
    ConversationStage.EVOKE: (ConversationStage.FOCUS, ConversationStage.PLAN),
    ConversationStage.PLAN: (ConversationStage.EVOKE, ConversationStage.REVIEW),
    ConversationStage.REVIEW: (ConversationStage.PLAN,),
    ConversationStage.SAFETY: (),
}


# --------------------------------------------------------------------------
# Central configuration. Every weight and threshold lives here on purpose.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlannerWeights:
    """Utility weights. Positive terms sum to 1.0 so U(s,a) lands near [0, 1].

    Business meaning of each term:

    * ``stage``      — does the strategy belong to the current MI stage at all.
    * ``goal``       — does it serve what this turn is trying to achieve.
    * ``memory``     — how strongly Strategy Memory supports it.
    * ``motivation`` — does it fit the user's stated change importance.
    * ``need``       — does it answer the psychological need the user voiced.
    * ``anchor``     — does it help stay on / return to the original focus.
    * ``repeat``     — penalty for re-using the same strategy several turns.
    * ``resist``     — penalty for pushing when the user is holding back.
    """

    stage: float = 0.28
    goal: float = 0.22
    memory: float = 0.18
    motivation: float = 0.14
    need: float = 0.10
    anchor: float = 0.08
    repeat: float = 0.25
    resist: float = 0.30


@dataclass(frozen=True)
class RouterWeights:
    """Weights of the familiarity/confidence router.

    ``confidence = familiarity + state + margin - conflict`` using these
    weights. ``conflict`` is deliberately the heaviest term: a contradiction
    between signals is stronger evidence that the fast path is unsafe than any
    amount of familiarity is evidence that it is safe.
    """

    familiarity: float = 0.35
    state_confidence: float = 0.25
    margin: float = 0.40
    conflict: float = 1.0


@dataclass(frozen=True)
class PlannerConfig:
    weights: PlannerWeights = field(default_factory=PlannerWeights)
    router_weights: RouterWeights = field(default_factory=RouterWeights)
    #: A utility gap of this size between rank 1 and rank 2 counts as fully decided.
    #: Calibrated to the observed spread of ``score_strategy``, whose weights are
    #: spread over six near-0.5..1.0 terms, so real gaps land around 0.02-0.12.
    margin_reference: float = 0.12
    #: Router confidence at or above this uses the fast planner, below uses look-ahead.
    #: Calibrated so a familiar, uncontradicted turn clears it while any
    #: ambivalence or missing-memory turn does not.
    fast_threshold: float = 0.46
    #: Discount applied to future utility in the depth-2 look-ahead.
    gamma: float = 0.7
    memory_top_k: int = 3
    memory_min_score: float = 0.5
    max_candidates: int = 5
    min_candidates: int = 3
    #: How many recent turns count for the repetition penalty.
    recent_window: int = 3
    #: Look-ahead branching factor for the second ply.
    lookahead_branch: int = 3


DEFAULT_CONFIG = PlannerConfig()


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------


def _heuristic_prior(heuristic: ConversationAnalysis | None) -> list[MIStrategy]:
    """``mi.py`` still runs; its result is reused as a candidate source and prior."""

    return list(heuristic.mi_strategies) if heuristic is not None else []


def generate_candidate_strategies(
    analysis: ConversationAnalysis,
    session_state: SessionState,
    memories: list[StrategyMemoryMatch],
    *,
    heuristic: ConversationAnalysis | None = None,
    config: PlannerConfig = DEFAULT_CONFIG,
) -> list[MIStrategy]:
    """Build the 3–5 strategies actually considered this turn.

    Sources, in priority order: the stage mapping, the existing ``mi.py``
    heuristic result, and Strategy Memory. Strategies Memory marks as *avoid*
    are dropped unless the stage mapping insists on them.
    """

    del session_state  # reserved for future context-aware candidate selection

    ordered: list[MIStrategy] = list(STAGE_CANDIDATES.get(analysis.stage, ()))
    for strategy in _heuristic_prior(heuristic):
        if strategy not in ordered:
            ordered.append(strategy)
    for match in memories:
        for strategy in match.preferred_strategies:
            if strategy not in ordered:
                ordered.append(strategy)

    avoided = {strategy for match in memories for strategy in match.avoid_strategies}
    stage_set = set(STAGE_CANDIDATES.get(analysis.stage, ()))
    ordered = [s for s in ordered if s not in avoided or s in stage_set]

    # Always give the utility function something to compare.
    for strategy in STAGE_CANDIDATES.get(analysis.stage, ()):
        if len(ordered) >= config.min_candidates:
            break
        if strategy not in ordered:
            ordered.append(strategy)

    return ordered[: config.max_candidates]


# --------------------------------------------------------------------------
# Utility scoring
# --------------------------------------------------------------------------


def _stage_match(strategy: MIStrategy, stage: ConversationStage) -> float:
    if strategy in STAGE_CANDIDATES.get(stage, ()):
        return 1.0
    for neighbour in NEIGHBOUR_STAGES.get(stage, ()):
        if strategy in STAGE_CANDIDATES.get(neighbour, ()):
            return 0.4
    return 0.0


def _goal_match(strategy: MIStrategy, analysis: ConversationAnalysis) -> float:
    """How well the strategy serves what this turn is trying to achieve.

    Expressed as documented rules over the current state, not a learned model.
    The strongest applicable rule wins.
    """

    importance = analysis.motivation.importance
    change = bool(analysis.change_talk)
    sustain = bool(analysis.sustain_talk)
    scores: list[float] = []

    if change and sustain:
        scores.append({MIStrategy.DOUBLE_SIDED_REFLECTION: 1.0,
                       MIStrategy.ELICIT_CHANGE_TALK: 0.8}.get(strategy, 0.35))
    if change and not sustain:
        scores.append({MIStrategy.ELICIT_CHANGE_TALK: 1.0,
                       MIStrategy.AFFIRMATION: 0.8}.get(strategy, 0.5))
    if sustain and not change:
        scores.append({MIStrategy.AUTONOMY_SUPPORT: 1.0,
                       MIStrategy.COMPLEX_REFLECTION: 0.7}.get(strategy, 0.35))
    if importance is not None and importance <= 3:
        scores.append({MIStrategy.AUTONOMY_SUPPORT: 1.0,
                       MIStrategy.READINESS_RULER: 0.9}.get(strategy, 0.3))
        if strategy is MIStrategy.ACTION_PLANNING:
            scores.append(0.0)
    if importance is not None and importance >= 8:
        scores.append({MIStrategy.ASK_PERMISSION: 1.0,
                       MIStrategy.ACTION_PLANNING: 0.9,
                       MIStrategy.AFFIRMATION: 0.8}.get(strategy, 0.5))
    if analysis.stage is ConversationStage.REVIEW:
        scores.append({MIStrategy.REVIEW_AND_ADJUST: 1.0,
                       MIStrategy.AFFIRMATION: 0.85}.get(strategy, 0.4))
    if analysis.stage is ConversationStage.PLAN and strategy in ADVANCING_STRATEGIES:
        scores.append(0.9)

    if not scores:
        return 0.5 if strategy in STAGE_CANDIDATES.get(analysis.stage, ()) else 0.15
    return max(scores)


def _memory_prior(
    strategy: MIStrategy,
    memories: list[StrategyMemoryMatch],
) -> float:
    """Support for the strategy across matched cases, in [-1, 1]."""

    support = 0.0
    for match in memories:
        if strategy in match.preferred_strategies:
            support += match.score
        if strategy in match.avoid_strategies:
            support -= match.score
    if not memories:
        return 0.0
    return max(-1.0, min(1.0, support / len(memories)))


def _motivation_match(strategy: MIStrategy, analysis: ConversationAnalysis) -> float:
    importance = analysis.motivation.importance
    if importance is None:
        return 0.5
    if importance <= 3:
        return 0.15 if strategy in ADVANCING_STRATEGIES else 0.9
    if importance >= 8:
        return 0.95 if strategy in ADVANCING_STRATEGIES else 0.5
    return 0.6


def _need_match(strategy: MIStrategy, analysis: ConversationAnalysis) -> float:
    needs = {item.name for item in analysis.psychological_needs}
    if not needs:
        return 0.5
    score = 0.5
    if "autonomy" in needs:
        if strategy is MIStrategy.AUTONOMY_SUPPORT:
            score = max(score, 1.0)
        elif strategy in ADVANCING_STRATEGIES:
            score = min(score, 0.25)
    if needs & {"belonging", "connection"}:
        if strategy in {MIStrategy.AFFIRMATION, MIStrategy.COMPLEX_REFLECTION,
                        MIStrategy.DOUBLE_SIDED_REFLECTION}:
            score = max(score, 0.85)
    if "achievement" in needs and strategy is MIStrategy.AFFIRMATION:
        score = max(score, 0.85)
    return score


def _anchor_alignment(strategy: MIStrategy, session_state: SessionState,
                      analysis: ConversationAnalysis) -> float:
    """Reward strategies that can naturally bridge back to the original focus."""

    primary = session_state.primary_focus_topic
    if primary is None:
        return 0.5
    drifted = analysis.focus_topic is not None and analysis.focus_topic != primary
    if not drifted:
        return 0.6
    if strategy in {MIStrategy.SUMMARY, MIStrategy.OPEN_QUESTION,
                    MIStrategy.COMPLEX_REFLECTION}:
        return 1.0
    if strategy in ADVANCING_STRATEGIES:
        return 0.2
    return 0.5


def _repetition_penalty(
    strategy: MIStrategy,
    session_state: SessionState,
    config: PlannerConfig,
) -> float:
    """Penalise re-using a strategy that dominated the last few turns."""

    recent = session_state.recent_strategies[-config.recent_window :]
    if not recent:
        return 0.0
    repeats = sum(1 for item in recent if item is strategy)
    return min(1.0, repeats / len(recent))


def _resistance_penalty(strategy: MIStrategy, analysis: ConversationAnalysis) -> float:
    """Penalise pushing when the user is holding back."""

    if strategy not in ADVANCING_STRATEGIES:
        return 0.0
    importance = analysis.motivation.importance
    low_motivation = importance is not None and importance <= 3
    if analysis.sustain_talk and low_motivation:
        return 1.0
    if analysis.sustain_talk or low_motivation:
        return 0.6
    return 0.0


def score_strategy(
    session_state: SessionState,
    analysis: ConversationAnalysis,
    strategy: MIStrategy,
    memories: list[StrategyMemoryMatch],
    *,
    config: PlannerConfig = DEFAULT_CONFIG,
    history: list[MIStrategy] | None = None,
) -> StrategyCandidate:
    """Transparent utility U(s, a). Every term is inspectable in the trace."""

    weights = config.weights
    stage_score = _stage_match(strategy, analysis.stage)
    goal_score = _goal_match(strategy, analysis)
    memory_score = (_memory_prior(strategy, memories) + 1.0) / 2.0  # to [0, 1]
    motivation_score = _motivation_match(strategy, analysis)
    need_score = _need_match(strategy, analysis)
    anchor_score = _anchor_alignment(strategy, session_state, analysis)

    repeat = _repetition_penalty(strategy, session_state, config)
    if history:
        recent = history[-config.recent_window :]
        if recent:
            repeat = max(repeat, sum(1 for item in recent if item is strategy) / len(recent))
    resist = _resistance_penalty(strategy, analysis)

    utility = (
        weights.stage * stage_score
        + weights.goal * goal_score
        + weights.memory * memory_score
        + weights.motivation * motivation_score
        + weights.need * need_score
        + weights.anchor * anchor_score
        - weights.repeat * repeat
        - weights.resist * resist
    )

    explanation = (
        f"stage={stage_score:.2f} goal={goal_score:.2f} memory={memory_score:.2f} "
        f"motivation={motivation_score:.2f} need={need_score:.2f} anchor={anchor_score:.2f} "
        f"repeat=-{repeat:.2f} resist=-{resist:.2f}"
    )
    return StrategyCandidate(
        strategy=strategy,
        rule_score=round(stage_score, 4),
        memory_score=round(memory_score, 4),
        goal_score=round(goal_score, 4),
        state_score=round(motivation_score, 4),
        repetition_penalty=round(repeat, 4),
        resistance_penalty=round(resist, 4),
        utility_score=round(utility, 4),
        explanation=explanation,
    )


def rank_candidates(
    session_state: SessionState,
    analysis: ConversationAnalysis,
    candidates: list[MIStrategy],
    memories: list[StrategyMemoryMatch],
    *,
    config: PlannerConfig = DEFAULT_CONFIG,
) -> list[StrategyCandidate]:
    scored = [
        score_strategy(session_state, analysis, strategy, memories, config=config)
        for strategy in candidates
    ]
    scored.sort(key=lambda item: item.utility_score, reverse=True)
    return scored


# --------------------------------------------------------------------------
# Abstract planning transition (used only for relative comparison)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanningState:
    """Abstract planning variables. NOT a prediction of the user's mind.

    These numbers exist so the look-ahead can compare candidate paths against
    each other. They must never be surfaced to the user as facts about them.
    """

    stage: ConversationStage
    change_talk: float = 0.5
    resistance: float = 0.5
    readiness: float = 0.5
    plan_readiness: float = 0.5
    focus_alignment: float = 0.5
    information_gain: float = 0.5
    repetition_burden: float = 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def initial_planning_state(analysis: ConversationAnalysis) -> PlanningState:
    importance = analysis.motivation.importance
    readiness = 0.5 if importance is None else importance / 10
    return PlanningState(
        stage=analysis.stage,
        change_talk=1.0 if analysis.change_talk else (0.3 if analysis.sustain_talk else 0.5),
        resistance=0.85 if analysis.sustain_talk else 0.3,
        readiness=readiness,
        plan_readiness=_clamp(readiness - 0.15) if analysis.stage is not ConversationStage.PLAN else readiness,
        focus_alignment=0.7 if analysis.focus_topic else 0.5,
    )


SIMULATION_DELTA = 0.25


def simulate_next_state(
    planning_state: PlanningState,
    strategy: MIStrategy,
) -> PlanningState:
    """One abstract planning transition.

    Documented tendencies only — this is not a psychological prediction model
    and its output is never shown to the user or asserted as fact.
    """

    ps = planning_state
    change, resist, readiness = ps.change_talk, ps.resistance, ps.readiness
    plan_ready, focus = ps.plan_readiness, ps.focus_alignment
    info, burden = ps.information_gain, ps.repetition_burden
    stage = ps.stage

    if strategy is MIStrategy.DOUBLE_SIDED_REFLECTION:
        resist = _clamp(resist - SIMULATION_DELTA)
        change = _clamp(change + SIMULATION_DELTA)
        # Deliberately does not move the conversation into PLAN.
    elif strategy is MIStrategy.ELICIT_CHANGE_TALK:
        # High resistance dampens the effect of inviting change talk.
        gain = SIMULATION_DELTA * (1.0 - resist)
        change = _clamp(change + gain)
        readiness = _clamp(readiness + gain * 0.5)
    elif strategy is MIStrategy.ACTION_PLANNING:
        if readiness >= 0.6:
            plan_ready = _clamp(plan_ready + SIMULATION_DELTA * 1.2)
        else:
            resist = _clamp(resist + SIMULATION_DELTA)
    elif strategy is MIStrategy.AUTONOMY_SUPPORT:
        resist = _clamp(resist - SIMULATION_DELTA)
        # Readiness is deliberately not forced upward.
    elif strategy is MIStrategy.OPEN_QUESTION:
        info = _clamp(info + SIMULATION_DELTA)
        burden = _clamp(burden + SIMULATION_DELTA * 0.8)
    elif strategy is MIStrategy.SUMMARY:
        focus = _clamp(focus + SIMULATION_DELTA)
        change = _clamp(change + SIMULATION_DELTA * 0.4)
    elif strategy in {MIStrategy.SIMPLE_REFLECTION, MIStrategy.COMPLEX_REFLECTION}:
        resist = _clamp(resist - SIMULATION_DELTA * 0.5)
        change = _clamp(change + SIMULATION_DELTA * 0.4)
    elif strategy is MIStrategy.AFFIRMATION:
        resist = _clamp(resist - SIMULATION_DELTA * 0.4)
        readiness = _clamp(readiness + SIMULATION_DELTA * 0.3)
    elif strategy is MIStrategy.READINESS_RULER:
        info = _clamp(info + SIMULATION_DELTA * 0.6)
    elif strategy is MIStrategy.CONFIDENCE_RULER:
        info = _clamp(info + SIMULATION_DELTA * 0.6)
        plan_ready = _clamp(plan_ready + SIMULATION_DELTA * 0.4)
    elif strategy is MIStrategy.ASK_PERMISSION:
        plan_ready = _clamp(plan_ready + SIMULATION_DELTA * 0.6)
        resist = _clamp(resist - SIMULATION_DELTA * 0.3)
    elif strategy is MIStrategy.REVIEW_AND_ADJUST:
        plan_ready = _clamp(plan_ready + SIMULATION_DELTA * 0.7)
        info = _clamp(info + SIMULATION_DELTA * 0.4)

    if stage is ConversationStage.PLAN and plan_ready >= 0.75:
        stage = ConversationStage.REVIEW
    return replace(
        ps,
        stage=stage,
        change_talk=change,
        resistance=resist,
        readiness=readiness,
        plan_readiness=plan_ready,
        focus_alignment=focus,
        information_gain=info,
        repetition_burden=burden,
    )


def simulated_utility(
    planning_state: PlanningState,
    strategy: MIStrategy,
    *,
    config: PlannerConfig = DEFAULT_CONFIG,
) -> float:
    """Utility of a strategy against an *abstract* planning state.

    Same weight structure as ``score_strategy``, but evaluated over simulated
    variables. Used only to rank look-ahead paths relative to each other.
    """

    weights = config.weights
    ps = planning_state

    stage_fit = 1.0 if strategy in STAGE_CANDIDATES.get(ps.stage, ()) else 0.2
    fits_readiness = readiness_fit(ps, strategy)
    need_fit = 1.0 - ps.resistance if strategy in SUPPORTIVE_STRATEGIES else 0.5 * (1.0 - ps.resistance)
    anchor = ps.focus_alignment
    repeat = ps.repetition_burden if strategy is MIStrategy.OPEN_QUESTION else 0.0
    resist_penalty = ps.resistance if strategy in ADVANCING_STRATEGIES else 0.0

    return (
        weights.stage * stage_fit
        + weights.goal * fits_readiness
        + weights.motivation * fits_readiness
        + weights.need * need_fit
        + weights.anchor * anchor
        - weights.repeat * repeat
        - weights.resist * resist_penalty
    )


def readiness_fit(planning_state: PlanningState, strategy: MIStrategy) -> float:
    """Whether a strategy fits the simulated readiness / plan-readiness level."""

    ps = planning_state
    if strategy in ADVANCING_STRATEGIES:
        return ps.plan_readiness
    if strategy is MIStrategy.AUTONOMY_SUPPORT:
        return 1.0 - ps.plan_readiness * 0.3
    return 0.5 + (1.0 - ps.resistance) * 0.3


# --------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------


class FastPlanner:
    """Experience-augmented utility planner: argmax U(s, a). No look-ahead."""

    def __init__(self, config: PlannerConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def rank(
        self,
        session_state: SessionState,
        analysis: ConversationAnalysis,
        candidates: list[MIStrategy],
        memories: list[StrategyMemoryMatch],
    ) -> list[StrategyCandidate]:
        return rank_candidates(
            session_state, analysis, candidates, memories, config=self.config
        )


class Depth2LookaheadPlanner:
    """Bounded depth-2 look-ahead over abstract planning transitions.

    For every candidate ``a``: simulate the abstract next state, then take the
    best next strategy from that state.
    ``PathScore = ImmediateUtility + gamma * FutureUtility``.

    ``simulator`` and ``utility`` are injectable so the behaviour is testable
    without network access, randomness, or a real user.
    """

    def __init__(
        self,
        config: PlannerConfig = DEFAULT_CONFIG,
        *,
        simulator=None,
        utility=None,
    ) -> None:
        self.config = config
        self._simulate = simulator or simulate_next_state
        self._utility = utility or simulated_utility

    def _next_candidates(
        self,
        planning_state: PlanningState,
        memories: list[StrategyMemoryMatch],
    ) -> list[MIStrategy]:
        ordered: list[MIStrategy] = list(STAGE_CANDIDATES.get(planning_state.stage, ()))
        for match in memories:
            for strategy in match.preferred_strategies:
                if strategy not in ordered:
                    ordered.append(strategy)
        return ordered[: self.config.lookahead_branch]

    def search(
        self,
        analysis: ConversationAnalysis,
        scored: list[StrategyCandidate],
        memories: list[StrategyMemoryMatch],
    ) -> list[LookaheadPath]:
        immediate = {item.strategy: item.utility_score for item in scored}
        root = initial_planning_state(analysis)

        paths: list[LookaheadPath] = []
        for item in scored:
            next_state = self._simulate(root, item.strategy)
            best_next: MIStrategy | None = None
            future = 0.0
            for candidate in self._next_candidates(next_state, memories):
                value = self._utility(next_state, candidate, config=self.config)
                if best_next is None or value > future:
                    future, best_next = value, candidate
            paths.append(
                LookaheadPath(
                    first_strategy=item.strategy,
                    immediate=round(immediate.get(item.strategy, 0.0), 4),
                    best_next=best_next,
                    future=round(future, 4),
                    total=round(immediate.get(item.strategy, 0.0) + self.config.gamma * future, 4),
                )
            )

        paths.sort(key=lambda path: path.total, reverse=True)
        return paths


NEXT_GOAL_BY_STAGE: dict[ConversationStage, str] = {
    ConversationStage.ENGAGE: "理解游戏对用户的价值",
    ConversationStage.FOCUS: "与用户共同选定一个优先问题",
    ConversationStage.EVOKE: "引出用户自己的改变理由",
    ConversationStage.PLAN: "形成足够小且由用户选择的行动实验",
    ConversationStage.REVIEW: "复盘有效条件并调整实验",
    ConversationStage.SAFETY: "确认当前安全并连接现实支持",
}


# --------------------------------------------------------------------------
# Familiarity / confidence router
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RouterAssessment:
    familiarity: float
    state_confidence: float
    margin: float
    conflict: float
    confidence: float
    mode: str


class FamiliarityConfidenceRouter:
    """Decide whether the fast planner is trustworthy enough for this turn.

    Not a simple ``if complex: slow`` switch: it weighs how well Strategy Memory
    covers the situation, how complete the state estimate is, how far apart the
    top two utilities are, and how much the signals contradict each other.
    """

    def __init__(self, config: PlannerConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    @staticmethod
    def _state_confidence(analysis: ConversationAnalysis) -> float:
        """Transparent engineering confidence — not a psychological measurement.

        The stage is always known, so it supplies the base. A fresh ENGAGE turn
        legitimately has no focus or motivation yet and must not be treated as
        poorly understood just because the conversation is young.
        """

        score = 0.50
        if analysis.focus_topic is not None:
            score += 0.20
        if analysis.motivation.importance is not None:
            score += 0.20
        if analysis.change_talk or analysis.sustain_talk:
            score += 0.10
        return round(min(1.0, score), 4)

    def _conflict(
        self,
        session_state: SessionState,
        analysis: ConversationAnalysis,
        memories: list[StrategyMemoryMatch],
        scored: list[StrategyCandidate],
    ) -> float:
        score = 0.0
        if analysis.change_talk and analysis.sustain_talk:
            score += 0.45
        primary = session_state.primary_focus_topic
        if primary is not None and analysis.focus_topic is not None and analysis.focus_topic != primary:
            score += 0.20
        if len(scored) >= 2 and (scored[0].utility_score - scored[1].utility_score) < 0.05:
            score += 0.25
        if not memories:
            score += 0.25
        return round(min(1.0, score), 4)

    def assess(
        self,
        session_state: SessionState,
        analysis: ConversationAnalysis,
        memories: list[StrategyMemoryMatch],
        scored: list[StrategyCandidate],
    ) -> RouterAssessment:
        # Familiarity: how specifically a known situation matched. A generic
        # wildcard case contributes little, which is why specificity is folded in.
        familiarity = max((m.score * m.specificity for m in memories), default=0.0)

        state_confidence = self._state_confidence(analysis)

        if len(scored) >= 2:
            gap = scored[0].utility_score - scored[1].utility_score
            margin = _clamp(gap / self.config.margin_reference)
        else:
            margin = 0.0

        conflict = self._conflict(session_state, analysis, memories, scored)

        rw = self.config.router_weights
        confidence = (
            rw.familiarity * familiarity
            + rw.state_confidence * state_confidence
            + rw.margin * margin
            - rw.conflict * conflict
        )
        confidence = round(max(0.0, min(1.0, confidence)), 4)
        mode = "fast" if confidence >= self.config.fast_threshold else "slow"
        return RouterAssessment(
            familiarity=round(familiarity, 4),
            state_confidence=state_confidence,
            margin=round(margin, 4),
            conflict=conflict,
            confidence=confidence,
            mode=mode,
        )


# --------------------------------------------------------------------------
# Facade
# --------------------------------------------------------------------------


class StrategyPlanner:
    """Single entry point used by ``ConversationService``.

    Guarantees:

    * Safety turns never reach the planners — they arrive with ``blocked_reason``
      set and return ``mode="bypassed"`` with no selection.
    * Any internal failure degrades to ``mode="fallback"`` with no selection, so
      the caller keeps the existing ``mi.py`` heuristic strategies.
    * Fully deterministic: no randomness, no network, no LLM judge.
    """

    def __init__(
        self,
        config: PlannerConfig = DEFAULT_CONFIG,
        *,
        fast: FastPlanner | None = None,
        slow: Depth2LookaheadPlanner | None = None,
        router: FamiliarityConfidenceRouter | None = None,
    ) -> None:
        self.config = config
        self.fast = fast or FastPlanner(config)
        self.slow = slow or Depth2LookaheadPlanner(config)
        self.router = router or FamiliarityConfidenceRouter(config)

    def plan(
        self,
        session_state: SessionState,
        analysis: ConversationAnalysis,
        heuristic_analysis: ConversationAnalysis | None = None,
        *,
        blocked_reason: str | None = None,
    ) -> PlannerDecision:
        if blocked_reason is not None:
            return PlannerDecision(
                mode="bypassed",
                selected_strategy=analysis.mi_strategies[0] if analysis.mi_strategies else None,
                selection_reason=f"未进入普通策略规划：{blocked_reason}",
            )

        try:
            memories = retrieve_strategy_memories(
                session_state,
                analysis,
                top_k=self.config.memory_top_k,
                min_score=self.config.memory_min_score,
            )
            candidates = generate_candidate_strategies(
                analysis,
                session_state,
                memories,
                heuristic=heuristic_analysis,
                config=self.config,
            )
            scored = self.fast.rank(session_state, analysis, candidates, memories)
            if not scored:
                return PlannerDecision(mode="fallback", selection_reason="没有可用候选策略")

            assessment = self.router.assess(session_state, analysis, memories, scored)
            secondary = scored[1].strategy if len(scored) > 1 else None

            if assessment.mode == "fast":
                best = scored[0]
                return PlannerDecision(
                    mode="fast",
                    familiarity_score=assessment.familiarity,
                    confidence_score=assessment.state_confidence,
                    score_margin=assessment.margin,
                    conflict_score=assessment.conflict,
                    planner_confidence=assessment.confidence,
                    candidate_strategies=scored,
                    selected_strategy=best.strategy,
                    secondary_strategy=secondary,
                    selection_reason=(
                        f"熟悉且高置信（confidence={assessment.confidence:.2f}），"
                        f"直接采用效用最高的 {best.strategy.value}"
                    ),
                    expected_next_goal=analysis.next_goal,
                )

            paths = self.slow.search(analysis, scored, memories)
            if not paths:
                return PlannerDecision(mode="fallback", selection_reason="前瞻搜索无结果")

            best_path = paths[0]
            expected_goal = None
            if best_path.best_next is not None:
                next_state = simulate_next_state(initial_planning_state(analysis), best_path.first_strategy)
                expected_goal = NEXT_GOAL_BY_STAGE.get(next_state.stage)

            return PlannerDecision(
                mode="slow",
                familiarity_score=assessment.familiarity,
                confidence_score=assessment.state_confidence,
                score_margin=assessment.margin,
                conflict_score=assessment.conflict,
                planner_confidence=assessment.confidence,
                candidate_strategies=scored,
                selected_strategy=best_path.first_strategy,
                secondary_strategy=secondary,
                selection_reason=(
                    f"置信度不足（confidence={assessment.confidence:.2f}），"
                    f"深度2前瞻后选择 {best_path.first_strategy.value}"
                    f"（总路径分 {best_path.total:.3f}）"
                ),
                lookahead_paths=paths,
                expected_next_goal=expected_goal,
            )
        except Exception as exc:  # planner must never take the conversation down
            logger.warning("Strategy planner failed; keeping heuristic strategies: %s", exc)
            return PlannerDecision(
                mode="fallback",
                selection_reason=f"规划器异常，回退到启发式策略：{type(exc).__name__}",
            )
