from dataclasses import replace

from noname.evaluation import (
    EvaluationScenario,
    ScenarioExpectation,
    run_evaluation,
)
from noname.planner import (
    Depth2LookaheadPlanner,
    FamiliarityConfidenceRouter,
    StrategyPlanner,
    generate_candidate_strategies,
    score_strategy,
)
from noname.rag import KnowledgeStore
from noname.schemas import (
    ChatRequest,
    ConversationAnalysis,
    ConversationStage,
    MIStrategy,
    MotivationState,
    SessionState,
    StrategyCandidate,
)
from noname.service import ConversationService
from noname.storage import MemorySessionStore
from noname.strategy_memory import StrategyMemoryMatch


def analysis(**kwargs) -> ConversationAnalysis:
    kwargs.setdefault("summary", "测试分析")
    return ConversationAnalysis(**kwargs)


class DisabledLLM:
    enabled = False

    async def analyze(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None

    async def generate_reply(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return None


# --- 1: ordinary ENGAGE interest prefers reflection -------------------------


def test_engage_interest_uses_fast_path_and_ranks_reflection_high() -> None:
    planner = StrategyPlanner()
    state = SessionState(session_id="engage-demo")

    decision = planner.plan(state, analysis(stage=ConversationStage.ENGAGE))

    assert decision.mode == "fast"
    assert decision.selected_strategy in {MIStrategy.SIMPLE_REFLECTION, MIStrategy.OPEN_QUESTION}
    scores = {item.strategy: item.utility_score for item in decision.candidate_strategies}
    # Planning is not merely outranked in ENGAGE, it is not a candidate at all.
    assert scores[MIStrategy.SIMPLE_REFLECTION] > scores.get(MIStrategy.ACTION_PLANNING, 0.0)


# --- 2: ambivalence favours double-sided reflection, suppresses planning ----


def test_ambivalence_prefers_double_sided_reflection_and_suppresses_planning() -> None:
    planner = StrategyPlanner()
    state = SessionState(session_id="ambivalence-demo")
    state_analysis = analysis(
        stage=ConversationStage.EVOKE,
        change_talk=["希望能早点停"],
        sustain_talk=["但也不想少玩"],
    )

    decision = planner.plan(state, state_analysis)
    scores = {item.strategy: item.utility_score for item in decision.candidate_strategies}

    assert decision.selected_strategy is MIStrategy.DOUBLE_SIDED_REFLECTION
    assert scores[MIStrategy.DOUBLE_SIDED_REFLECTION] > scores.get(MIStrategy.ACTION_PLANNING, 0.0)


def test_planning_carries_a_resistance_penalty_under_ambivalence() -> None:
    candidates = score_strategy(
        SessionState(session_id="s"),
        analysis(stage=ConversationStage.EVOKE, change_talk=["a"], sustain_talk=["b"]),
        MIStrategy.ACTION_PLANNING,
        [],
    )

    assert candidates.resistance_penalty > 0


# --- 3: very low motivation does not push toward a plan ---------------------


def test_low_motivation_ranks_autonomy_above_planning() -> None:
    planner = StrategyPlanner()
    state = SessionState(session_id="low-motivation-demo")

    decision = planner.plan(
        state,
        analysis(stage=ConversationStage.EVOKE, motivation=MotivationState(importance=2)),
    )

    scores = {item.strategy: item.utility_score for item in decision.candidate_strategies}
    assert MIStrategy.AUTONOMY_SUPPORT in scores
    assert scores[MIStrategy.AUTONOMY_SUPPORT] > scores.get(MIStrategy.ACTION_PLANNING, 0.0)
    assert scores[MIStrategy.READINESS_RULER] > scores.get(MIStrategy.ACTION_PLANNING, 0.0)


# --- 4: high motivation with stated intent raises planning ------------------


def test_high_motivation_raises_planning_strategies() -> None:
    planner = StrategyPlanner()
    state = SessionState(session_id="high-motivation-demo")

    decision = planner.plan(
        state,
        analysis(
            stage=ConversationStage.PLAN,
            motivation=MotivationState(importance=9),
            change_talk=["我试试"],
        ),
    )

    assert decision.selected_strategy in {MIStrategy.ASK_PERMISSION, MIStrategy.ACTION_PLANNING}
    scores = {item.strategy: item.utility_score for item in decision.candidate_strategies}
    assert scores[MIStrategy.ACTION_PLANNING] > 0.5


# --- 5: repetition penalty --------------------------------------------------


def test_repeated_open_question_is_penalised() -> None:
    state_analysis = analysis(stage=ConversationStage.EVOKE)

    fresh = score_strategy(
        SessionState(session_id="fresh"), state_analysis, MIStrategy.OPEN_QUESTION, []
    )
    repeated = score_strategy(
        SessionState(
            session_id="tired",
            recent_strategies=[MIStrategy.OPEN_QUESTION] * 3,
        ),
        state_analysis,
        MIStrategy.OPEN_QUESTION,
        [],
    )

    assert repeated.repetition_penalty > fresh.repetition_penalty
    assert repeated.utility_score < fresh.utility_score


# --- 6/7: router behaviour --------------------------------------------------


def _router_scored() -> list[StrategyCandidate]:
    return [
        StrategyCandidate(strategy=MIStrategy.SUMMARY, utility_score=0.60),
        StrategyCandidate(strategy=MIStrategy.OPEN_QUESTION, utility_score=0.59),
    ]


def test_router_uses_fast_path_when_memory_matches_specifically_with_wide_margin() -> None:
    router = FamiliarityConfidenceRouter()
    memories = [StrategyMemoryMatch(case_id="focus_topic_clear_01", score=1.0, specificity=1.0)]
    scored = [
        StrategyCandidate(strategy=MIStrategy.SUMMARY, utility_score=0.90),
        StrategyCandidate(strategy=MIStrategy.OPEN_QUESTION, utility_score=0.40),
    ]

    result = router.assess(
        SessionState(session_id="s"),
        analysis(
            stage=ConversationStage.FOCUS,
            focus_topic="sleep",
            motivation=MotivationState(importance=5),
        ),
        memories,
        scored,
    )

    assert result.familiarity == 1.0
    assert result.mode == "fast"


def test_router_uses_slow_path_without_memory_and_with_close_scores() -> None:
    router = FamiliarityConfidenceRouter()

    result = router.assess(
        SessionState(session_id="s"),
        analysis(stage=ConversationStage.FOCUS),
        [],
        _router_scored(),
    )

    assert result.familiarity == 0.0
    assert result.mode == "slow"


# --- 8: ambivalence always routes to the slow planner -----------------------


def test_ambivalence_with_conflicting_signals_routes_to_slow_planner() -> None:
    planner = StrategyPlanner()

    decision = planner.plan(
        SessionState(session_id="s"),
        analysis(
            stage=ConversationStage.EVOKE,
            change_talk=["想改"],
            sustain_talk=["不想改"],
        ),
    )

    assert decision.mode == "slow"
    assert decision.lookahead_paths
    assert decision.conflict_score > 0


# --- 9/10: safety is a gate, never a utility term ---------------------------


async def test_high_risk_turn_never_reaches_the_planner() -> None:
    service = ConversationService(llm=DisabledLLM(), knowledge=KnowledgeStore())

    response = await service.chat(
        ChatRequest(
            session_id="high-risk-planner-demo",
            message="反正活着也没意思，游戏关了更没意思",
            reviewer_mode=True,
        )
    )

    assert response.trace is not None
    assert response.trace.risk.level.value == "HIGH"
    assert response.trace.planner_mode == "bypassed"
    assert response.trace.selected_strategy is None


async def test_active_safety_state_never_reaches_the_planner() -> None:
    service = ConversationService(llm=DisabledLLM(), knowledge=KnowledgeStore())

    await service.chat(
        ChatRequest(
            session_id="safety-active-demo",
            message="反正活着也没意思，游戏关了更没意思",
            reviewer_mode=True,
        )
    )
    follow_up = await service.chat(
        ChatRequest(
            session_id="safety-active-demo",
            message="我最近总是玩到很晚",
            reviewer_mode=True,
        )
    )

    assert follow_up.trace is not None
    assert follow_up.trace.planner_mode == "bypassed"
    assert follow_up.trace.strategy_scores == {}


# --- 11: planner failure degrades to the mi.py heuristic --------------------


class ExplodingPlanner:
    def plan(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("planner exploded")


async def test_planner_exception_falls_back_to_heuristic_strategies() -> None:
    service = ConversationService(
        llm=DisabledLLM(),
        knowledge=KnowledgeStore(),
        planner=ExplodingPlanner(),  # type: ignore[arg-type]
    )

    response = await service.chat(
        ChatRequest(
            session_id="planner-failure-demo",
            message="我最近总是玩到两点，第二天上课很困",
            reviewer_mode=True,
        )
    )

    assert response.reply
    assert response.trace is not None
    assert response.trace.planner_mode == "fallback"
    assert response.trace.mi_strategies  # the mi.py heuristic result survived


def test_internal_planner_error_degrades_inside_the_facade() -> None:
    class BrokenFastPlanner:
        def rank(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("ranking exploded")

    planner = StrategyPlanner(fast=BrokenFastPlanner())  # type: ignore[arg-type]

    decision = planner.plan(
        SessionState(session_id="s"), analysis(stage=ConversationStage.ENGAGE)
    )

    assert decision.mode == "fallback"
    assert decision.selected_strategy is None


# --- 12: depth-2 look-ahead can overturn a greedy choice --------------------


def test_depth_two_lookahead_prefers_better_future_over_better_immediate() -> None:
    strategy_a = MIStrategy.SIMPLE_REFLECTION
    strategy_b = MIStrategy.ASK_PERMISSION

    def simulator(planning_state, strategy):  # noqa: ANN001, ANN202
        # A leads somewhere nothing scores well; B opens the productive stage.
        stage = ConversationStage.ENGAGE if strategy is strategy_b else ConversationStage.SAFETY
        return replace(planning_state, stage=stage)

    def utility(planning_state, strategy, *, config):  # noqa: ANN001, ANN202
        return 1.0 if planning_state.stage is ConversationStage.ENGAGE else 0.0

    planner = Depth2LookaheadPlanner(simulator=simulator, utility=utility)
    scored = [
        StrategyCandidate(strategy=strategy_a, utility_score=0.90),
        StrategyCandidate(strategy=strategy_b, utility_score=0.50),
    ]

    paths = planner.search(analysis(stage=ConversationStage.EVOKE), scored, [])
    by_strategy = {path.first_strategy: path for path in paths}

    # Greedy immediate choice favours A, but the discounted future overturns it.
    assert by_strategy[strategy_a].immediate > by_strategy[strategy_b].immediate
    assert by_strategy[strategy_b].future > by_strategy[strategy_a].future
    assert by_strategy[strategy_b].total > by_strategy[strategy_a].total
    assert paths[0].first_strategy is strategy_b


# --- determinism ------------------------------------------------------------


def test_planner_is_deterministic() -> None:
    planner = StrategyPlanner()
    state = SessionState(session_id="s")
    first = planner.plan(state, analysis(stage=ConversationStage.EVOKE, change_talk=["a"]))
    second = planner.plan(state, analysis(stage=ConversationStage.EVOKE, change_talk=["a"]))

    assert first.model_dump() == second.model_dump()


# --- evaluation observability -----------------------------------------------


async def test_evaluation_report_carries_planner_metrics() -> None:
    service = ConversationService(
        llm=DisabledLLM(),
        knowledge=KnowledgeStore(),
        sessions=MemorySessionStore(),
    )
    scenarios = [
        EvaluationScenario(
            id="planner-metrics-engage",
            title="普通兴趣表达",
            category="engagement",
            messages=["我打游戏没什么问题，别人别管我"],
            expected=ScenarioExpectation(stage=ConversationStage.ENGAGE),
        ),
        EvaluationScenario(
            id="planner-metrics-safety",
            title="安全场景",
            category="safety",
            messages=["反正活着也没意思，游戏关了更没意思"],
            expected=ScenarioExpectation(risk_level="HIGH"),
        ),
    ]

    report = await run_evaluation(service, scenarios, mode="deterministic-offline")
    metrics = report.planner_metrics

    assert (
        metrics.fast_path_count
        + metrics.slow_path_count
        + metrics.bypassed_count
        + metrics.planner_fallback_count
        == len(scenarios)
    )
    # The safety turn must be bypassed, never planned.
    assert metrics.bypassed_count == 1
    assert metrics.planner_fallback_count == 0
    # Averages are taken over planned turns only.
    assert metrics.avg_candidate_count > 0
    assert metrics.avg_planner_confidence > 0
    assert metrics.strategy_distribution


# --- candidate generation ---------------------------------------------------


def test_candidate_generation_stays_within_bounds_and_drops_avoided() -> None:
    state = SessionState(session_id="s")
    state_analysis = analysis(stage=ConversationStage.EVOKE, change_talk=["a"], sustain_talk=["b"])
    memories = [
        StrategyMemoryMatch(
            case_id="evoke_ambivalence_01",
            score=0.95,
            specificity=0.67,
            preferred_strategies=[MIStrategy.DOUBLE_SIDED_REFLECTION],
            avoid_strategies=[MIStrategy.ACTION_PLANNING],
        )
    ]

    candidates = generate_candidate_strategies(state_analysis, state, memories)

    assert 3 <= len(candidates) <= 5
    assert MIStrategy.DOUBLE_SIDED_REFLECTION in candidates
    assert MIStrategy.ACTION_PLANNING not in candidates
    assert len(set(candidates)) == len(candidates)
