"""
Evaluations — LangSmith-backed eval pipeline.

Runs after every agent response to track:
  - relevance:     did the response answer the question?
  - groundedness:  is the response grounded in KB context?
  - hallucination: did the agent make things up?
  - latency:       how fast was the response?
  - resolution:    was the ticket resolved or escalated?

All evals are async and non-blocking — they never slow down the response.
Results are logged to LangSmith and local metrics store.
"""

import os
import time
import logging
import asyncio
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Eval result schema
# ─────────────────────────────────────────────

@dataclass
class EvalScore:
    name:    str
    score:   float       # 0.0 - 1.0
    reason:  str
    passed:  bool        # score >= threshold


@dataclass
class EvalResult:
    conversation_id:  str
    run_id:           Optional[str]
    scores:           list[EvalScore]
    overall_passed:   bool
    latency_ms:       float
    resolved:         bool
    escalated:        bool
    category:         Optional[str]
    timestamp:        float


# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────

THRESHOLDS = {
    "relevance":     0.7,
    "groundedness":  0.6,
    "hallucination": 0.8,   # score = confidence it's NOT hallucinated
}


# ─────────────────────────────────────────────
# LLM-based evaluators
# ─────────────────────────────────────────────

async def _eval_relevance(question: str, response: str, model) -> EvalScore:
    """
    Scores how well the response answers the question.
    Uses LLM-as-judge pattern.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    prompt = f"""You are an evaluator. Score how well the RESPONSE answers the QUESTION.

QUESTION: {question}
RESPONSE: {response}

Return ONLY a JSON object:
{{"score": 0.0, "reason": "one sentence"}}

Score guide:
1.0 = perfectly answers the question
0.7 = mostly answers with minor gaps
0.5 = partially answers
0.3 = barely relevant
0.0 = completely irrelevant or wrong
"""
    try:
        result = await model.ainvoke([
            SystemMessage(content="You are a strict evaluator. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        import json
        data = json.loads(result.content)
        score = float(data.get("score", 0.5))
        return EvalScore(
            name="relevance",
            score=score,
            reason=data.get("reason", ""),
            passed=score >= THRESHOLDS["relevance"],
        )
    except Exception as e:
        logger.error(f"Relevance eval failed: {e}")
        return EvalScore(name="relevance", score=0.5, reason="eval failed", passed=False)


async def _eval_groundedness(response: str, kb_context: str, model) -> EvalScore:
    """
    Scores whether the response is grounded in the KB context.
    Detects hallucinations relative to retrieved documents.
    """
    from langchain_core.messages import SystemMessage, HumanMessage

    if not kb_context:
        return EvalScore(
            name="groundedness",
            score=1.0,
            reason="No KB context used — escalation path, not applicable.",
            passed=True,
        )

    prompt = f"""You are an evaluator checking if a RESPONSE is grounded in CONTEXT.

CONTEXT (knowledge base articles used):
{kb_context[:1500]}

RESPONSE:
{response}

Return ONLY a JSON object:
{{"score": 0.0, "reason": "one sentence"}}

Score guide:
1.0 = every claim in response comes directly from context
0.7 = mostly grounded, minor additions
0.5 = partially grounded
0.3 = significant claims not in context
0.0 = response contradicts or ignores context entirely
"""
    try:
        result = await model.ainvoke([
            SystemMessage(content="You are a strict evaluator. Return only valid JSON."),
            HumanMessage(content=prompt),
        ])
        import json
        data = json.loads(result.content)
        score = float(data.get("score", 0.5))
        return EvalScore(
            name="groundedness",
            score=score,
            reason=data.get("reason", ""),
            passed=score >= THRESHOLDS["groundedness"],
        )
    except Exception as e:
        logger.error(f"Groundedness eval failed: {e}")
        return EvalScore(name="groundedness", score=0.5, reason="eval failed", passed=False)


def _eval_latency(latency_ms: float) -> EvalScore:
    """Score response latency. < 3s = great, < 8s = ok, > 8s = slow."""
    if latency_ms < 3000:
        score, reason = 1.0, f"Fast response: {latency_ms:.0f}ms"
    elif latency_ms < 8000:
        score, reason = 0.7, f"Acceptable latency: {latency_ms:.0f}ms"
    else:
        score, reason = 0.3, f"Slow response: {latency_ms:.0f}ms"

    return EvalScore(
        name="latency",
        score=score,
        reason=reason,
        passed=latency_ms < 8000,
    )


def _eval_resolution(resolved: bool, escalated: bool, category: str) -> EvalScore:
    """
    Score the resolution outcome.
    L1 tickets should resolve. L2 tickets should escalate.
    """
    is_l2 = category and category.startswith("L2")
    is_urgent = category == "URGENT"

    if is_l2 or is_urgent:
        # Correct to escalate
        passed = escalated
        score = 1.0 if escalated else 0.0
        reason = "Correctly escalated complex issue" if escalated else "Should have escalated L2/URGENT"
    else:
        # Should resolve
        score = 1.0 if resolved else 0.5
        passed = resolved
        reason = "Successfully resolved via RAG" if resolved else "L1 issue escalated — check RAG coverage"

    return EvalScore(name="resolution", score=score, reason=reason, passed=passed)


# ─────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────

class AgentEvaluator:
    """
    Runs eval suite after every agent response.
    Logs to LangSmith + local in-memory metrics.
    """

    def __init__(self):
        self._metrics: list[EvalResult] = []   # in-memory for dashboard
        self._eval_model = None

    def _get_model(self):
        """Use cheapest model for evals — cost optimization."""
        if self._eval_model is None:
            from core.llm_router import llm_router, TaskType
            model, _ = llm_router.get_model(TaskType.CLASSIFY)
            self._eval_model = model
        return self._eval_model

    async def evaluate(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        kb_context: str = "",
        latency_ms: float = 0,
        resolved: bool = False,
        escalated: bool = False,
        category: str = None,
        run_id: str = None,
    ) -> EvalResult:
        """
        Run full eval suite. Called as background task — non-blocking.
        """
        model = self._get_model()

        # Run LLM evals concurrently
        relevance_task     = _eval_relevance(user_message, agent_response, model)
        groundedness_task  = _eval_groundedness(agent_response, kb_context, model)

        relevance, groundedness = await asyncio.gather(
            relevance_task,
            groundedness_task,
            return_exceptions=True,
        )

        # Handle exceptions from gather
        if isinstance(relevance, Exception):
            relevance = EvalScore(name="relevance", score=0.5, reason="eval error", passed=False)
        if isinstance(groundedness, Exception):
            groundedness = EvalScore(name="groundedness", score=0.5, reason="eval error", passed=False)

        # Sync evals
        latency_score    = _eval_latency(latency_ms)
        resolution_score = _eval_resolution(resolved, escalated, category or "")

        scores = [relevance, groundedness, latency_score, resolution_score]
        overall_passed = all(s.passed for s in scores)

        result = EvalResult(
            conversation_id=conversation_id,
            run_id=run_id,
            scores=scores,
            overall_passed=overall_passed,
            latency_ms=latency_ms,
            resolved=resolved,
            escalated=escalated,
            category=category,
            timestamp=time.time(),
        )

        # Store locally
        self._metrics.append(result)
        if len(self._metrics) > 1000:
            self._metrics = self._metrics[-1000:]

        # Log to LangSmith
        await self._log_to_langsmith(result)

        # Log summary
        score_summary = {s.name: round(s.score, 2) for s in scores}
        logger.info(f"Eval complete [{conversation_id[:8]}]: {score_summary} | passed={overall_passed}")

        return result

    async def _log_to_langsmith(self, result: EvalResult) -> None:
        """Push eval scores to LangSmith as feedback on the run."""
        try:
            from langsmith import Client
            client = Client()

            if not result.run_id:
                return

            for score in result.scores:
                client.create_feedback(
                    run_id=result.run_id,
                    key=score.name,
                    score=score.score,
                    comment=score.reason,
                )

        except Exception as e:
            logger.debug(f"LangSmith feedback log failed (non-fatal): {e}")

    def get_metrics_summary(self) -> dict:
        """
        Aggregate metrics for dashboard.
        Returns averages across last 100 evals.
        """
        recent = self._metrics[-100:]

        if not recent:
            return {
                "total_evals": 0,
                "pass_rate": 0,
                "avg_latency_ms": 0,
                "resolution_rate": 0,
                "escalation_rate": 0,
                "avg_scores": {},
            }

        total = len(recent)
        passed = sum(1 for r in recent if r.overall_passed)
        resolved = sum(1 for r in recent if r.resolved)
        escalated = sum(1 for r in recent if r.escalated)
        avg_latency = sum(r.latency_ms for r in recent) / total

        # Average per score type
        score_totals: dict[str, list[float]] = {}
        for result in recent:
            for s in result.scores:
                score_totals.setdefault(s.name, []).append(s.score)

        avg_scores = {
            name: round(sum(vals) / len(vals), 3)
            for name, vals in score_totals.items()
        }

        return {
            "total_evals":     total,
            "pass_rate":       round(passed / total, 3),
            "avg_latency_ms":  round(avg_latency, 1),
            "resolution_rate": round(resolved / total, 3),
            "escalation_rate": round(escalated / total, 3),
            "avg_scores":      avg_scores,
        }


# ─────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────

evaluator = AgentEvaluator()