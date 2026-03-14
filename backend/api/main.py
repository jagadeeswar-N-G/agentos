"""
AgentOS — FastAPI backend
Production-grade: guardrails + memory + evaluations + health checks
"""

import os
import uuid
import logging
import asyncio
import time
from contextlib import asynccontextmanager

# LangSmith must be configured before any LangChain imports
os.environ["LANGCHAIN_TRACING_V2"]  = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGSMITH_PROJECT", "agentos")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from agents.graphy import run_support_agent
from core.guardrails import input_guardrail, output_guardrail, GuardrailAction
from core.memory import memory
from core.evaluations import evaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tickets: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AgentOS starting up...")
    redis_ok = await memory.ping()
    logger.info(f"Redis: {'OK' if redis_ok else 'UNAVAILABLE'}")
    try:
        from qdrant_client import AsyncQdrantClient
        client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        await client.get_collection("support_knowledge_base")
        logger.info("Qdrant: OK")
    except Exception as e:
        logger.warning(f"Qdrant: UNAVAILABLE — {e}")
    logger.info("AgentOS ready.")
    yield
    logger.info("AgentOS shutting down.")


app = FastAPI(title="AgentOS", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SupportChatRequest(BaseModel):
    message:         str           = Field(..., min_length=1, max_length=5000)
    conversation_id: Optional[str] = None
    customer_id:     Optional[str] = "anonymous"


class SupportChatResponse(BaseModel):
    conversation_id:     str
    response:            str
    classification:      Optional[dict]
    resolved:            bool
    escalated:           bool
    model_used:          Optional[dict]
    processing_time_ms:  float
    guardrail_triggered: bool = False
    guardrail_reason:    Optional[str] = None


async def run_eval_background(conversation_id, user_message, agent_response,
                               kb_context, latency_ms, resolved, escalated, category):
    try:
        await evaluator.evaluate(
            conversation_id=conversation_id,
            user_message=user_message,
            agent_response=agent_response,
            kb_context=kb_context,
            latency_ms=latency_ms,
            resolved=resolved,
            escalated=escalated,
            category=category,
        )
    except Exception as e:
        logger.error(f"Background eval failed: {e}")


@app.get("/health")
async def health():
    redis_ok = await memory.ping()
    try:
        from qdrant_client import AsyncQdrantClient
        qc = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        await qc.get_collection("support_knowledge_base")
        qdrant_ok = True
    except Exception:
        qdrant_ok = False
    return {
        "status":  "healthy" if (redis_ok and qdrant_ok) else "degraded",
        "version": "1.0.0",
        "checks":  {"redis": "ok" if redis_ok else "error", "qdrant": "ok" if qdrant_ok else "error"},
    }


@app.post("/api/support/chat", response_model=SupportChatResponse)
async def support_chat(req: SupportChatRequest, background_tasks: BackgroundTasks):
    conversation_id = req.conversation_id or str(uuid.uuid4())
    start_time = time.time()

    # 1. Input guardrail
    guard_result = input_guardrail.run(req.message)
    if guard_result.action == GuardrailAction.BLOCK:
        return SupportChatResponse(
            conversation_id=conversation_id,
            response=f"I'm unable to process that message. {guard_result.reason}",
            classification=None, resolved=False, escalated=False, model_used=None,
            processing_time_ms=round((time.time() - start_time) * 1000, 2),
            guardrail_triggered=True, guardrail_reason=guard_result.reason,
        )

    safe_message = guard_result.safe_content if guard_result.action == GuardrailAction.REDACT else req.message

    # 2. Load memory
    conversation_history = await memory.get_messages(conversation_id, last_n=10)

    # 3. Run agent
    result = await run_support_agent(
        conversation_id=conversation_id,
        customer_id=req.customer_id,
        user_message=safe_message,
        converstaion_history=conversation_history,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Agent failed"))

    agent_response = result.get("response", "")
    classification = result.get("classification", {})
    resolved       = result.get("resolved", False)
    escalated      = result.get("escalated", False)
    models_used    = result.get("models_used", {})
    latency_ms     = result.get("processing_time_ms", 0)
    kb_context     = result.get("kb_context", "")

    # 4. Output guardrail
    out_result     = output_guardrail.run(agent_response)
    final_response = out_result.safe_content if out_result.safe_content else agent_response
    out_triggered  = out_result.action != GuardrailAction.PASS

    # 5. Save to memory
    await memory.add_message(conversation_id, "user", safe_message)
    await memory.add_message(conversation_id, "assistant", final_response)

    # 6. Create ticket
    if escalated:
        ticket_id = f"TKT-{str(uuid.uuid4())[:8].upper()}"
        _tickets[ticket_id] = {
            "id": ticket_id, "conversation_id": conversation_id,
            "customer_id": req.customer_id, "subject": safe_message[:80],
            "category": classification.get("category", "UNKNOWN"),
            "priority": "HIGH" if classification.get("category") == "URGENT" else "MEDIUM",
            "status": "open", "description": result.get("escalation_summary", safe_message),
            "created_at": time.time(),
        }

    # 7. Background evals
    background_tasks.add_task(
        run_eval_background, conversation_id, safe_message, final_response,
        kb_context, latency_ms, resolved, escalated, classification.get("category"),
    )

    return SupportChatResponse(
        conversation_id=conversation_id, response=final_response,
        classification=classification, resolved=resolved, escalated=escalated,
        model_used=models_used, processing_time_ms=latency_ms,
        guardrail_triggered=out_triggered,
        guardrail_reason=out_result.reason if out_triggered else None,
    )


@app.get("/api/tickets")
async def get_tickets():
    tickets = sorted(_tickets.values(), key=lambda t: t["created_at"], reverse=True)
    return {"tickets": tickets, "total": len(tickets)}


@app.get("/api/metrics")
async def get_metrics():
    return evaluator.get_metrics_summary()