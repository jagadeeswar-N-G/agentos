import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

class TaskType(str, Enum):
    CLASSIFY = "classify"
    FAQ_RESOLVE = "faq_resolve"
    COMPLEX_REASON = "complex_reason"
    ESCALATION = "escalation"
    DRAFT_EMAIL = "draft_email"

@dataclass
class RouterDecision:
    model_name: str
    model_provider: str
    reasoning: str
    estimated_cost_tier: str

ROUTING_TABLE = {
    TaskType.CLASSIFY: RouterDecision(
        model_name="gpt-4o-mini",
        model_provider="openai",
        reasoning="classification needs speed and low cost",
        estimated_cost_tier="low"
    ),
    TaskType.FAQ_RESOLVE: RouterDecision(
        model_name="gpt-4o-mini",
        model_provider="openai",
        reasoning="faq resolution needs accuracy and low cost",
        estimated_cost_tier="low"
    ),
    TaskType.COMPLEX_REASON: RouterDecision(
        model_name="gpt-4o-mini",
        model_provider="openai",
        reasoning="complex reasoning needs accuracy and high cost",
        estimated_cost_tier="high"
    ),
    TaskType.ESCALATION: RouterDecision(
        model_name="gpt-4o-mini",
        model_provider="openai",
        reasoning="escalation needs accuracy and high cost",
        estimated_cost_tier="high"
    ),
    TaskType.DRAFT_EMAIL: RouterDecision(
        model_name="gpt-4o-mini",
        model_provider="openai",
        reasoning="draft email needs accuracy and low cost",
        estimated_cost_tier="low"
    ),
}

class LLMRouter:
    def __init__(self):
        self._openai_key = os.getenv("OPENAI_API_KEY")
        self._anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._model_cache = {}

    def get_model(self, task_type: TaskType, temperature: float = 0.0):
        decision = ROUTING_TABLE[task_type]
        cache_key = f"{task_type.value}_{temperature}"

        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = self._build_model(decision, temperature)

        return self._model_cache[cache_key], decision

    def _build_model(self, decision: RouterDecision, temperature: float):
        if decision.model_provider == "openai":
            return ChatOpenAI(
                model=decision.model_name,
                temperature=temperature,
                api_key=self._openai_key
            )
        elif decision.model_provider == "anthropic":
            return ChatAnthropic(
                model=decision.model_name,
                temperature=temperature,
                api_key=self._anthropic_key
            )
        else:
            raise ValueError(f"Invalid provider: {decision.model_provider}")

llm_router = LLMRouter()