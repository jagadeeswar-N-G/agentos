from typing import Optional, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class SupportAgentState(TypedDict):
    conversation_id: str
    customer_id: str
    user_message: str
    conversation_history: List[BaseMessage]

    #classification
    classification: dict
    model_used_classify: str

    #rag resolution
    kb_context: str
    draft_response: str
    resolved: bool
    model_used_resolve: str

    #escalation
    escalation_summary: str
    ticket_created: bool
    final_response: Optional[str]
    model_used_escalate: str

    #metadata
    processing_time: float