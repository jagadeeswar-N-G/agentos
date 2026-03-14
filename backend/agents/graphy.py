import time
import logging
from langgraph.graph import StateGraph, START, END
from agents.state import SupportAgentState
from agents.support_agent.nodes import (
    classification_node,
    rag_resolve_node,
    escalate_node,
    route_after_classify,
    route_after_rag,
    respond_node
)

logger = logging.getLogger(__name__)


def build_support_agent_graph():
    graph = StateGraph(SupportAgentState)
    graph.add_node("classify", classification_node)
    graph.add_node("rag_resolve", rag_resolve_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("respond", respond_node)

    graph.add_edge(START, "classify")

    graph.add_conditional_edges(
         "classify",
        route_after_classify,
       {
          "rag_resolve": "rag_resolve",
          "escalate": "escalate",
          "spam_filter": "escalate",  # route spam to escalate for now
       },
    )


    graph.add_conditional_edges(
        "rag_resolve",
        route_after_rag,
       {
          "quality_check": "respond",  # no quality_check node yet
          "escalate": "escalate",       # so both go to escalate for now
        },
   )

    graph.add_edge("respond", END)   # ← respond goes to END
    graph.add_edge("escalate", END)
    
    return graph.compile()


support_agent = build_support_agent_graph()


async def run_support_agent(
    conversation_id: str,
    customer_id: str,
    user_message: str,
    converstaion_history: list = None,
) -> dict:
    start_time = time.time()
    initial_state: SupportAgentState = {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "user_message": user_message,
        "converstaion_history": converstaion_history,
    }

    try:
        final_state = await support_agent.ainvoke(initial_state)
        processing_time = (time.time() - start_time) * 1000

        return {
            "success": True,
            "conversation_id": conversation_id,
            "response": final_state.get("final_response"),
            "classification": final_state.get("classification"),
            "resolved": final_state.get("resolved", False),
            "escalated": final_state.get("ticket_created", False),
            "models_used": {
                "classify": final_state.get("model_used_classify"),
                "resolve": final_state.get("model_used_resolve"),
                "escalate": final_state.get("model_used_escalate"),
            },
            "processing_time_ms": round(processing_time, 2),
        }

    except Exception as e:
        logger.error(f"Agent failed: {e}")
        return {
            "success": False,
            "response": "Something went wrong. Please try again.",
            "error": str(e),
        }

    

