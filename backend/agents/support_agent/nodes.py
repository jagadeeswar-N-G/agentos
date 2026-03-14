import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from agents.state import SupportAgentState
from core.llm_router import llm_router, TaskType


logger = logging.getLogger(__name__)

async def classification_node(state: SupportAgentState) -> SupportAgentState:
    model, decision = llm_router.get_model(TaskType.CLASSIFY)
    messages = [
        SystemMessage(content="""You are a support ticket classification agent.
        Classify the customer message into one of the following categories:
        - L1_SUPPORT: Simple questions answerable from knowledge base
        - L1_TECHNICAL: Basic technical issues
        - L2_COMPLEX: Complex issues requiring investigation
        - L2_BILLING: Billing disputes or refund requests
        - SPAM: Irrelevant or abusive messages
        - URGENT: System down or business critical

        Respond ONLY in valid JSON with no extra text:
        {
            "category": "<category>",
            "confidence": 0.0,
            "reasoning": "<one sentence>",
            "sentiment": "positive|neutral|negative|frustrated|angry"
        }
        """),
        HumanMessage(content=f"Classify this message: {state['user_message']}")
    ]

    response = await model.ainvoke(messages)

    try:
        classification = json.loads(response.content)
    except json.JSONDecodeError:
        classification = {
            "category": "L1_FAQ",
            "confidence": 0.5,
            "reasoning": "Classification failed - defaulting to L1",
            "sentiment": "neutral"
        }

    return {
        **state,
        "classification": classification,
        "model_used_classify": decision.model_name
    }



async def rag_resolve_node(state: SupportAgentState) -> SupportAgentState:
    from qdrant_client import AsyncQdrantClient
    from langchain_openai import OpenAIEmbeddings
    import os

    # embed the users question
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    query_vector = await embedder.aembed_query(state["user_message"])

    # search for the Qdrant for relevant articles
    client = AsyncQdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
    
    results = await client.search(
        collection_name = "support_knowledge_base",
        query_vector=query_vector,
        limit =3,
        score_threshold=0.3
        
    )
   # ADD THESE
    print(f"🔍 Search results count: {len(results)}")
    for r in results:
        print(f"📄 Score: {r.score} | Title: {r.payload.get('title')}")

     # Step 3 — format results into context string
    context = ""
    for r in results:
        context += f"\n{r.payload.get('title')}\n{r.payload.get('content')}\n---"

    # Step 4 — generate response using context
    model, decision = llm_router.get_model(TaskType.FAQ_RESOLVE, temperature=0.3)

    messages = [
        SystemMessage(content=f"""You are a helpful support agent.
        Answer the customer's question using ONLY this knowledge base context:
        
        {context}
        
        If context doesn't answer the question say so honestly."""),
        HumanMessage(content=state["user_message"])
    ]

    response = await model.ainvoke(messages)
    
    resolved = len(results) > 0 and results[0].score > 0.5

    return {
        **state,
        "kb_context": context,
        "draft_response": response.content,
        "resolved": resolved,
        "model_used_resolve": decision.model_name
    }


async def escalate_node(state: SupportAgentState) -> SupportAgentState:
    
    model, decision = llm_router.get_model(TaskType.ESCALATION, temperature=0.0)
    
    messages = [
        SystemMessage(content="""You are a senior support analyst.
        Summarize this customer issue for a human agent in under 150 words.
        Include:
        1. What the customer's problem is
        2. Urgency level (LOW/MEDIUM/HIGH/CRITICAL)
        3. Recommended next steps
        Be precise. Human agents are busy."""),
        HumanMessage(content=f"""
        Customer message: {state['user_message']}
        Classification: {json.dumps(state.get('classification', {}))}
        KB search attempted: {bool(state.get('kb_context'))}
        """)
    ]
    
    response = await model.ainvoke(messages)
    
    escalation_response = """I've escalated your issue to our specialist team. 
    A support agent will reach out within 2-4 business hours."""
    
    return {
        **state,
        "escalation_summary": response.content,
        "final_response": escalation_response,
        "ticket_created": True,
        "model_used_escalate": decision.model_name
    }


def route_after_classify(state: SupportAgentState):
    category = state["classification"].get("category", "L1_FAQ")
    confidence = state["classification"].get("confidence", 0.5)

    if category == "SPAM":
        return "spam_filter"
    elif category.startswith("L2") or category == "URGENT":
        return "escalate"
    elif confidence < 0.5:
        return "escalate"
    else:
        return "rag_resolve"
def route_after_rag(state: SupportAgentState):
    if state.get("resolved"):
        return "quality_check"
    return "escalate"

async def respond_node(state: SupportAgentState) -> SupportAgentState:
    return {
        **state,
        "final_response": state.get("draft_response"),
    }
    
