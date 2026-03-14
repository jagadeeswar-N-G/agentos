"""
Converstaion memory reddis backend for the stateful memory

Each conversation maintains
- short term memory (N last messages , slidding window )
- Long term : summarized history
- Meta data: ticket_id, user_id, agent_type, timestamps

"""

import json
import os
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass, asdict

import redis.asyncio as redis
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage


@dataclass
class ConversationTurn:
    role:str
    content: str
    timestamp: str
    model_used:Optional[str] = None
    tokens_used: Optional[int] = None

class ConversationMemory:
    """
    Redis conversation memory with sliding window plus summarization.
    """
    SHORT_TERM_WINDOW = 10
    TTL_SECONDS = 60*60*24

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
 
    async def _get_client(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379"),
                encoding="utf-8",
                decode_responses=True,
            )
        return self._redis
 
    def _key(self, conversation_id: str) -> str:
        return f"agentOS:conv:{conversation_id}"
 
    def _meta_key(self, conversation_id: str) -> str:
        return f"agentOS:meta:{conversation_id}"
    
    async def ping(self) -> bool:
      """Health check — returns True if Redis is reachable."""
      try:
        client = await self._get_client()
        return await client.ping()
      except Exception:
        return False
 
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        tokens_used: Optional[int] = None,
    ) -> None:
        client = await self._get_client()
        turn = ConversationTurn(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_used=model_used,
            tokens_used=tokens_used,
        )
        key = self._key(conversation_id)
        await client.rpush(key, json.dumps(asdict(turn)))
        await client.expire(key, self.TTL_SECONDS)
 
    async def get_messages(
        self,
        conversation_id: str,
        last_n: Optional[int] = None,
    ) -> list[BaseMessage]:
        """Returns LangChain-compatible message list."""
        client = await self._get_client()
        key = self._key(conversation_id)
        n = last_n or self.SHORT_TERM_WINDOW
        raw = await client.lrange(key, -n, -1)
 
        messages: list[BaseMessage] = []
        for item in raw:
            turn = json.loads(item)
            if turn["role"] == "human":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "ai":
                messages.append(AIMessage(content=turn["content"]))
            elif turn["role"] == "system":
                messages.append(SystemMessage(content=turn["content"]))
        return messages
 
    async def get_full_history(self, conversation_id: str) -> list[dict]:
        """Returns raw history with metadata — for analytics/dashboard."""
        client = await self._get_client()
        raw = await client.lrange(self._key(conversation_id), 0, -1)
        return [json.loads(item) for item in raw]
 
    async def set_metadata(self, conversation_id: str, metadata: dict) -> None:
        client = await self._get_client()
        key = self._meta_key(conversation_id)
        await client.hset(key, mapping={k: json.dumps(v) for k, v in metadata.items()})
        await client.expire(key, self.TTL_SECONDS)
 
    async def get_metadata(self, conversation_id: str) -> dict:
        client = await self._get_client()
        raw = await client.hgetall(self._meta_key(conversation_id))
        return {k: json.loads(v) for k, v in raw.items()}
 
    async def clear(self, conversation_id: str) -> None:
        client = await self._get_client()
        await client.delete(self._key(conversation_id))
        await client.delete(self._meta_key(conversation_id))
 
 
# Singleton
memory = ConversationMemory()