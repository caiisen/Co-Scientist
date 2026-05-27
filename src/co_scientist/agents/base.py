from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from co_scientist.config import AppConfig
from co_scientist.llm.client import ChatMessage, LLMClient, LLMRouter
from co_scientist.memory.models import Task
from co_scientist.memory.store import SQLiteStore
from co_scientist.tools.registry import ToolCallable
from co_scientist.utils.prompts import PromptTemplateStore, build_prompt_messages

from .results import AgentResult


@dataclass
class AgentContext:
    store: SQLiteStore
    llm_router: LLMRouter
    config: AppConfig
    session_id: str
    tools: Mapping[str, ToolCallable] = field(default_factory=dict)
    current_feedback: str | None = None
    http_session: aiohttp.ClientSession | None = None
    prompt_store: PromptTemplateStore = field(default_factory=PromptTemplateStore)
    metrics_sink: Any | None = None

    def llm_for(self, agent_name: str | None = None) -> LLMClient:
        return self.llm_router.client_for(agent_name)

    def embedding_llm_for(self, agent_name: str | None = None) -> LLMClient:
        return self.llm_router.embedding_client_for(agent_name)


class Agent(ABC):
    name: str
    system_prompt: str = "You are a specialized Co-Scientist agent."

    def __init__(self, *, name: str | None = None, system_prompt: str | None = None) -> None:
        if name is not None:
            self.name = name
        if system_prompt is not None:
            self.system_prompt = system_prompt
        if not getattr(self, "name", None):
            raise ValueError("agent name must be set")

    @abstractmethod
    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        """Execute a task and return a typed result for Supervisor follow-up."""

    def render_prompt(
        self,
        ctx: AgentContext,
        template_name: str,
        **variables: Any,
    ) -> str:
        return ctx.prompt_store.render(template_name, **variables)

    def build_messages(
        self,
        ctx: AgentContext,
        *,
        user_prompt: str,
        system_prompt: str | None = None,
        include_feedback: bool = True,
    ) -> list[ChatMessage]:
        return build_prompt_messages(
            system_prompt=system_prompt or self.system_prompt,
            user_prompt=user_prompt,
            feedback=ctx.current_feedback if include_feedback else None,
        )

    async def chat(
        self,
        ctx: AgentContext,
        messages: list[ChatMessage],
        **kwargs: Any,
    ) -> str:
        client = ctx.llm_for(self.name)
        if hasattr(client, "chat_with_metadata"):
            result = await client.chat_with_metadata(messages, **kwargs)
            text = result.text
            metadata = result.metadata
        else:
            text = await client.chat(messages, **kwargs)
            metadata = getattr(client, "last_call", None)
        if ctx.metrics_sink is not None and metadata is not None:
            ctx.metrics_sink.emit(
                ctx.session_id,
                "llm.chat",
                agent=self.name,
                model=metadata.model,
                latency_seconds=metadata.latency_seconds,
                prompt_tokens=metadata.prompt_tokens,
                completion_tokens=metadata.completion_tokens,
                total_tokens=metadata.total_tokens,
            )
        return text
