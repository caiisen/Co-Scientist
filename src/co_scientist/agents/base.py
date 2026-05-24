from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

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
    prompt_store: PromptTemplateStore = field(default_factory=PromptTemplateStore)

    def llm_for(self, agent_name: str | None = None) -> LLMClient:
        return self.llm_router.client_for(agent_name)


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
        return await ctx.llm_for(self.name).chat(messages, **kwargs)

