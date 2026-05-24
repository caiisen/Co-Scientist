from __future__ import annotations

from co_scientist.memory.models import Task

from .base import Agent, AgentContext
from .results import AgentResult, AgentResultKind


class EchoAgent(Agent):
    name = "echo"
    system_prompt = "You echo the rendered task payload for integration tests."

    async def execute(self, task: Task, ctx: AgentContext) -> AgentResult:
        prompt = self.render_prompt(
            ctx,
            task.payload_json.get("template", "echo"),
            text=task.payload_json.get("text", ""),
        )
        return AgentResult(
            kind=AgentResultKind.NOOP,
            payload={"prompt": prompt, "agent": self.name},
            raw_text=prompt,
        )

