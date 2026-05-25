"""Specialized Co-Scientist agent foundations."""

from co_scientist.agents.base import Agent, AgentContext
from co_scientist.agents.evolution import EvolutionAgent
from co_scientist.agents.metareview import MetaReviewAgent
from co_scientist.agents.proximity import ProximityAgent
from co_scientist.agents.ranking import RankingAgent
from co_scientist.agents.results import AgentResult, AgentResultKind

__all__ = [
    "Agent",
    "AgentContext",
    "AgentResult",
    "AgentResultKind",
    "EvolutionAgent",
    "MetaReviewAgent",
    "ProximityAgent",
    "RankingAgent",
]
