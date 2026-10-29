"""Butterfly: a self-correcting constraint architecture for adaptive AI agents.

Reference implementation of the white paper by RJ Thomas (docs/).
"""

from .agent import ButterflyAgent, TurnResult
from .config import ButterflyConfig
from .constraints import BehavioralConstraint, ConstraintSet, prune_constraints
from .extractor import ConstraintExtractor
from .llm import MockLLM, OllamaClient, OpenAICompatibleClient
from .quality import ResponseQualityEvaluator
from .traits import TraitProfile

__version__ = "2.0.0"
__all__ = [
    "ButterflyAgent", "TurnResult", "ButterflyConfig", "BehavioralConstraint",
    "ConstraintSet", "prune_constraints", "ConstraintExtractor", "MockLLM",
    "OllamaClient", "OpenAICompatibleClient", "ResponseQualityEvaluator", "TraitProfile",
]
