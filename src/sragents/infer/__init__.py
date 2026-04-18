"""Stage 2: inference (SkillProvider × InferenceEngine).

Public API::

    from sragents.infer import get_provider, get_engine
    from sragents.infer.base import InferenceResult

Built-in providers and engines are registered on import.
"""

# Trigger registration of built-in providers & engines.
from sragents.infer import providers as _providers  # noqa: F401
from sragents.infer import engines as _engines  # noqa: F401
from sragents.infer.base import (
    InferenceEngine,
    InferenceResult,
    SkillProvider,
    get_engine,
    get_provider,
    list_engines,
    list_providers,
    register_engine,
    register_provider,
)

__all__ = [
    "InferenceResult",
    "SkillProvider",
    "InferenceEngine",
    "get_provider",
    "get_engine",
    "list_providers",
    "list_engines",
    "register_provider",
    "register_engine",
]
