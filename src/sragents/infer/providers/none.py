"""Skill-free provider: returns an empty skill list for every instance."""

from sragents.infer.base import register_provider


@register_provider("none")
class NoneProvider:
    """Returns an empty skill list regardless of instance."""

    def provide(self, instance: dict) -> list[dict]:
        return []
