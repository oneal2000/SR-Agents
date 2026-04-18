"""Shared CLI helpers."""

import json
import sys
from pathlib import Path


def parse_kv_list(items: list[str]) -> dict:
    """Parse ``["k=v", "k2=v2"]`` into ``{"k": v, "k2": v2}``.

    Values are parsed as JSON if possible (so numbers and booleans come
    through typed); otherwise kept as strings. Raises :class:`SystemExit`
    with a readable message on malformed input, so callers don't need to
    wrap in try/except.
    """
    out: dict = {}
    for item in items:
        if "=" not in item:
            sys.exit(
                f"sragents: error: expected KEY=VALUE, got {item!r}. "
                "Use --provider-arg source=path/to/file.json (for example)."
            )
        k, v = item.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def require_exists(path: Path | None, what: str) -> Path:
    """Exit with a clean error message if ``path`` does not exist.

    Call this from subcommand ``run()`` functions (not from argparse
    ``type=`` converters). Returns the validated ``Path`` on success.
    """
    if path is None:
        sys.exit(f"sragents: error: missing --{what}")
    p = Path(path)
    if not p.exists():
        sys.exit(f"sragents: error: {what} not found: {p}")
    return p
