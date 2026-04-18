"""``sragents infer`` — Stage 2."""

import inspect
import json
from pathlib import Path

from sragents.cli._common import parse_kv_list, require_exists
from sragents.infer import get_engine, get_provider, list_engines, list_providers
from sragents.infer.runner import run_many
from sragents.infer.base import _ENGINES, _PROVIDERS  # noqa: WPS437
from sragents.llm import create_llm_client


def _accepts_kwarg(factory, name: str) -> bool:
    """True if ``factory(**kwargs)`` would accept ``kwargs[name]``."""
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return True  # be permissive if we can't introspect
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == name:
            return True
    return False


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "infer", help="Run LLM inference (stage 2)",
        description=(
            "Combines a SkillProvider (how skills are obtained) with an "
            "InferenceEngine (how they are used) to produce one "
            "raw_output per instance."
        ),
    )
    # Required
    p.add_argument("--instances", type=Path, required=True,
                   help="Path to a bench instances JSON "
                        "(e.g. data/bench/instances/theoremqa.json)")
    p.add_argument("--output", type=Path, required=True,
                   help="Output JSONL path. Re-invoking with the same path resumes.")
    p.add_argument("--model", required=True,
                   help="Model name or path passed to the OpenAI-compatible server")

    # Provider × Engine
    p.add_argument(
        "--provider", required=True,
        help=f"Provider name. Built-in: {', '.join(list_providers()) or '(loading...)'}",
    )
    p.add_argument(
        "--provider-arg", action="append", default=[],
        metavar="KEY=VALUE",
        help="Provider-specific argument (repeatable). Example: "
             "--provider-arg source=bm25.json --provider-arg k=1",
    )
    p.add_argument(
        "--engine", default="direct",
        help=f"Engine name. Built-in: {', '.join(list_engines()) or '(loading...)'}",
    )
    p.add_argument(
        "--engine-arg", action="append", default=[],
        metavar="KEY=VALUE",
        help="Engine-specific argument (repeatable). Example: "
             "--engine-arg temperature=0.5",
    )

    # Standard
    p.add_argument("--api-base", default=None,
                   help="OpenAI-compatible endpoint (default: $OPENAI_API_BASE)")
    p.add_argument("--workers", type=int, default=32,
                   help="Parallel workers (default: 32; reduce if rate-limited)")
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (forwarded to engines that "
                        "accept it; default: 0.7)")
    p.add_argument("--max-tokens", type=int, default=4096,
                   help="Max tokens per generation (forwarded to engines "
                        "that accept it; default: 4096)")
    p.add_argument("--label", default=None,
                   help="Method label for output records. "
                        "Default: {provider}_{engine}.")
    p.add_argument("--thinking", action="store_true",
                   help="Enable thinking mode for hybrid-thinking models "
                        "(Qwen3 / GLM-5 / Kimi)")
    p.add_argument("--force", action="store_true",
                   help="Truncate the output file before running "
                        "(discards existing per-instance resume data)")
    p.set_defaults(func=run)


def run(args) -> None:
    require_exists(args.instances, "instances")
    instances = json.loads(args.instances.read_text())

    provider_kwargs = parse_kv_list(args.provider_arg)
    engine_kwargs = parse_kv_list(args.engine_arg)

    # Forward common engine kwargs only when the engine factory declares
    # them, so third-party engines with minimal signatures don't crash.
    engine_factory = _ENGINES.get(args.engine)
    for key, value in (
        ("temperature", args.temperature),
        ("max_tokens", args.max_tokens),
        ("thinking", args.thinking if args.thinking else None),
    ):
        if value is None:
            continue
        if engine_factory is None or _accepts_kwarg(engine_factory, key):
            engine_kwargs.setdefault(key, value)

    # Providers that drive their own LLM call (e.g. llm_select) need
    # model + api_base. Forward them automatically when the provider
    # factory declares them. External providers opt in by adding the
    # same parameter names to their ``__init__``.
    provider_factory = _PROVIDERS.get(args.provider)
    for key, value in (
        ("model", args.model),
        ("api_base", args.api_base),
    ):
        if provider_factory is not None and _accepts_kwarg(provider_factory, key):
            provider_kwargs.setdefault(key, value)

    label = args.label or f"{args.provider}_{args.engine}"

    if args.force and args.output.exists():
        args.output.unlink()

    client = create_llm_client(api_base=args.api_base)
    provider = get_provider(args.provider, **provider_kwargs)
    engine = get_engine(args.engine, **engine_kwargs)

    print(f"Provider: {args.provider} {provider_kwargs}")
    print(f"Engine:   {args.engine}   {engine_kwargs}")
    print(f"Label:    {label}")
    print(f"Model:    {args.model}")

    run_many(
        instances=instances,
        provider=provider,
        engine=engine,
        client=client,
        model=args.model,
        output_path=args.output,
        label=label,
        workers=args.workers,
    )
