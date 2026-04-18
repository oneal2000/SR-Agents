"""``sragents list <what>`` — enumerate registered plugins.

For providers, engines and retrievers this prints the first line of the
implementation's docstring plus its constructor signature, so users can
discover available options without reading source.
"""

import inspect


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "list", help="List registered plugins / datasets / experiments",
    )
    p.add_argument("what", choices=[
        "retrievers", "providers", "engines", "datasets", "experiments",
    ])
    p.set_defaults(func=run)


def _first_line(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    return doc.splitlines()[0] if doc else ""


def _signature_str(factory) -> str:
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return ""
    params = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty:
            params.append(name)
        else:
            params.append(f"{name}={p.default!r}")
    return "(" + ", ".join(params) + ")"


def _print_registry(registry: dict) -> None:
    width = max((len(name) for name in registry), default=0)
    for name in sorted(registry):
        factory = registry[name]
        print(f"  {name:<{width}}  {_signature_str(factory)}")
        doc = _first_line(factory)
        if doc:
            print(f"  {' ' * width}    {doc}")


def run(args) -> None:
    if args.what == "retrievers":
        from sragents.retrieve.base import _REGISTRY
        _print_registry(_REGISTRY)
    elif args.what == "providers":
        from sragents.infer.base import _PROVIDERS
        _print_registry(_PROVIDERS)
    elif args.what == "engines":
        from sragents.infer.base import _ENGINES
        _print_registry(_ENGINES)
    elif args.what == "datasets":
        # A dataset is "runnable" only if it has BOTH a prompt builder and
        # an evaluator. Show each, marking missing halves so extension
        # authors can see what they still need to register.
        from sragents.evaluate.base import _REGISTRY as _EVAL
        from sragents.prompts import _BUILDERS as _PROMPTS
        all_names = sorted(set(_EVAL) | set(_PROMPTS))
        for name in all_names:
            has_eval = name in _EVAL
            has_prompt = name in _PROMPTS
            flags = []
            if not has_eval:
                flags.append("missing evaluator")
            if not has_prompt:
                flags.append("missing prompt builder")
            status = f" ({', '.join(flags)})" if flags else ""
            print(f"  {name}{status}")
    elif args.what == "experiments":
        from sragents.experiments import EXPERIMENTS
        for name, exp in sorted(EXPERIMENTS.items()):
            print(f"  {name}")
            print(f"      {exp.description}")
            for m in exp.methods:
                parts = [f"provider={m.provider}"]
                if m.provider_args:
                    parts.append(f"args={m.provider_args}")
                parts.append(f"engine={m.engine}")
                if m.engine_toolqa:
                    parts.append(f"toolqa→{m.engine_toolqa}")
                display = m.display()
                head = (f"{m.label:<24s} — {display}"
                        if display != m.label else m.label)
                print(f"      • {head}  [{' | '.join(parts)}]")
    else:
        raise ValueError(args.what)
