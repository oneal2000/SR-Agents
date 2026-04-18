"""Parallel per-instance inference with append-based resume.

Orchestrates a Provider × Engine pair over a list of instances, writing
one :class:`~sragents.infer.schema.InferenceRecord` per line to the
output JSONL. Re-invocation with the same ``--output`` path skips
already-completed instances automatically.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from sragents.config import model_short_name
from sragents.infer.base import InferenceEngine, SkillProvider
from sragents.infer.schema import InferenceRecord

_write_lock = threading.Lock()


def _already_done(out_path: Path) -> set[str]:
    """Return completed instance_ids. Truncates any trailing partial
    line from a previous interrupted write so the resume append starts
    on a clean record boundary."""
    if not out_path.exists():
        return set()
    done: set[str] = set()
    good_bytes = 0
    with open(out_path, "rb") as f:
        data = f.read()
    for raw in data.splitlines(keepends=True):
        stripped = raw.decode("utf-8", errors="replace").strip()
        if not stripped:
            good_bytes += len(raw)
            continue
        if not raw.endswith(b"\n"):
            # Trailing partial line — don't count, don't advance.
            break
        try:
            rec = json.loads(stripped)
            done.add(rec["instance_id"])
            good_bytes += len(raw)
        except (json.JSONDecodeError, KeyError):
            # Stop at the first malformed line so the truncate below
            # rewinds to the end of the last good record.
            break
    if good_bytes < len(data):
        with open(out_path, "rb+") as f:
            f.truncate(good_bytes)
    return done


def _append(fout, record: InferenceRecord) -> None:
    line = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    with _write_lock:
        fout.write(line)
        fout.flush()


def run_many(
    instances: list[dict],
    provider: SkillProvider,
    engine: InferenceEngine,
    client,
    model: str,
    output_path: Path,
    label: str,
    workers: int = 32,
    engine_kwargs: dict | None = None,
) -> None:
    """Run provider×engine on ``instances``, streaming results to ``output_path``.

    Skips instances already present in ``output_path`` (per-instance resume).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done = _already_done(output_path)
    pending = [i for i in instances if i["instance_id"] not in done]

    if not pending:
        print(f"  all {len(instances)} instances already complete → {output_path}")
        return
    if done:
        print(f"  resuming: {len(done)}/{len(instances)} done, {len(pending)} remaining")

    model_name = model_short_name(model)
    engine_kwargs = engine_kwargs or {}

    def _one(inst: dict) -> InferenceRecord:
        try:
            skills = provider.provide(inst)
            result = engine.run(inst, skills, client, model, **engine_kwargs)
            return InferenceRecord(
                instance_id=inst["instance_id"],
                dataset=inst["dataset"],
                method=label,
                model=model_name,
                raw_output=result.raw_output,
                transcript=result.transcript,
                skill_ids_used=result.skill_ids_used,
                meta=result.meta,
            )
        except Exception as e:  # noqa: BLE001
            print(f"\n  ERROR on {inst['instance_id']}: {e}", file=sys.stderr)
            return InferenceRecord(
                instance_id=inst["instance_id"],
                dataset=inst["dataset"],
                method=label,
                model=model_name,
                raw_output="",
                error=str(e),
            )

    n_workers = max(workers, 1)
    with open(output_path, "a") as fout:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_one, i): i for i in pending}
            with tqdm(total=len(pending), desc=f"  {output_path.name}") as bar:
                for fut in as_completed(futures):
                    _append(fout, fut.result())
                    bar.update(1)

    print(f"  wrote {len(pending)} records → {output_path}")
