"""Inference result record format.

One JSON object per line in the output ``.jsonl`` file::

    {
      "instance_id": "...",
      "dataset": "theoremqa",
      "method": "bm25_top1",        # label (for aggregation & reports)
      "model": "Qwen3-32B",
      "raw_output": "...",           # model-generated tokens only
      "transcript": "...",           # optional, agent modes
      "skill_ids_used": ["..."],     # optional
      "meta": {"n_steps": 3}         # optional, engine-specific
    }

For failed instances::

    {..., "raw_output": "", "error": "exception message"}
"""

from dataclasses import asdict, dataclass, field


@dataclass
class InferenceRecord:
    instance_id: str
    dataset: str
    method: str
    model: str
    raw_output: str
    transcript: str | None = None
    skill_ids_used: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop empty optionals to keep files small & diffable.
        if d["transcript"] is None:
            d.pop("transcript")
        if not d["skill_ids_used"]:
            d.pop("skill_ids_used")
        if not d["meta"]:
            d.pop("meta")
        if d["error"] is None:
            d.pop("error")
        return d
