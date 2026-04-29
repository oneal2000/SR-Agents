# SR-Agents

[![Dataset on 🤗](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)](https://huggingface.co/datasets/WeihangSu/SRA-Bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)

A benchmark and research toolkit for **skill-retrieval-augmented LLM
agents**.

Modern LLM agents increasingly rely on reusable external *skills* —
modular capability packages that pair natural-language guidance with
executable resources — to solve tasks that exceed their native
parametric ability. As skill libraries grow into the tens or hundreds
of thousands, the prevailing practice of enumerating every candidate
skill in the prompt stops scaling: context budgets fill up, and
selection accuracy degrades as the skill list grows.

**Skill-Retrieval Augmentation (SRA)** is an alternative paradigm in
which the agent dynamically retrieves, incorporates, and applies
relevant skills from a large external skill library on demand. This
repository provides:

* **SRA-Bench** — 5,400 capability-intensive test instances across six
  task families, each paired with manually curated gold skill(s),
  embedded in a realistic skill library of 26,262 skills (636 gold +
  25,626 web-collected distractors).
* **SR-Agents** — a baseline family of skill-retrieval-augmented
  agents, spanning five skill-use methods (LLM Direct, Oracle Skill,
  Full-Skill Injection, LLM Selection, Progressive Disclosure) and six
  retrievers (BM25, TF-IDF, BGE, Contriever, Hybrid, BM25 + LLM Rerank).

![SRA paradigm overview](assets/overall.png)

*The Skill-Retrieval Augmentation paradigm: the agent retrieves candidate
skills from a large external skill library, selectively incorporates
useful ones into context, and applies them for downstream reasoning
and acting.*

## SRA-Bench at a glance

**6 source datasets · 5,400 test instances · 636 gold skills** embedded
in a skill library of **26,262 skills** (2.4% gold, 25,626 web-collected
distractors).

Gold-skill annotations associate each instance with either a single
gold skill (**Single**) or multiple gold skills (**Multi**).

| Dataset | Capability Type | #Inst. | #Skills | Skill Mapping | Evaluation |
|---|---|---:|---:|---|---|
| TheoremQA | Theorem Application | 747 | 320 | Single | Rule-Based |
| LogicBench | Logical Reasoning Patterns | 760 | 19 | Single | Rule-Based |
| ToolQA | Tool-Use Workflows | 1,430 | 14 | Single | Rule-Based |
| MedCalc-Bench | Medical Calculators | 1,100 | 55 | Single | Rule-Based |
| CHAMP | Mathematical Concepts | 223 | 89 | Multi | Rule-Based |
| BigCodeBench | Software Libraries | 1,140 | 139 | Multi | Execution |

## Install

Requires Python 3.10 – 3.12.

```bash
pip install -e .       # or: uv sync
```

Download SRA-Bench (skill corpus + test instances) from the
[HuggingFace dataset page](https://huggingface.co/datasets/WeihangSu/SRA-Bench):

```bash
huggingface-cli download WeihangSu/SRA-Bench --repo-type dataset \
    --local-dir data/bench
```

<details><summary>ToolQA external corpus (only needed to run ToolQA)</summary>

Download from the
[ToolQA Google Drive link](https://drive.google.com/file/d/1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk/view?usp=drive_link),
unzip, and place the result under `data/external/toolqa/`.
</details>

Inference requires an OpenAI-compatible chat endpoint. Point
`--api-base` at any compatible server (OpenAI, vLLM, SGLang, Ollama, …).
`--model` is the served model identifier — a model ID like
`gpt-4o-mini` for hosted APIs, or whatever string the local server is
serving (often a path like `/models/Qwen3-32B`) for vLLM/SGLang. For
endpoints that require auth, set `OPENAI_API_KEY`; local unauthenticated
servers accept any value.

## Quickstart

An end-to-end run of the three stages on TheoremQA using Full-Skill
Injection with the BM25 top-1 skill:

```bash
# Pick one (examples; any OpenAI-compatible endpoint works).
# Hosted:  MODEL=gpt-4o-mini           API_BASE=https://api.openai.com/v1
# Local :  MODEL=/models/Qwen3-32B     API_BASE=http://localhost:8000/v1
MODEL=<MODEL>
API_BASE=<API_BASE>

# 1. Retrieve — BM25 top-50 per query.
sragents retrieve --retriever bm25 \
    --corpus data/bench/corpus/corpus.json \
    --instances data/bench/instances/theoremqa.json \
    --output results/retrieval/theoremqa-bm25.json

# 2. Infer — prepend the top-1 BM25 skill and generate an answer.
sragents infer \
    --instances data/bench/instances/theoremqa.json \
    --output results/inference/theoremqa-bm25_top1.jsonl \
    --model $MODEL --api-base $API_BASE \
    --provider topk \
      --provider-arg source=results/retrieval/theoremqa-bm25.json \
      --provider-arg k=1 \
    --engine direct --label bm25_top1

# 3. Evaluate — extract + score against ground truth.
sragents evaluate \
    --input results/inference/theoremqa-bm25_top1.jsonl \
    --instances data/bench/instances/theoremqa.json \
    --output results/eval/theoremqa-bm25_top1.json
```

The evaluator prints overall accuracy and saves a JSON of the form

```json
{
  "dataset": "theoremqa", "method": "bm25_top1", "model": "...",
  "metrics": {"accuracy": 0.XX, "correct": N, "total": 747},
  "details": [{"instance_id": "...", "extracted_answer": "...",
               "correct": true, "ground_truth": "...", ...}]
}
```

## Pipeline

The paper formulates SRA as three tightly coupled stages — **skill
retrieval**, **skill incorporation**, and **skill application** (paper
§2.2). The codebase operationalizes them as three executable stages
that communicate via JSON files:

```
Retrieve  ─── Retriever          ───▶ retrieval/*.json   (skill retrieval)
Infer     ─── Provider × Engine  ───▶ inference/*.jsonl  (skill incorporation + application)
Evaluate  ─── Evaluator          ───▶ eval/*.json        (end-task scoring)
```

`infer` jointly covers paper stages 2 and 3. The `Provider` supplies
the instance's *candidate* skills — possibly one pre-selected skill,
possibly a larger pool for the engine to narrow further. The `Engine`
consumes those candidates and produces the answer: simple engines
(`direct`) statically prepend everything they receive; agentic engines
(`progressive_disclosure`, `react`, `react_progressive_disclosure`)
interleave further skill selection with solving inside their reasoning
loop. `evaluate` scores the
end-task output — it is not one of the paper's SRA stages.

Each stage's output is consumed by the next via an explicit path
argument, so any stage can be swapped or rerun in isolation. Every
stage supports per-instance resume.

Inference is decomposed along two orthogonal axes:

* **SkillProvider** — the *candidate skills* an instance receives
  (none / oracle / top-K retrieval / LLM-selected / oracle +
  hard-negative distractors). May already be narrowed to one skill
  or left as a pool.
* **InferenceEngine** — how those candidates are turned into an
  answer (static prepending / progressive-disclosure agent loop /
  ReAct loop for ToolQA). Agentic engines perform further
  in-loop skill selection.

Five built-in skill-use methods, each a specific (Provider, Engine)
combination:

| Method | Provider | Engine | Description |
|---|---|---|---|
| **LLM Direct** | `none` | `direct` | No external skill — parametric-only baseline |
| **Oracle Skill** | `oracle` | `direct` | Annotated gold skill prepended (upper bound) |
| **Full-Skill Injection** | `topk(k=1)` | `direct` | Full content of the BM25 rank-1 skill prepended to the prompt |
| **LLM Selection** | `llm_select(pool=50)` | `direct` | Model picks one skill from BM25 top-50, then answers |
| **Progressive Disclosure** | `topk(k=50)` | `progressive_disclosure` | Model sees a compact skill catalog and loads skills on demand |

For ToolQA the `direct` engine is replaced by `react` and
`progressive_disclosure` by `react_progressive_disclosure`; the
experiment runner selects the right engine per dataset.

## CLI

A single `sragents` command is installed (also invokable as
`python -m sragents.cli.main`). Every subcommand has `--help`.

```bash
sragents list retrievers     # bm25 tfidf bge contriever
sragents list providers      # none oracle topk llm_select oracle_distractor
sragents list engines        # direct progressive_disclosure react react_progressive_disclosure
sragents list datasets       # theoremqa logicbench toolqa champ medcalcbench bigcodebench
sragents list experiments    # main, retrieval_comparison, topk_sweep, distractor, ...
```

### 1. Retrieve

```bash
sragents retrieve \
    --retriever bm25 \
    --corpus data/bench/corpus/corpus.json \
    --instances data/bench/instances/theoremqa.json \
    --output results/retrieval/theoremqa-bm25.json \
    --top-k 50
```

Dense retrievers default to sensible checkpoints (`BAAI/bge-base-en-v1.5`
for `bge`, `facebook/contriever-msmarco` for `contriever`). Override
with `--retriever-arg model_path=<hf-name>` to swap in a different model.

Retrievers can also be cascaded. `sragents rerank` is a second-stage
retriever: it reads an existing retrieval file, uses an LLM to reorder
the top-K candidates per query, and writes a new retrieval file in the
same format. A typical cascade is BM25 → LLM rerank:

```bash
sragents rerank \
    --input results/retrieval/theoremqa-bm25.json \
    --output results/retrieval/theoremqa-rerank_bm25.json \
    --instances data/bench/instances/theoremqa.json \
    --model <MODEL> --api-base <API_BASE> \
    --top-k 50
```

`sragents hybrid` is the round-robin fuser used for the paper's Hybrid
(BM25 + BGE) retriever — see the reproduction recipe below.

### 2. Infer

```bash
sragents infer \
    --instances data/bench/instances/theoremqa.json \
    --output results/inference/theoremqa-Qwen3-32B-bm25_top1.jsonl \
    --model <MODEL> --api-base <API_BASE> \
    --provider topk \
      --provider-arg source=results/retrieval/theoremqa-bm25.json \
      --provider-arg k=1 \
    --engine direct \
    --label bm25_top1
```

Common recipes:

| Method | CLI fragment |
|---|---|
| LLM Direct | `--provider none --engine direct` |
| Oracle Skill | `--provider oracle --engine direct` |
| Full-Skill Injection | `--provider topk --provider-arg source=… --provider-arg k=1 --engine direct` |
| LLM Selection | `--provider llm_select --provider-arg source=… --provider-arg pool=50 --engine direct` |
| Progressive Disclosure | `--provider topk --provider-arg source=… --provider-arg k=50 --engine progressive_disclosure` |
| ToolQA | For any of the above, use `--engine react` in place of `direct`, or `--engine react_progressive_disclosure` in place of `progressive_disclosure` |

### 3. Evaluate

```bash
sragents evaluate \
    --input results/inference/theoremqa-Qwen3-32B-bm25_top1.jsonl \
    --instances data/bench/instances/theoremqa.json \
    --output results/eval/theoremqa-Qwen3-32B-bm25_top1.json
```

## Reproducing the paper experiments

The paper reports six retrievers at the retrieval-metric level
(Recall@K, nDCG@K): BM25, TF-IDF, BGE, Contriever, a round-robin
hybrid of BM25 and BGE, and an LLM rerank of BM25's top-50. The
end-to-end retriever-comparison experiment (`retrieval_comparison`
below) runs five of those — the rank-1 Hybrid is not included because
round-robin fusion always returns BM25's top-1 at rank 1, so its
end-to-end numbers are identical to BM25.

Pre-compute the first four retrievers directly, then fuse BM25 + BGE
into the hybrid file:

```bash
DATASETS="theoremqa logicbench toolqa champ medcalcbench bigcodebench"

# First-stage retrievers
for ds in $DATASETS; do
    for r in bm25 tfidf bge contriever; do
        sragents retrieve --retriever $r \
            --corpus data/bench/corpus/corpus.json \
            --instances data/bench/instances/$ds.json \
            --output results/retrieval/$ds-$r.json
    done
done

# Round-robin fusion of BM25 + BGE (for retrieval-metric evaluation).
for ds in $DATASETS; do
    sragents hybrid \
        --input results/retrieval/$ds-bm25.json \
                results/retrieval/$ds-bge.json \
        --output results/retrieval/$ds-hybrid_bm25_bge.json
done
```

The LLM rerank output is model-dependent (file name:
`*-rerank_bm25-<model>.json`), so it has to be produced once per
evaluated model:

```bash
MODEL=<MODEL>          # same identifier you pass to `sragents infer --model`
API_BASE=<API_BASE>
for ds in $DATASETS; do
    sragents rerank \
        --input  results/retrieval/$ds-bm25.json \
        --output results/retrieval/$ds-rerank_bm25-$(basename $MODEL).json \
        --instances data/bench/instances/$ds.json \
        --model $MODEL --api-base $API_BASE --top-k 50
done
```

`sragents experiment --exp retrieval_comparison` will also trigger
`sragents rerank` on demand if the file is missing.

Then run each experiment with the named catalog:

```bash
# Main table: 5 skill-use methods × 6 datasets.
sragents experiment --exp main \
    --model <MODEL> --api-base <API_BASE>

# Retriever comparison (rank-1 end-to-end under BM25 / TF-IDF / BGE /
# Contriever / BM25 + Rerank).
sragents experiment --exp retrieval_comparison \
    --model <MODEL> --api-base <API_BASE>

# Retrieval-depth sweep: BM25 top-K skills under both Full-Skill
# Injection and Progressive Disclosure exposure modes (K ∈ {1, 2, 4, 8};
# K=1 Full-Skill Injection overlaps with the main experiment's bm25_top1).
# Use `--exp topk_sweep_injection` or
# `--exp topk_sweep_progressive_disclosure` to run only one mode.
sragents experiment --exp topk_sweep \
    --model <MODEL> --api-base <API_BASE>

# Noise robustness (oracle + N hard-negative distractors) under both
# Full Skill Injection and Progressive Disclosure exposure modes.
sragents experiment --exp distractor \
    --model <MODEL> --api-base <API_BASE>
```

Narrow the scope with `--dataset theoremqa logicbench` or
`--methods bm25_top1 progressive_disclosure` (use the method labels —
the technical identifiers — not the paper-style display names). The
runner invokes `sragents infer` and `sragents evaluate` for each
(dataset, method) cell and skips cells whose output files already
exist.

## Project layout

```
src/sragents/
├── config.py               # path constants, ALL_DATASETS
├── corpus.py               # skill corpus loading
├── llm.py                  # OpenAI-compatible client
├── prompts.py              # per-dataset prompt builders
│
├── retrieve/               # Stage 1
│   ├── base.py             #   Retriever protocol + registry
│   ├── bm25.py · tfidf.py · dense.py · hybrid.py · llm_rerank.py
│   └── metrics.py · schema.py
│
├── infer/                  # Stage 2
│   ├── base.py             #   SkillProvider + InferenceEngine protocols
│   ├── runner.py           #   per-instance parallel execution + resume
│   ├── providers/          #   none · oracle · topk · llm_select · oracle_distractor
│   └── engines/            #   direct · progressive_disclosure · react · react_progressive_disclosure
│
├── evaluate/               # Stage 3
│   ├── base.py             #   Evaluator protocol + registry
│   ├── datasets/           #   6 per-dataset evaluators
│   └── metrics.py
│
├── toolqa/                 # ToolQA tools (used by ReAct engines)
├── experiments/            # paper experiment catalog + runner
└── cli/                    # sragents <subcommand> entry points

data/bench/
├── corpus/corpus.json      # 26,262 skills (636 gold + 25,626 web)
└── instances/*.json        # 6 datasets
```

## License

See [LICENSE](LICENSE).
