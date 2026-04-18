# SR-Agents

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
  25,626 web-collected distractors);
* **SR-Agents** — a baseline family of skill-retrieval-augmented
  agents, decomposed into pluggable Retriever / Provider / Engine
  components so new methods can be added by writing one small file;
* A three-stage evaluation pipeline (retrieve → infer → evaluate) that
  measures the full SRA process and lets you diagnose where a method
  succeeds or fails.

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

```bash
pip install -e .       # or: uv sync
```

Unzip the skill library:

```bash
unzip data/bench/corpus/corpus.json.zip -d data/bench/corpus/
```

<details><summary>ToolQA external corpus (only needed to run ToolQA)</summary>

Download from the
[ToolQA Google Drive link](https://drive.google.com/file/d/1zRbHzPW2x4dDcfmphBWlan8cxUCRNmqk/view?usp=drive_link)
(~2.6 GB), unzip, and place the result under `data/external/toolqa/`.
</details>

Inference requires an OpenAI-compatible chat endpoint. Point
`--api-base` at any compatible server (OpenAI, vLLM, SGLang, Ollama, …).

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

`infer` jointly covers paper stages 2 and 3: the `Provider` selects
and transforms the skills to actually incorporate, and the `Engine`
applies them while solving the task. `evaluate` scores the end-task
output — it is not one of the paper's SRA stages.

Each stage's output is consumed by the next via an explicit path
argument, so any stage can be swapped or rerun in isolation. Every
stage supports per-instance resume.

Inference is decomposed along two orthogonal axes:

* **SkillProvider** — *which* skills an instance receives (none / oracle
  / top-K retrieval / LLM-selected / oracle + hard-negative
  distractors).
* **InferenceEngine** — *how* those skills are exposed to the model
  (skill prepending / progressive-disclosure agent loop / ReAct loop
  for ToolQA).

Five built-in skill-use methods, each a specific (Provider, Engine)
combination:

| Method | Provider | Engine | Description |
|---|---|---|---|
| **LLM Direct** | `none` | `direct` | No external skill — parametric-only baseline |
| **Oracle Skill** | `oracle` | `direct` | Annotated gold skill prepended (upper bound) |
| **BM25 Top-1** | `topk(k=1)` | `direct` | Skill Prepending with the BM25 rank-1 skill |
| **BM25 Select** | `llm_select(pool=50)` | `direct` | Model picks one skill from BM25 top-50, then answers |
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
sragents list experiments    # main, retrieval_comparison, distractor, ...
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
    --model /path/to/model --api-base http://localhost:8000/v1 \
    --top-k 50
```

`sragents hybrid` is the round-robin fuser used for the paper's Hybrid
(BM25 + BGE) retriever — see the reproduction recipe below.

### 2. Infer

```bash
sragents infer \
    --instances data/bench/instances/theoremqa.json \
    --output results/inference/theoremqa-Qwen3-32B-bm25_top1.jsonl \
    --model /path/to/model --api-base http://localhost:8000/v1 \
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
| BM25 Top-1 | `--provider topk --provider-arg source=… --provider-arg k=1 --engine direct` |
| BM25 Select | `--provider llm_select --provider-arg source=… --provider-arg pool=50 --engine direct` |
| Progressive Disclosure | `--provider topk --provider-arg source=… --provider-arg k=50 --engine progressive_disclosure` |
| ToolQA | Replace `--engine direct` with `--engine react` (or `--engine progressive_disclosure` with `--engine react_progressive_disclosure`) |

### 3. Evaluate

```bash
sragents evaluate \
    --input results/inference/theoremqa-Qwen3-32B-bm25_top1.jsonl \
    --instances data/bench/instances/theoremqa.json \
    --output results/eval/theoremqa-Qwen3-32B-bm25_top1.json
```

## Reproducing the paper experiments

The paper evaluates six retrievers: BM25, TF-IDF, BGE, Contriever, a
round-robin hybrid of BM25 and BGE, and an LLM rerank of BM25's top-50.
Pre-compute the first four directly, then fuse BM25 and BGE into the
hybrid file:

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

# Round-robin fusion of BM25 + BGE
for ds in $DATASETS; do
    sragents hybrid \
        --input results/retrieval/$ds-bm25.json \
                results/retrieval/$ds-bge.json \
        --output results/retrieval/$ds-hybrid_bm25_bge.json
done
```

The LLM rerank file (`*-rerank_bm25-<model>.json`) is model-dependent
and produced on demand by `sragents experiment --exp
retrieval_comparison` (it invokes `sragents rerank` once per dataset
for the evaluated model).

Then run each experiment with the named catalog:

```bash
# Main table: 5 skill-use methods × 6 datasets.
sragents experiment --exp main \
    --model /path/to/model --api-base http://localhost:8000/v1

# Retriever comparison (rank-1 end-to-end under BM25 / TF-IDF / BGE /
# Contriever / Hybrid / BM25 + Rerank).
sragents experiment --exp retrieval_comparison ...

# Noise robustness (oracle + N hard-negative distractors) in both
# context-injection and progressive-disclosure modes.
sragents experiment --exp distractor ...
```

Narrow the scope with `--dataset theoremqa logicbench` or
`--methods bm25_top1 progressive_disclosure`. The runner invokes
`sragents infer` and `sragents evaluate` for each (dataset, method)
cell and skips cells whose output files already exist.

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
