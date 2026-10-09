# refusal-lens

[![CI](https://github.com/fahd04/refusal-prob/actions/workflows/ci.yml/badge.svg)](https://github.com/fahd04/refusal-prob/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

A local toolkit for analyzing and editing refusal behavior in open-weight LLMs
(Qwen2.5, Llama 3.2).

> **Status:** pre-alpha. The project structure, configuration, device handling
> and CLI are in place; the analysis and editing stages are stubs.

## Overview

Instruction-tuned models decide whether to refuse a request, and research has
shown that this behavior is largely mediated by a single direction in the
residual stream. refusal-lens makes that direction something you can measure,
look at and edit:

1. **Record** residual-stream activations on harmful and harmless prompts.
2. **Compute** a refusal direction per layer as the difference of means.
3. **Visualize** how the direction emerges and changes across layers.
4. **Ablate** it with inference-time hooks, or permanently by orthogonalizing
   the weights that write to the residual stream.
5. **Evaluate** refusal rate, KL divergence on harmless prompts, and
   capability retention.
6. **Export** the edited model as safetensors or GGUF.

Everything runs locally. It is developed on an Apple M4 Pro (MPS, 24 GB) and
also supports CUDA and CPU.

| Module | Purpose |
| --- | --- |
| `config` | Pydantic run config loaded from YAML; seeding; resolved-config snapshots |
| `device` | Device (`mps` → `cuda` → `cpu`) and dtype selection |
| `models` | Model/tokenizer loading, per-architecture adapters |
| `prompts` | Harmful/harmless prompt sets, chat templates |
| `activations` | Residual-stream recording via forward hooks |
| `directions` | Difference-of-means refusal directions |
| `ablation` | Hook-based ablation and weight orthogonalization |
| `evaluate` | Refusal rate, KL divergence, capability benchmark |
| `visualize` | Plots across layers and prompts |
| `export` | safetensors and GGUF (via llama.cpp) |
| `cli` | `refusal-lens` Typer CLI |

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fahd04/refusal-prob.git
cd refusal-prob
uv sync                     # creates .venv with runtime + dev dependencies
uv run refusal-lens --help
```

**Apple Silicon.** Some PyTorch operators are not implemented on MPS yet. Set
this before running so they fall back to the CPU instead of raising:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

refusal-lens warns at startup if MPS is selected without it.

**CUDA.** On Linux, `uv sync` installs CPU-only PyTorch so CI and CPU machines
don't download multi-GB CUDA wheels. To use a GPU, replace it with a CUDA
build that matches your driver, e.g.:

```bash
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

**GGUF export** needs a local [llama.cpp](https://github.com/ggml-org/llama.cpp)
checkout; point `export.llama_cpp_dir` in your config at it.

### Development

```bash
uv run pre-commit install   # ruff + mypy on every commit
uv run ruff check . && uv run ruff format --check .
uv run mypy                 # strict on src/
uv run pytest               # CPU only, tiny random models, no downloads
```

## Quickstart

*Placeholder — available once the pipeline stages are implemented.*

```bash
uv run refusal-lens analyze  --config configs/qwen2.5-1.5b.yaml
uv run refusal-lens ablate   --config configs/qwen2.5-1.5b.yaml
uv run refusal-lens evaluate --config configs/qwen2.5-1.5b.yaml
uv run refusal-lens export   --config configs/qwen2.5-1.5b.yaml
```

Each run writes to `outputs/<name>/`, starting with `resolved_config.yaml`: the
full config with defaults filled in, plus the refusal-lens and torch versions.
Passing that file back with `--config` reproduces the run. Seeds come from the
config's `seed` field; set `deterministic: true` to also request deterministic
kernels.

## How it works

*Placeholder.* This section will cover how activations are recorded, how the
difference-of-means direction is computed and selected, the math behind
directional ablation and weight orthogonalization, and how architecture
adapters map Qwen2 and Llama modules onto that interface.

## Evaluation

*Placeholder.* This section will report refusal rates on held-out harmful and
harmless prompts, the KL divergence between base and edited models on
harmless prompts, and capability-benchmark deltas, along with exact configs
for reproducing each number.

## Intended use

refusal-lens is meant for interpretability and safety research: understanding
how refusal is represented inside a model, how robust safety fine-tuning is,
and how weight edits change model behavior. It is also an educational
reference implementation of published techniques.

## Limitations

- **Edited models are less safe.** Removing the refusal direction removes a
  safety mechanism. An edited model will comply with harmful requests it
  previously refused. Do not deploy edited models in user-facing products, and
  do not use them to cause harm. You are responsible for what you do with them.
- **Refusal is not only one direction.** The single-direction picture is a
  strong approximation. Some refusal behavior can remain, and edits can
  degrade unrelated capabilities.
- **Evaluation is approximate.** Detecting refusals by string matching misses
  some refusals and flags some non-refusals. KL divergence and small
  benchmarks only partly capture capability loss.
- **Scope.** Only Qwen2/Qwen2.5 and Llama 3.x architectures have adapters, and
  development targets small models that fit on a single consumer machine.
- **Numerics.** Results can differ slightly between MPS, CUDA and CPU, and
  between dtypes.

## Related work

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., &
  Nanda, N. (2024). *Refusal in Language Models Is Mediated by a Single
  Direction.* [arXiv:2406.11717](https://arxiv.org/abs/2406.11717).
  The method refusal-lens implements.
- FailSpy, [abliterator](https://github.com/FailSpy/abliterator): an early
  library for "abliteration" built on TransformerLens.
- Sumandora, [remove-refusals-with-transformers](https://github.com/Sumandora/remove-refusals-with-transformers):
  a minimal implementation using plain Hugging Face transformers.
- [Heretic](https://github.com/p-e-w/heretic): automated refusal removal that
  tunes ablation parameters with optimization.

## License

Code is released under the [Apache License 2.0](LICENSE).

**Model licenses still apply.** refusal-lens does not ship model weights. Any
model you edit or export with it remains subject to the original model's
license and acceptable-use policy (for example the Qwen license or the Llama 3.2
Community License). Check those terms before you share or use an edited model.
