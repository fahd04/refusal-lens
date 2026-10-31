"""``refusal-lens`` command-line interface.

Every command takes ``--config``. Before doing any work it seeds all RNGs from
the config and writes the resolved config into the run directory, so each
output directory is self-describing and reproducible.

The stages build on each other's outputs in the run directory::

    analyze  -> directions.safetensors, analysis.json, plots/
    ablate   -> ablation.json (+ edited_model/ for method "orthogonalize")
    evaluate -> evaluation.json
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import torch
import typer
from rich.console import Console
from rich.table import Table
from transformers import PreTrainedTokenizerBase

from refusal_lens import __version__
from refusal_lens.ablation import DirectionAblation, orthogonalize_weights
from refusal_lens.activations import ActivationCache, record_activations
from refusal_lens.config import RunConfig, load_config, save_resolved_config, seed_everything
from refusal_lens.device import resolve_device, resolve_dtype
from refusal_lens.directions import (
    RefusalDirections,
    compute_refusal_directions,
    load_directions,
    save_directions,
    select_direction,
    separation_scores,
)
from refusal_lens.evaluate import (
    EvaluationReport,
    generate_completions,
    kl_divergence,
    next_token_logprobs,
    refusal_rate,
    select_ablation_layer,
)
from refusal_lens.models import ArchitectureAdapter, get_adapter, load_model_and_tokenizer
from refusal_lens.prompts import format_chat, load_prompt_sets
from refusal_lens.visualize import (
    plot_direction_norms,
    plot_direction_similarity,
    plot_layer_scores,
    plot_projections,
    save_figure,
)

DIRECTIONS_FILENAME = "directions.safetensors"
ABLATION_FILENAME = "ablation.json"
EDITED_MODEL_DIRNAME = "edited_model"

# Refusals ("I'm sorry", "As an AI") show up in the first few tokens, so scoring
# candidate layers needs only short answers. Keeps the search over all layers fast.
_SELECTION_MAX_NEW_TOKENS = 16

app = typer.Typer(
    name="refusal-lens",
    help="Analyze and edit refusal behavior in open-weight LLMs.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(stderr=True)

ConfigOption = Annotated[
    Path,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a run config YAML (see configs/).",
    ),
]


class MissingStageOutputError(Exception):
    """A stage needs the output of an earlier stage that hasn't been run yet."""


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"refusal-lens {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """Analyze and edit refusal behavior in open-weight LLMs."""


def _prepare_run(config_path: Path) -> RunConfig:
    """Load config, seed RNGs, create the run dir and save the resolved config."""
    config = load_config(config_path)
    seed_everything(config.seed, deterministic=config.deterministic)
    resolved = save_resolved_config(config)
    console.print(f"[bold]run[/] {config.name}  [dim]seed={config.seed}  config -> {resolved}[/]")
    return config


def _run_stage(name: str, stage: Callable[[RunConfig], None], config: RunConfig) -> None:
    """Run a pipeline stage, reporting unimplemented stubs and missing inputs cleanly."""
    try:
        stage(config)
    except NotImplementedError:
        console.print(f"[yellow]{name}: not implemented yet[/]")
        raise typer.Exit(code=1) from None
    except MissingStageOutputError as error:
        console.print(f"[red]{name}: {error}[/]")
        raise typer.Exit(code=1) from None


def _load_model(
    config: RunConfig, model_dir: Path | None = None
) -> tuple[ArchitectureAdapter, PreTrainedTokenizerBase]:
    """Load the configured model, or a saved one from ``model_dir`` with the same settings."""
    model_config = config.model
    if model_dir is not None:
        model_config = model_config.model_copy(update={"id": str(model_dir), "revision": None})
    device = resolve_device(model_config.device)
    dtype = resolve_dtype(model_config.dtype, device)
    console.print(f"loading [bold]{model_config.id}[/] on {device} ({dtype})")
    model, tokenizer = load_model_and_tokenizer(model_config, device, dtype)
    return get_adapter(model), tokenizer


def _require_directions(config: RunConfig) -> RefusalDirections:
    path = config.run_dir / DIRECTIONS_FILENAME
    if not path.is_file():
        raise MissingStageOutputError(f"{path} not found; run `refusal-lens analyze` first")
    return load_directions(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _analyze(config: RunConfig) -> None:
    """Record activations, compute per-layer directions, and plot them."""
    adapter, tokenizer = _load_model(config)
    prompts = load_prompt_sets(config.prompts.harmful, config.prompts.harmless, config.seed)
    layers = config.analysis.layers
    if layers == "all":
        layers = list(range(adapter.num_layers))
    positions = config.analysis.positions

    def record(instructions: list[str], label: str) -> ActivationCache:
        console.print(f"recording {len(instructions)} {label} prompts")
        chats = format_chat(tokenizer, instructions, config.prompts.system_prompt)
        return record_activations(
            adapter, tokenizer, chats, layers, positions, config.prompts.batch_size
        )

    # Directions come from the train prompts; scores from the held-out eval prompts.
    directions = compute_refusal_directions(
        record(prompts.harmful.train, "harmful train"),
        record(prompts.harmless.train, "harmless train"),
    )
    harmful_eval = record(prompts.harmful.eval, "harmful eval")
    harmless_eval = record(prompts.harmless.eval, "harmless eval")
    scores = separation_scores(directions, harmful_eval, harmless_eval)
    best_layer, best_position = select_direction(directions, scores)
    p = positions.index(best_position)

    run_dir = config.run_dir
    save_directions(directions, run_dir / DIRECTIONS_FILENAME)
    plots = run_dir / "plots"
    best_unit = directions.unit(best_layer, best_position)
    save_figure(plot_direction_norms(directions, best_position), plots / "direction_norms.png")
    save_figure(
        plot_direction_similarity(directions, best_position), plots / "direction_similarity.png"
    )
    save_figure(
        plot_projections(harmful_eval, harmless_eval, best_unit, best_position),
        plots / "projections.png",
    )
    save_figure(
        plot_layer_scores(scores[:, p], layers, "separation (Cohen's d)"),
        plots / "separation.png",
    )

    _write_json(
        run_dir / "analysis.json",
        {
            "model": config.model.id,
            "best_layer": best_layer,
            "best_position": best_position,
            "n_prompts": {
                "harmful_train": len(prompts.harmful.train),
                "harmful_eval": len(prompts.harmful.eval),
                "harmless_train": len(prompts.harmless.train),
                "harmless_eval": len(prompts.harmless.eval),
            },
            # JSON has no NaN, so undefined scores (zero directions) become null.
            "separation": {
                str(layer): None if math.isnan(score) else round(score, 4)
                for layer, score in zip(layers, scores[:, p].tolist(), strict=True)
            },
        },
    )
    console.print(
        f"[green]done[/]: clearest separation at layer {best_layer} "
        f"(d={scores[layers.index(best_layer), p]:.2f}); outputs in {run_dir}"
    )


def _ablate(config: RunConfig) -> None:
    """Remove the selected refusal direction and save the edited model."""
    directions = _require_directions(config)
    adapter, tokenizer = _load_model(config)
    model = adapter.model
    position = config.ablation.direction_position
    batch_size = config.prompts.batch_size
    candidates: dict[str, dict[str, float]] = {}

    layer = config.ablation.direction_layer
    if layer is None:
        # Try every layer's direction with hooks on a few *train* prompts; the eval
        # prompts stay unseen for `evaluate`.
        prompts = load_prompt_sets(config.prompts.harmful, config.prompts.harmless, config.seed)
        n = config.ablation.selection_prompts
        system_prompt = config.prompts.system_prompt
        harmful = format_chat(tokenizer, prompts.harmful.train[:n], system_prompt)
        harmless = format_chat(tokenizer, prompts.harmless.train[:n], system_prompt)
        base_logprobs = next_token_logprobs(model, tokenizer, harmless, batch_size)

        p = directions.positions.index(position)
        for i, candidate in enumerate(directions.layers):
            if not directions.directions[i, p].any():
                continue  # zero direction (e.g. layer 0): nothing to remove
            with DirectionAblation(adapter, directions.unit(candidate, position)):
                completions = generate_completions(
                    model, tokenizer, harmful, _SELECTION_MAX_NEW_TOKENS, batch_size
                )
                edited_logprobs = next_token_logprobs(model, tokenizer, harmless, batch_size)
            rate = refusal_rate(completions, config.evaluation.refusal_markers)
            kl = kl_divergence(base_logprobs, edited_logprobs)
            candidates[str(candidate)] = {"refusal_rate": rate, "kl": kl}
            console.print(f"layer {candidate:>3}: refusal {rate:4.0%}  KL {kl:.3f}")

        layers = [int(c) for c in candidates]
        rates = [c["refusal_rate"] for c in candidates.values()]
        kls = [c["kl"] for c in candidates.values()]
        layer = select_ablation_layer(layers, rates, kls)
        plots = config.run_dir / "plots"
        save_figure(
            plot_layer_scores(torch.tensor(rates), layers, "refusal rate with direction removed"),
            plots / "ablation_refusal.png",
        )
        save_figure(
            plot_layer_scores(torch.tensor(kls), layers, "KL on harmless prompts"),
            plots / "ablation_kl.png",
        )

    method = config.ablation.method
    if method == "orthogonalize":
        orthogonalize_weights(adapter, directions.unit(layer, position))
        out_dir = config.run_dir / EDITED_MODEL_DIRNAME
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        console.print(f"saved edited model to {out_dir}")

    _write_json(
        config.run_dir / ABLATION_FILENAME,
        {"method": method, "layer": layer, "position": position, "candidates": candidates},
    )
    console.print(f"[green]done[/]: removing the layer-{layer} direction ({method})")


def _measure(
    config: RunConfig,
    adapter: ArchitectureAdapter,
    tokenizer: PreTrainedTokenizerBase,
    harmful: list[str],
    harmless: list[str],
    base_logprobs: torch.Tensor | None,
) -> tuple[EvaluationReport, torch.Tensor]:
    """Refusal rates on both prompt sets, and KL against ``base_logprobs`` if given."""
    settings = config.evaluation
    batch_size = config.prompts.batch_size
    model = adapter.model
    harmful_out = generate_completions(
        model, tokenizer, harmful, settings.max_new_tokens, batch_size
    )
    harmless_out = generate_completions(
        model, tokenizer, harmless, settings.max_new_tokens, batch_size
    )
    logprobs = next_token_logprobs(model, tokenizer, harmless, batch_size)
    report = EvaluationReport(
        harmful_refusal_rate=refusal_rate(harmful_out, settings.refusal_markers),
        harmless_refusal_rate=refusal_rate(harmless_out, settings.refusal_markers),
        harmless_kl=None if base_logprobs is None else kl_divergence(base_logprobs, logprobs),
    )
    return report, logprobs


def _evaluate(config: RunConfig) -> None:
    """Compare base and edited models on held-out prompts."""
    ablation_path = config.run_dir / ABLATION_FILENAME
    if not ablation_path.is_file():
        raise MissingStageOutputError(f"{ablation_path} not found; run `refusal-lens ablate` first")
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    if config.evaluation.capability_benchmark is not None:
        raise NotImplementedError  # capability benchmarks are on the backlog

    adapter, tokenizer = _load_model(config)
    prompts = load_prompt_sets(config.prompts.harmful, config.prompts.harmless, config.seed)
    harmful = format_chat(tokenizer, prompts.harmful.eval, config.prompts.system_prompt)
    harmless = format_chat(tokenizer, prompts.harmless.eval, config.prompts.system_prompt)

    console.print("measuring the base model")
    base, base_logprobs = _measure(config, adapter, tokenizer, harmful, harmless, None)
    console.print("measuring the edited model")
    if ablation["method"] == "hook":
        direction = _require_directions(config).unit(ablation["layer"], ablation["position"])
        with DirectionAblation(adapter, direction):
            edited, _ = _measure(config, adapter, tokenizer, harmful, harmless, base_logprobs)
    else:
        # Evaluate the saved file itself: it is what gets exported.
        del adapter
        edited_adapter, _ = _load_model(config, config.run_dir / EDITED_MODEL_DIRNAME)
        edited, _ = _measure(config, edited_adapter, tokenizer, harmful, harmless, base_logprobs)

    _write_json(
        config.run_dir / "evaluation.json",
        {
            "method": ablation["method"],
            "layer": ablation["layer"],
            "position": ablation["position"],
            "base": asdict(base),
            "edited": asdict(edited),
        },
    )
    table = Table(title=f"layer {ablation['layer']} direction removed ({ablation['method']})")
    table.add_column("metric")
    table.add_column("base", justify="right")
    table.add_column("edited", justify="right")
    table.add_row(
        "harmful refused", f"{base.harmful_refusal_rate:.0%}", f"{edited.harmful_refusal_rate:.0%}"
    )
    table.add_row(
        "harmless refused",
        f"{base.harmless_refusal_rate:.0%}",
        f"{edited.harmless_refusal_rate:.0%}",
    )
    table.add_row("harmless KL", "0", f"{edited.harmless_kl:.3f}")
    console.print(table)


def _export(config: RunConfig) -> None:
    """Export the edited model to the configured formats."""
    raise NotImplementedError


@app.command()
def analyze(config: ConfigOption) -> None:
    """Record activations and compute refusal directions per layer."""
    _run_stage("analyze", _analyze, _prepare_run(config))


@app.command()
def ablate(config: ConfigOption) -> None:
    """Ablate the refusal direction (hooks or weight orthogonalization)."""
    _run_stage("ablate", _ablate, _prepare_run(config))


@app.command()
def evaluate(config: ConfigOption) -> None:
    """Measure refusal rate, KL divergence and capability retention."""
    _run_stage("evaluate", _evaluate, _prepare_run(config))


@app.command()
def export(config: ConfigOption) -> None:
    """Export the edited model to safetensors and/or GGUF."""
    _run_stage("export", _export, _prepare_run(config))


if __name__ == "__main__":  # pragma: no cover
    app()
