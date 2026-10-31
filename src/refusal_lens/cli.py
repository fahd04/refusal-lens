"""``refusal-lens`` command-line interface.

Every command takes ``--config``. Before doing any work it seeds all RNGs from
the config and writes the resolved config into the run directory, so each
output directory is self-describing and reproducible.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from refusal_lens import __version__
from refusal_lens.activations import ActivationCache, record_activations
from refusal_lens.config import RunConfig, load_config, save_resolved_config, seed_everything
from refusal_lens.device import resolve_device, resolve_dtype
from refusal_lens.directions import (
    compute_refusal_directions,
    save_directions,
    select_direction,
    separation_scores,
)
from refusal_lens.models import get_adapter, load_model_and_tokenizer
from refusal_lens.prompts import format_chat, load_prompt_sets
from refusal_lens.visualize import (
    plot_direction_norms,
    plot_direction_similarity,
    plot_layer_scores,
    plot_projections,
    save_figure,
)

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
    """Run a pipeline stage, reporting unimplemented stubs cleanly."""
    try:
        stage(config)
    except NotImplementedError:
        console.print(f"[yellow]{name}: not implemented yet[/]")
        raise typer.Exit(code=1) from None


def _analyze(config: RunConfig) -> None:
    """Record activations, compute per-layer directions, and plot them."""
    device = resolve_device(config.model.device)
    dtype = resolve_dtype(config.model.dtype, device)
    console.print(f"loading [bold]{config.model.id}[/] on {device} ({dtype})")
    model, tokenizer = load_model_and_tokenizer(config.model, device, dtype)
    adapter = get_adapter(model)

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
    save_directions(directions, run_dir / "directions.safetensors")
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

    summary = {
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
    }
    (run_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[green]done[/]: clearest separation at layer {best_layer} "
        f"(d={scores[layers.index(best_layer), p]:.2f}); outputs in {run_dir}"
    )


def _ablate(config: RunConfig) -> None:
    """Remove the selected refusal direction and save the edited model."""
    raise NotImplementedError


def _evaluate(config: RunConfig) -> None:
    """Compare base and edited models on held-out prompts."""
    raise NotImplementedError


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
