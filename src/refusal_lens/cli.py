"""``refusal-lens`` command-line interface.

Every command takes ``--config``. Before doing any work it seeds all RNGs from
the config and writes the resolved config into the run directory, so each
output directory is self-describing and reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from refusal_lens import __version__
from refusal_lens.config import RunConfig, load_config, save_resolved_config, seed_everything

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
    raise NotImplementedError


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
