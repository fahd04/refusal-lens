# refusal-lens documentation

Planned pages:

- **method.md** — difference-of-means directions, direction selection, ablation and
  orthogonalization math.
- **adapters.md** — how to add a new architecture by subclassing
  `refusal_lens.models.ArchitectureAdapter`.
- **configuration.md** — every config field, defaults, and reproducibility notes.
- **apple-silicon.md** — running on MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1`, dtype choices,
  memory limits on unified memory.
- **evaluation.md** — metrics, datasets, and how to reproduce reported results.

## Reproducibility contract

Every CLI command:

1. loads and validates the YAML config (unknown keys are errors),
2. seeds Python and torch RNGs from `seed`,
3. writes `outputs/<name>/resolved_config.yaml` before doing any work.

The resolved file contains all defaults and can be passed back with `--config`.
