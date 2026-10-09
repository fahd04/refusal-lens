"""refusal-lens: analyze and edit refusal behavior in open-weight LLMs.

The pipeline is:

1. :mod:`refusal_lens.activations` records residual-stream activations on
   harmful and harmless prompts.
2. :mod:`refusal_lens.directions` computes a per-layer refusal direction as the
   difference of means between the two activation sets.
3. :mod:`refusal_lens.visualize` plots how the direction behaves across layers.
4. :mod:`refusal_lens.ablation` removes the direction, either at inference time
   with hooks or permanently by orthogonalizing weights.
5. :mod:`refusal_lens.evaluate` measures refusal rate, KL divergence and
   capability retention.
6. :mod:`refusal_lens.export` saves the edited model as safetensors or GGUF.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("refusal-lens")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
