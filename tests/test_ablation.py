from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from refusal_lens.ablation import (
    DirectionAblation,
    orthogonalize_matrix,
    orthogonalize_weights,
    project_out,
)
from refusal_lens.activations import ResidualStreamRecorder
from refusal_lens.models import ArchitectureAdapter, get_adapter

D_MODEL = 32  # tiny_model hidden size


@pytest.fixture
def adapter(tiny_model: PreTrainedModel) -> ArchitectureAdapter:
    return get_adapter(tiny_model)


@pytest.fixture
def direction() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(D_MODEL) * 3  # deliberately not unit length


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, 100, (3, 9))


def _logits(adapter: ArchitectureAdapter, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return adapter.model(input_ids=input_ids).logits


def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def _tied_model(arch: str) -> PreTrainedModel:
    """Like `tiny_model`, but the output layer shares the input embedding (as in Qwen2.5-1.5B)."""
    torch.manual_seed(0)
    dims = {
        "vocab_size": 128,
        "hidden_size": D_MODEL,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "tie_word_embeddings": True,
    }
    model: PreTrainedModel
    if arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**dims))
    else:
        model = LlamaForCausalLM(LlamaConfig(**dims))
    model.train(False)
    return model


# --- project_out -------------------------------------------------------------------


def test_project_out_removes_the_component(direction: torch.Tensor) -> None:
    x = torch.randn(4, 5, D_MODEL)

    out = project_out(x, direction)

    torch.testing.assert_close(out @ _unit(direction), torch.zeros(4, 5), atol=1e-5, rtol=0)


def test_project_out_keeps_everything_else(direction: torch.Tensor) -> None:
    """Only the part along the direction changes: x = out + (x . r) r."""
    x = torch.randn(6, D_MODEL)
    r = _unit(direction)

    out = project_out(x, direction)

    torch.testing.assert_close(out + (x @ r)[:, None] * r, x)


def test_project_out_is_idempotent(direction: torch.Tensor) -> None:
    once = project_out(torch.randn(3, D_MODEL), direction)

    torch.testing.assert_close(project_out(once, direction), once, atol=1e-5, rtol=0)


def test_project_out_keeps_dtype(direction: torch.Tensor) -> None:
    x = torch.randn(2, D_MODEL, dtype=torch.bfloat16)

    assert project_out(x, direction).dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("direction", "match"), [(torch.zeros(D_MODEL), "zero"), (torch.ones(D_MODEL + 1), "size")]
)
def test_project_out_bad_direction_raises(direction: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        project_out(torch.randn(2, D_MODEL), direction)


# --- orthogonalize_matrix --------------------------------------------------------------


def test_orthogonalized_linear_cannot_write_along_direction(direction: torch.Tensor) -> None:
    weight = torch.randn(D_MODEL, 64)  # nn.Linear(64 -> d_model)

    edited = orthogonalize_matrix(weight, direction, output_dim=0)

    outputs = torch.randn(10, 64) @ edited.T  # [10, d_model]
    torch.testing.assert_close(outputs @ _unit(direction), torch.zeros(10), atol=1e-4, rtol=0)


def test_orthogonalized_embedding_rows_are_orthogonal(direction: torch.Tensor) -> None:
    weight = torch.randn(128, D_MODEL)  # nn.Embedding(vocab, d_model)

    edited = orthogonalize_matrix(weight, direction, output_dim=1)

    torch.testing.assert_close(edited @ _unit(direction), torch.zeros(128), atol=1e-4, rtol=0)


def test_square_matrix_uses_output_dim(direction: torch.Tensor) -> None:
    """o_proj is square in Qwen ([d_model, d_model]): the shape alone can't tell the axes apart."""
    weight = torch.randn(D_MODEL, D_MODEL)

    as_linear = orthogonalize_matrix(weight, direction, output_dim=0)
    as_embedding = orthogonalize_matrix(weight, direction, output_dim=1)

    r = _unit(direction)
    torch.testing.assert_close(r @ as_linear, torch.zeros(D_MODEL), atol=1e-4, rtol=0)
    torch.testing.assert_close(as_embedding @ r, torch.zeros(D_MODEL), atol=1e-4, rtol=0)


def test_orthogonalize_matrix_returns_new_tensor_same_dtype(direction: torch.Tensor) -> None:
    weight = torch.randn(D_MODEL, 64, dtype=torch.bfloat16)
    original = weight.clone()

    edited = orthogonalize_matrix(weight, direction)

    assert edited.dtype == torch.bfloat16
    assert edited.shape == weight.shape
    assert weight.equal(original)


def test_orthogonalize_matrix_wrong_size_raises(direction: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="size"):
        orthogonalize_matrix(torch.randn(64, D_MODEL), direction, output_dim=0)


# --- DirectionAblation (temporary, hooks) -------------------------------------------------


def test_hook_ablation_clears_direction_from_every_layer(
    adapter: ArchitectureAdapter, direction: torch.Tensor, input_ids: torch.Tensor
) -> None:
    layers = list(range(adapter.num_layers))
    with (
        DirectionAblation(adapter, direction),
        ResidualStreamRecorder(adapter, layers, positions=[-1, -3]) as rec,
    ):
        adapter.model(input_ids=input_ids)

    projections = rec.collect().activations @ _unit(direction)
    torch.testing.assert_close(projections, torch.zeros_like(projections), atol=1e-4, rtol=0)


def test_hook_ablation_changes_output_then_restores_it(
    adapter: ArchitectureAdapter, direction: torch.Tensor, input_ids: torch.Tensor
) -> None:
    before = _logits(adapter, input_ids)

    with DirectionAblation(adapter, direction):
        during = _logits(adapter, input_ids)
    after = _logits(adapter, input_ids)

    assert not torch.allclose(during, before)
    torch.testing.assert_close(after, before)


def test_hook_ablation_removes_hooks_on_error(
    adapter: ArchitectureAdapter, direction: torch.Tensor, input_ids: torch.Tensor
) -> None:
    before = _logits(adapter, input_ids)

    with pytest.raises(RuntimeError, match="boom"), DirectionAblation(adapter, direction):
        raise RuntimeError("boom")

    torch.testing.assert_close(_logits(adapter, input_ids), before)


def test_hook_ablation_bad_layer_raises(
    adapter: ArchitectureAdapter, direction: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="layer"):
        DirectionAblation(adapter, direction, layers=[adapter.num_layers])


# --- orthogonalize_weights (permanent) ----------------------------------------------------


def test_weight_edit_matches_hooks(
    adapter: ArchitectureAdapter, direction: torch.Tensor, input_ids: torch.Tensor
) -> None:
    """The key guarantee: permanent and temporary ablation give the same model."""
    with DirectionAblation(adapter, direction):
        hooked = _logits(adapter, input_ids)

    orthogonalize_weights(adapter, direction)

    torch.testing.assert_close(_logits(adapter, input_ids), hooked, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(("layers", "include_embedding"), [([1], True), ([0], False)])
def test_weight_edit_matches_hooks_for_subsets(
    adapter: ArchitectureAdapter,
    direction: torch.Tensor,
    input_ids: torch.Tensor,
    layers: list[int],
    include_embedding: bool,
) -> None:
    with DirectionAblation(adapter, direction, layers, include_embedding=include_embedding):
        hooked = _logits(adapter, input_ids)

    orthogonalize_weights(adapter, direction, layers, include_embedding=include_embedding)

    torch.testing.assert_close(_logits(adapter, input_ids), hooked, atol=1e-4, rtol=1e-4)


def test_weight_edit_only_touches_selected_layers(
    adapter: ArchitectureAdapter, direction: torch.Tensor
) -> None:
    untouched = [w.weight.clone() for w in adapter.residual_writers(0)]
    embedding = adapter.embedding.weight.clone()

    orthogonalize_weights(adapter, direction, layers=[1], include_embedding=False)

    for writer, original in zip(adapter.residual_writers(0), untouched, strict=True):
        assert writer.weight.equal(original)
    assert adapter.embedding.weight.equal(embedding)
    assert not any(
        w.weight.equal(o) for w, o in zip(adapter.residual_writers(1), untouched, strict=True)
    )


def test_weight_edit_keeps_dtype(direction: torch.Tensor, tiny_model: PreTrainedModel) -> None:
    adapter = get_adapter(tiny_model.to(torch.bfloat16))

    orthogonalize_weights(adapter, direction)

    assert adapter.embedding.weight.dtype == torch.bfloat16
    assert all(w.weight.dtype == torch.bfloat16 for w in adapter.residual_writers(0))


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_tied_embeddings_are_untied_and_still_match_hooks(
    arch: str, direction: torch.Tensor, input_ids: torch.Tensor
) -> None:
    """Editing a shared embedding must not silently edit the output layer too."""
    adapter = get_adapter(_tied_model(arch))
    lm_head = adapter.model.get_output_embeddings()
    assert lm_head is not None
    assert lm_head.weight is adapter.embedding.weight  # tied before
    with DirectionAblation(adapter, direction):
        hooked = _logits(adapter, input_ids)

    orthogonalize_weights(adapter, direction)

    lm_head = adapter.model.get_output_embeddings()
    assert lm_head is not None
    assert lm_head.weight is not adapter.embedding.weight
    assert adapter.model.config.tie_word_embeddings is False
    torch.testing.assert_close(_logits(adapter, input_ids), hooked, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("arch", ["qwen2", "llama"])
def test_edited_tied_model_survives_save_and_reload(
    arch: str, direction: torch.Tensor, input_ids: torch.Tensor, tmp_path: Path
) -> None:
    adapter = get_adapter(_tied_model(arch))
    orthogonalize_weights(adapter, direction)
    edited = _logits(adapter, input_ids)

    adapter.model.save_pretrained(tmp_path / "edited")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "edited", dtype=torch.float32)

    torch.testing.assert_close(_logits(get_adapter(reloaded), input_ids), edited)
