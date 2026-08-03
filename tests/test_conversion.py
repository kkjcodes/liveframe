from __future__ import annotations

import pytest

from liveframe.backends import TensorDescriptor
from liveframe.converters.cfpp_to_mlx import ConversionError, plan_cfpp_conversion


def test_conversion_plan_maps_stock_wan_names_and_shapes() -> None:
    inventory = (
        TensorDescriptor("patch_embedding.weight", (1536, 16, 1, 2, 2), "torch.float32"),
        TensorDescriptor("text_embedding.0.weight", (1536, 4096), "torch.float32"),
        TensorDescriptor("blocks.0.ffn.0.weight", (8960, 1536), "torch.float32"),
        TensorDescriptor("head.head.bias", (64,), "torch.float32"),
    )

    plan = plan_cfpp_conversion(inventory)
    plan.require_complete()

    assert plan.rules[0].target_name == "patch_embedding_proj.weight"
    assert plan.rules[0].target_shape == (1536, 64)
    assert plan.rules[0].transform == "flatten-conv3d-input"
    assert plan.rules[1].target_name == "text_embedding_0.weight"
    assert plan.rules[2].target_name == "blocks.0.ffn.fc1.weight"
    assert all(rule.target_dtype == "bfloat16" for rule in plan.rules)


def test_unknown_cfpp_tensor_fails_closed() -> None:
    plan = plan_cfpp_conversion(
        (TensorDescriptor("causal_forcing.mystery", (1,), "torch.float32"),)
    )

    with pytest.raises(ConversionError, match="unmapped"):
        plan.require_complete()


def test_patch_embedding_shape_is_validated() -> None:
    inventory = (TensorDescriptor("patch_embedding.weight", (2, 2), "float32"),)

    with pytest.raises(ConversionError, match="five-dimensional"):
        plan_cfpp_conversion(inventory)
