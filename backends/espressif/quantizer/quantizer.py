# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch._ops import OpOverload
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    Quantizer,
)


@dataclass(frozen=True)
class EspQuantizationRule:
    is_supported: Callable[[Node], bool]
    annotate: Callable[[Node], QuantizationAnnotation]
    requirements: str


_INT8_ACTIVATION_SPEC = QuantizationSpec(
    dtype=torch.int8,
    observer_or_fake_quant_ctr=MinMaxObserver,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
)


def _is_supported_mul(node: Node) -> bool:
    values = []
    for arg in node.args[:2]:
        value = arg.meta.get("val") if isinstance(arg, Node) else None
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            return False
        if any(type(dim) is not int for dim in value.shape):
            return False
        if not 0 < value.numel() <= torch.iinfo(torch.int32).max:
            return False
        if not value.is_contiguous():
            return False
        values.append(value)
    return values[0].shape == values[1].shape


def _annotate_mul(node: Node) -> QuantizationAnnotation:
    return QuantizationAnnotation(
        input_qspec_map={arg: _INT8_ACTIVATION_SPEC for arg in node.args[:2]},
        output_qspec=_INT8_ACTIVATION_SPEC,
        _annotated=True,
    )


_QUANTIZATION_RULES: dict[OpOverload, EspQuantizationRule] = {
    torch.ops.aten.mul.Tensor: EspQuantizationRule(
        is_supported=_is_supported_mul,
        annotate=_annotate_mul,
        requirements=(
            "static, equal-shaped, contiguous FP32 inputs with a positive int32 "
            "element count"
        ),
    ),
}


class EspQuantizer(Quantizer):
    """Static affine int8 quantization for supported Espressif operators."""

    def annotate(self, model: GraphModule) -> GraphModule:
        for node in model.graph.nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is not None and annotation._annotated:
                continue
            rule = _QUANTIZATION_RULES.get(node.target)
            if rule is None or not rule.is_supported(node):
                continue
            node.meta["quantization_annotation"] = rule.annotate(node)
        return model

    def validate(self, model: GraphModule) -> None:
        for node in model.graph.nodes:
            rule = _QUANTIZATION_RULES.get(node.target)
            annotation = node.meta.get("quantization_annotation")
            if rule is None or annotation is None or not annotation._annotated:
                continue
            if not rule.is_supported(node):
                raise ValueError(
                    f"EspQuantizer requires {rule.requirements} "
                    f"at {node.name} ({node.target})"
                )
