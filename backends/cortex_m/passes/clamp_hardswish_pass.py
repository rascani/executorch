# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

import torch

from executorch.exir.dialects.edge._ops import EdgeOpOverload
from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from torch.fx.node import Argument


class ClampHardswishPass(ExportPass):
    """
    Clamps the input to hardswish to [-3, inf) before quantization.

    Mathematically a no-op, since hardswish is zero below -3, but it narrows
    the range the observer sees. Hardswish lowers to an int8 LUT and is near
    the identity for large inputs, so its input grid wants to be about as fine
    as its output grid. This is an accuracy heuristic; nothing depends on it
    for correctness.
    """

    def call_operator(
        self,
        op: EdgeOpOverload,
        args: tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        if op in (torch.ops.aten.hardswish.default, torch.ops.aten.hardswish_.default):
            clamped_args = (args[0], -3)
            clamped_input = super().call_operator(
                torch.ops.aten.clamp.default, clamped_args, {}, meta
            )
            args = cast(tuple[Argument, ...], (clamped_input,))

        return super().call_operator(op, args, kwargs, meta)
