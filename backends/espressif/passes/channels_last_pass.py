# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.canonicalize_view_copy_permute_pass import (
    CanonicalizeViewCopyPermutePass,
)
from executorch.backends.transforms.channels_last_layout import PERMUTE_COPY_TARGETS
from executorch.backends.transforms.remove_clone_ops import RemoveCloneOpsTransform
from executorch.backends.transforms.remove_permutes_around_elementwise_ops import (
    RemovePermutesAroundElementwiseOps,
)
from executorch.exir.pass_base import ExportPass, PassResult
from torch.fx import GraphModule


class EspChannelsLastPass(ExportPass):
    """Remove redundant copies around explicit channels-last regions."""

    def call(self, graph_module: GraphModule) -> PassResult:
        # Cancelling copies must not change a tensor's memory layout.
        for node in graph_module.graph.nodes:
            value = node.meta.get("val")
            if isinstance(value, torch.Tensor) and not value.is_contiguous():
                return PassResult(graph_module, False)

        modified = False
        for transform in (
            RemoveCloneOpsTransform(
                preserve_input_output_copies=True,
                eliminate_quant_dequant_pairs=False,
            ),
            RemovePermutesAroundElementwiseOps(),
            CanonicalizeViewCopyPermutePass(permute_targets=PERMUTE_COPY_TARGETS),
        ):
            result = transform(graph_module)
            graph_module = result.graph_module
            modified |= result.modified
        return PassResult(graph_module, modified)
