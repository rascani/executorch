#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Export a simple add model with XNNPACK delegation."""

import torch
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


class AddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y


model = AddModel()
example_inputs = (torch.randn(2, 2), torch.randn(2, 2))

exported = torch.export.export(model, example_inputs, strict=True)

et_program = to_edge_transform_and_lower(
    exported,
    partitioner=[XnnpackPartitioner()],
    compile_config=EdgeCompileConfig(_check_ir_validity=False),
).to_executorch()

with open("add_xnnpack.pte", "wb") as f:
    f.write(et_program.buffer)

print("Exported add_xnnpack.pte")
