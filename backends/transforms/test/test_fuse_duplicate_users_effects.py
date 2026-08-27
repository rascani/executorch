# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.transforms.fuse_duplicate_users_pass import (
    FuseDuplicateUsersPass,
)
from executorch.exir.dialects._ops import ops as exir_ops

ADD = exir_ops.edge.aten.add.Tensor
RAND_LIKE = exir_ops.edge.aten.rand_like.default
RELU = exir_ops.edge.aten.relu.default

SHAPE = (1, 2, 3, 4)


def _count(graph_module: torch.fx.GraphModule, target: object) -> int:
    return sum(
        node.op == "call_function" and node.target == target
        for node in graph_module.graph.nodes
    )


def _two_users(target: object, extra_args: tuple = ()) -> torch.fx.GraphModule:
    """A placeholder with two identical users of ``target``."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty(SHAPE)
    users = []
    for _ in range(2):
        user = graph.call_function(target, args=(x, *extra_args))
        user.meta["val"] = torch.empty(SHAPE)
        users.append(user)
    graph.output(tuple(users))
    graph.lint()
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_identical_pure_users_are_fused() -> None:
    graph_module = _two_users(RELU)

    result = FuseDuplicateUsersPass()(graph_module)

    assert result.modified
    assert _count(result.graph_module, RELU) == 1


def test_seeded_users_are_not_fused() -> None:
    """Two draws are two draws.

    Fusion shares one call's result among every consumer. For an operator that
    produces a fresh value per call that turns previously independent tensors
    into the same tensor, which is a change in what the graph computes rather
    than in how it computes it.
    """
    graph_module = _two_users(RAND_LIKE)

    result = FuseDuplicateUsersPass()(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RAND_LIKE) == 2


def test_excluded_targets_still_win() -> None:
    graph_module = _two_users(RELU)

    result = FuseDuplicateUsersPass(frozenset({RELU}))(graph_module)

    assert not result.modified
    assert _count(result.graph_module, RELU) == 2
