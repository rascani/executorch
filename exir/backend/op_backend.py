# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod

from torch._export.verifier import _verify_exported_program_signature
from torch.export import ExportedProgram


class OpBackend(ABC):
    """A backend that lowers operators in place rather than delegating them.

    A ``Partitioner`` may only tag nodes; ``to_backend`` asserts the graph it
    returns is identical. An operator backend rewrites the graph instead,
    replacing operators with its own kernels and adding the constants those
    kernels need, so it runs as a peer to a delegate rather than through one.

    Its kernels are outside the edge dialect, so the program has to reach it
    with ``_check_ir_validity`` off. Programs keep the verifier they were built
    with, so that is a property of the ``to_edge`` call, not of anything the
    backend can do: with it on, the backend's own ``_transform`` rejects the
    operators it just installed. ``to_backend`` clears the flag for delegates;
    an operator backend has no equivalent hook and its caller must.
    """

    @abstractmethod
    def lower(
        self, exported_program: ExportedProgram, method_name: str
    ) -> ExportedProgram:
        """Return a rewritten copy of one method's program.

        Do not mutate the argument; the caller may still be holding it.

        An input added here needs both a spec in
        ``graph_signature.input_specs`` and its tensor stored -- in
        ``state_dict`` for a parameter or persistent buffer, in ``constants``
        otherwise. It must also be placed before every user input, which the
        emitter does not require but both in-tree helpers do
        (``backends.transforms.utils.create_constant_placeholder``, and
        ``lift_constant_tensor_pass`` for a ``get_attr`` node).

        Raise, naming the node, for an operator this backend claimed but
        cannot lower: preserved operators are not decomposed, so there is no
        portable fallback and it would reach the runtime unlowered.

        ``method_name`` is for diagnostics.
        """


def to_op_backend(
    exported_program: ExportedProgram,
    op_backend: OpBackend,
    method_name: str = "forward",
) -> ExportedProgram:
    """Lower one method's program through an operator backend.

    The peer of ``to_backend``. Checks that the returned graph signature still
    describes the graph, which an added input easily breaks; unchecked, that
    surfaces later as a message-less assertion or a bare ``KeyError`` from the
    emitter, naming neither the backend nor the input.

    Deliberately not ``ExportedProgram.validate()``: that re-enters the edge
    verifier, which rejects the operators an operator backend exists to
    install.
    """
    lowered = op_backend.lower(exported_program, method_name)

    name = type(op_backend).__name__
    if not isinstance(lowered, ExportedProgram):
        raise TypeError(
            f"{name}.lower() must return an ExportedProgram, got {type(lowered)}"
        )

    try:
        _verify_exported_program_signature(lowered)
    except Exception as e:
        raise ValueError(f"{name}.lower() returned an inconsistent program: {e}") from e
    return lowered
