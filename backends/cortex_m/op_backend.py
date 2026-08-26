# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import logging
from dataclasses import dataclass

from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.passes.passes_utils import is_channels_last
from executorch.backends.cortex_m.target_config import CortexMTargetConfig
from executorch.exir.backend.op_backend import OpBackend
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import ExportedProgram


logger: logging.Logger = logging.getLogger(__name__)

_CONVOLUTIONS = (
    exir_ops.edge.aten.convolution.default,
    exir_ops.edge.aten.conv1d.default,
)


def _report_portable_kernels(method_name: str, program: ExportedProgram) -> None:
    """Name the portable kernels the runner has to register.

    A selective build missing one fails at load with OperatorMissing, so this
    is worth reporting on every export, not only on a fallback.
    """
    portable = sorted(
        {
            name
            for graph_module in program.graph_module.modules()
            if hasattr(graph_module, "graph")
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            for name in [getattr(node.target, "_name", "")]
            # "::" excludes the higher-order operators -- call_delegate, cond --
            # which name no kernel the runner registers. Submodules are walked
            # because a control-flow branch needs its kernels registered too.
            if "::" in name and not name.startswith("cortex_m::")
        }
    )
    if portable:
        logger.info(
            "Method '%s' needs portable kernels for: %s. The runner's operator "
            "list has to cover them.",
            method_name,
            ", ".join(portable),
        )


def _warn_on_contiguous_convolution(method_name: str, program: ExportedProgram) -> None:
    """Name convolutions the layout requirement kept on the portable kernel.

    Separate from the list above because this fallback has a fix the caller can
    apply: CMSIS-NN needs NHWC, and the conversion has to happen before
    quantization observes the model.
    """
    contiguous = []
    for node in program.graph.nodes:
        if node.op != "call_function" or node.target not in _CONVOLUTIONS:
            continue
        # The quantizer gates on each pattern node's own output, so that is
        # what has to be checked here; an input can be channels_last while the
        # result the checker sees is not.
        value = node.meta.get("val")
        if value is not None and value.dim() == 4 and not is_channels_last(value):
            contiguous.append(node.name)

    if contiguous:
        logger.warning(
            "Method '%s' keeps the portable float kernel for %s: the result is "
            "not channels_last, which the Cortex-M quantizer requires of a "
            "convolution. Trace from inputs built with "
            "`.to(memory_format=torch.channels_last)`; the layout at trace time "
            "is what the quantizer annotates against.",
            method_name,
            ", ".join(contiguous),
        )


@dataclass(frozen=True)
class CortexMOpBackend(OpBackend):
    """Lowers the graph to CMSIS-NN kernels.

    Each pass is built against the previous pass's output, so the pass manager
    drives the sequence rather than handing back a pass list.
    """

    target_config: CortexMTargetConfig

    def lower(
        self, exported_program: ExportedProgram, method_name: str
    ) -> ExportedProgram:
        # The passes store lowered weights in the state dict, and `_transform`
        # hands the same mapping down the chain. Copy it so the caller's
        # program is left as it was; the tensors themselves are shared.
        program = copy.copy(exported_program)
        program._state_dict = dict(exported_program.state_dict)

        program = CortexMPassManager(
            program, target_config=self.target_config
        ).transform()
        _report_portable_kernels(method_name, program)
        _warn_on_contiguous_convolution(method_name, program)
        return program
