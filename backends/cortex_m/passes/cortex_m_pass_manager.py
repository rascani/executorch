# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import inspect
from typing import Callable, cast, Optional, Type

from executorch.backends.arm._passes import (
    FoldAndAnnotateQParamsPass,
    ScalarsToAttributePass,
)
from executorch.backends.transforms.remove_getitem_op import RemoveGetItemPass
from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)
from executorch.exir.pass_base import ExportPass
from executorch.exir.pass_manager import PassManager
from executorch.exir.program._program import _transform, lift_constant_tensor_pass
from torch.export import ExportedProgram
from torch.fx.passes.infra.pass_base import PassResult

from torch.nn import Module

from .activation_fusion_pass import ActivationFusionPass
from .clamp_hardswish_pass import ClampHardswishPass
from .convert_gelu_pass import ConvertGELUPass
from .convert_layer_norm_pass import ConvertLayerNormPass
from .convert_to_cortex_m_pass import ConvertToCortexMPass
from .decompose_hardswish_pass import DecomposeHardswishPass
from .decompose_mean_pass import DecomposeMeanPass
from .dwconv_project_fusion_pass import DwconvProjectFusionPass
from .expand_dwconv_fusion_pass import ExpandDwconvFusionPass
from .expand_dwconv_project_fusion_pass import ExpandDwconvProjectFusionPass
from .quantize_stem_dwconv_project_fusion_pass import QuantizeStemDwconvProjectFusionPass
from .quantize_stem_inverted_residual_fusion_pass import (
    QuantizeStemInvertedResidualFusionPass,
)
from .stem_dwconv_project_fusion_pass import StemDwconvProjectFusionPass
from .quantized_clamp_activation_pass import QuantizedClampActivationPass
from .quantized_op_fusion_pass import QuantizedOpFusionPass
from .replace_quant_nodes_pass import ReplaceQuantNodesPass

PassClass = Type[ExportPass]


class CortexMPassManager(PassManager):
    pass_list: list[PassClass] = [
        # Run before folding so qparams attach to max_pool2d values, not tuple + getitem.
        RemoveGetItemPass,
        FoldAndAnnotateQParamsPass,
        ReplaceScalarWithTensorArgPass,
        ReplaceQuantNodesPass,
        # Phase 2: ConvertGELUPass must run before QuantizedOpFusionPass,
        # which strips the `approximate` kwarg from aten.gelu and would
        # otherwise lose the erf/tanh distinction.
        ConvertGELUPass,
        ActivationFusionPass,
        QuantizedClampActivationPass,
        DecomposeHardswishPass,
        QuantizedOpFusionPass,
        ConvertToCortexMPass,
        # Phase 1 (KWT transformer support): pattern-match
        # cortex_m.dequant → aten.native_layer_norm → getitem → cortex_m.quant
        # and rewrite to cortex_m::quantized_layer_norm.  Runs *after*
        # ConvertToCortexMPass so the surrounding quant/dequant ops are
        # already in the cortex_m namespace.
        ConvertLayerNormPass,
    ]

    # Opt-in extension that fuses MV2 expand+dwconv chains.  Not on the
    # default pass list because the cortex_m runtime kernel registry has no
    # CMSIS-NN entry for the fused op; only the standalone Cortex-M55 + MVE
    # codegen at examples/models/mv2_cortex_m_mve/ consumes it.
    pass_list_with_expand_dwconv_fusion: list[PassClass] = pass_list + [
        ExpandDwconvFusionPass,
    ]

    # Same opt-in path but with the additional project conv (and optional
    # residual add) absorbed into a single fused op per inverted-residual
    # block.  Eliminates both the dwconv output and the project output
    # intermediates from the arena.  DwconvProjectFusionPass runs last and
    # picks up MV2 B0-style blocks (expand_ratio=1) that don't match the
    # expand+dwconv pattern.
    pass_list_with_inverted_residual_fusion: list[PassClass] = pass_list + [
        ExpandDwconvFusionPass,
        ExpandDwconvProjectFusionPass,
        DwconvProjectFusionPass,
        StemDwconvProjectFusionPass,
        QuantizeStemDwconvProjectFusionPass,
    ]

    # Phase G: extends the inverted-residual fusion path one more step,
    # absorbing MV2's first inverted-residual block (B1) into the
    # quantize+stem+B0 mega-op.  Eliminates the B0 project output (200 KB
    # at 1.0/224) from the arena; the rolling-buffer C kernel was
    # validated bit-exact vs Phase F on host.  Opt-in until we have full
    # FVP validation; export_mv2.py's picker will fall back to Phase F
    # at small widths where the deeper fusion doesn't pay off.
    pass_list_with_phase_g: list[PassClass] = pass_list_with_inverted_residual_fusion + [
        QuantizeStemInvertedResidualFusionPass,
    ]

    pass_list_transform_for_annotation: list[PassClass] = [
        ScalarsToAttributePass,
        ReplaceScalarWithTensorArgPass,
        ClampHardswishPass,
        DecomposeMeanPass,
    ]

    def __init__(
        self, exported_program, passes: Optional[list[PassClass]] = None
    ) -> None:
        super().__init__(passes=[])
        self.exported_program = exported_program
        # PassManager.passes is typed as callables; this manager stores pass classes which are initialized at transform time with the exported_program.
        self.passes: list[PassClass] = (  # type: ignore[assignment]
            passes if passes is not None else self.pass_list  # type: ignore[assignment]
        )

    def transform_for_annotation(self, model):
        passes = self.pass_list_transform_for_annotation
        for p in passes:
            model = p().call(model).graph_module
        return model

    def transform(self) -> ExportedProgram:
        ep = self.exported_program
        for pass_cls in self.passes:
            signature = inspect.signature(pass_cls)
            if "exported_program" in signature.parameters:
                ep_pass_ctor = cast(Callable[[ExportedProgram], ExportPass], pass_cls)
                transform_pass = ep_pass_ctor(ep)
            else:
                transform_pass = pass_cls()
            pass_callable = cast(Callable[[Module], PassResult], transform_pass)
            ep = _transform(ep, pass_callable)

        # All constant tensors should be lifted to buffers at this point, re-run
        # lift_constant_tensor_pass in case new ones have been introduced by the passes above.
        ep = lift_constant_tensor_pass(ep)
        return ep
