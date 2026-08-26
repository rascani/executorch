# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


CORTEX_M_BACKEND: str = "cortex_m"


class CortexMRecipeType(RecipeType):
    """Cortex-M recipe types.

    Cortex-M lowers by rewriting edge operators into ``cortex_m::`` CMSIS-NN
    kernels rather than by delegating a subgraph, so a recipe carries no
    partitioner.

    CMSIS-NN works on NHWC data, and the quantizer's conv checkers reject a
    convolution whose operands are not channels_last, so a model with 4-D
    operands has to be exported from channels_last example inputs or from a
    model moved to channels_last::

        inputs = (torch.randn(1, 3, 96, 96).to(memory_format=torch.channels_last),)
        session = export(model=model, example_inputs=[inputs], export_recipe=recipe)

    The layout at trace time is what decides this, and the trace uses the
    first entry of ``example_inputs`` -- every entry is calibration data, but
    only the first is traced. ``aot_arm_compiler.py`` behaves the same way: it
    moves the model to channels_last, but only after its first export, so
    contiguous inputs are annotated contiguous there too. Neither path can fix
    it for you.

    Moving the model as well as the inputs is worth it for a conv-heavy model:
    leaving the model contiguous emits each convolution weight twice, once
    contiguous and once channels_last. Moving the model alone is not enough,
    and is worse than doing nothing -- the convolution then lowers to a CMSIS
    kernel that rejects its own NCHW input at runtime.

    Get the layout wrong and the convolutions stay on the portable float
    kernels: on a conv-heavy model that roughly doubles the program, and the
    extra ``aten::`` kernels have to be in the runner's operator list or it
    fails to load. The recipe warns when it sees this.

    Explicit-layout lowering removes this requirement by converting the layout
    in the graph. Once it is available the recipe should expose it, and this
    becomes legacy-mode behaviour.

    Accepted kwargs:
        target (str): ``cortex-m<variant>`` CPU to compile for, as accepted by
            ``CortexMTargetConfig.from_target_string``. Defaults to
            ``"cortex-m55"``.
        isa (cmsis_nn.Backend): Override the CMSIS-NN backend that ``target``
            would otherwise imply, for cores whose ISA extensions are optional
            (an M55 built without MVE, an M33 without DSP). Validated against
            the CPU's capabilities.

    """

    INT8 = "cortex_m_int8"

    @classmethod
    def get_backend_name(cls) -> str:
        return CORTEX_M_BACKEND
