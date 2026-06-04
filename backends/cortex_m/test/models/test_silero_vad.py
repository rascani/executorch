# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.backends.arm.test.common import parametrize
from executorch.backends.cortex_m.test.tester import CortexMTester, McuTestCase
from executorch.examples.models.silero_vad.export_silero_vad import (
    CONTEXT_SIZE,
    HIDDEN_DIM,
    SileroVAD16k,
    WINDOW_SIZE,
)


ops_before_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 6,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 5,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 5,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 4,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_sqrt_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_dequantize_per_tensor_default": 20,
    "executorch_exir_dialects_edge__ops_quantized_decomposed_quantize_per_tensor_default": 18,
}
# The STFT magnitude block `(real**2 + imag**2).sqrt()` now lowers: pow->mul
# (DecomposePowPass) gives 2 quantized_mul, the sum gives 1 quantized_add, and
# sqrt lowers to a quantized_activation LUT -- so the only remaining
# quantized_activation pair is sqrt + the final-conv sigmoid (2 total).
# The 3 remaining sigmoids and 2 tanhs are LSTMCell gates: PyTorch export
# captures nn.LSTMCell as a single high-level op, so the quantizer never sees
# the gate activations and can't annotate them. They're decomposed only at
# to_edge -- after the quantizer -- so the gates have no qparams to fold and
# the lowering pass correctly skips them (a pre-annotation LSTMCell decompose
# lands separately). The 6 Conv1ds likewise stay fp32 until conv1d support
# lands, so their conv-tail relus stay in aten.
ops_after_transforms: dict[str, int] = {
    "executorch_exir_dialects_edge__ops_aten_abs_default": 2,
    "executorch_exir_dialects_edge__ops_aten_add_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_arange_start_step": 1,
    "executorch_exir_dialects_edge__ops_aten_cat_default": 1,
    "executorch_exir_dialects_edge__ops_aten_convolution_default": 6,
    "executorch_exir_dialects_edge__ops_aten_index_Tensor": 1,
    "executorch_exir_dialects_edge__ops_aten_linear_default": 2,
    "executorch_exir_dialects_edge__ops_aten_mean_dim": 1,
    "executorch_exir_dialects_edge__ops_aten_mul_Tensor": 3,
    "executorch_exir_dialects_edge__ops_aten_relu_default": 5,
    "executorch_exir_dialects_edge__ops_aten_select_copy_int": 2,
    "executorch_exir_dialects_edge__ops_aten_sigmoid_default": 3,
    "executorch_exir_dialects_edge__ops_aten_slice_copy_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_split_with_sizes_copy_default": 1,
    "executorch_exir_dialects_edge__ops_aten_squeeze_copy_dims": 2,
    "executorch_exir_dialects_edge__ops_aten_sub_Tensor": 2,
    "executorch_exir_dialects_edge__ops_aten_tanh_default": 2,
    "executorch_exir_dialects_edge__ops_aten_unsqueeze_copy_default": 2,
    "executorch_exir_dialects_edge__ops_aten_view_copy_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_dequantize_per_tensor_default": 7,
    "executorch_exir_dialects_edge__ops_cortex_m_quantize_per_tensor_default": 6,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_activation_default": 2,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_add_default": 1,
    "executorch_exir_dialects_edge__ops_cortex_m_quantized_mul_default": 2,
}


pt_model = SileroVAD16k().eval()

x = torch.randn(
    1, CONTEXT_SIZE + WINDOW_SIZE
)  # (1, 576) — 64 context + 512 audio samples
state = torch.zeros(2, 1, HIDDEN_DIM)  # (2, 1, 128) — [h, c] LSTM state

test_cases = {
    "silero_vad_16k": McuTestCase(
        model=pt_model,
        example_inputs=lambda: (x, state),
    ),
}


@parametrize("test_case", test_cases)
def test_dialect_silero_vad_16k(test_case):
    """The STFT magnitude block (pow/add/sqrt) lowers to cortex_m ops; the
    LSTMCell gates and Conv1d encoder do not yet lower. This test tracks
    development progress."""
    inputs = test_case.get_example_inputs()
    tester = CortexMTester(test_case.model, inputs)
    tester.test_dialect(
        ops_before_transforms,
        ops_after_transforms,
        qtol=10,
    )
