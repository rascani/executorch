# Espressif backend

This backend is under development to accelerate ExecuTorch operators on ESP32-S3 using ESP-NN. `EspQuantizer`, available from `executorch.backends.espressif.quantizer`, provides TorchAO PT2E quantization for multiply with static, equal-shaped, contiguous FP32 inputs. Inputs and output use independent per-tensor affine int8 quantization. Unsupported cases stay unannotated; model lowering and ESP-NN kernels follow separately.

`EspPassManager`, available from `executorch.backends.espressif.passes`, integrates with `to_edge_transform_and_lower(program, transform_passes=EspPassManager())` and `edge.transform(EspPassManager())`. Its initial pass list is empty; operator lowering passes will be added alongside their kernels. `EspTester` exposes both paths through `.to_edge_transform_and_lower()` and `.to_edge().run_passes()`.

For PTE serialization, import `executorch.kernels.quantized` to register the portable Q/DQ out variants from the ExecuTorch build.

Run the backend tests in an ExecuTorch Python environment:

```bash
python -m pytest backends/espressif/test
```

## SDK selection

The [SDK setup script](../../examples/espressif/setup.sh) accepts `--idf-version=6.0.1` for Muse Gadgets and `--idf-version=6.1` for the standalone runner; 6.1 remains the default. The default install directories are separate. In an activated Python environment, before sourcing any existing IDF setup:

```bash
./examples/espressif/setup.sh --idf-version=6.0.1 --minimal
source "$HOME/.cache/executorch-espressif-v6.0.1/setup_path.sh"
```

Use a separate shell for the 6.1 configuration. Explicit `ESPRESSIF_TOOLS_DIR`, `IDF_PATH`, and `IDF_TOOLS_PATH` settings override the defaults; an existing SDK must match the selected tag. Keep firmware build directories separate between SDKs. Firmware compatibility with 6.0.1 still requires cross-build validation.
