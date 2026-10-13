# Espressif backend

This backend is under development to accelerate ExecuTorch operators on ESP32-S3 using ESP-NN. The intended frontend uses TorchAO PT2E quantization and the shared `channels_last` dialect for explicit activation layout. Model lowering and ESP-NN kernels remain to be implemented.

## SDK selection

The [SDK setup script](../../examples/espressif/setup.sh) accepts `--idf-version=6.0.1` for Muse Gadgets and `--idf-version=6.1` for the standalone runner; 6.1 remains the default. The default install directories are separate. In an activated Python environment, before sourcing any existing IDF setup:

```bash
./examples/espressif/setup.sh --idf-version=6.0.1 --minimal
source "$HOME/.cache/executorch-espressif-v6.0.1/setup_path.sh"
```

Use a separate shell for the 6.1 configuration. Explicit `ESPRESSIF_TOOLS_DIR`, `IDF_PATH`, and `IDF_TOOLS_PATH` settings override the defaults; an existing SDK must match the selected tag. Keep firmware build directories separate between SDKs. Firmware compatibility with 6.0.1 still requires cross-build validation.
