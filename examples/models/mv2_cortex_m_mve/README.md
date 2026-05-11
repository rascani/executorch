# Standalone MobileNetV2 inference on Cortex-M55 + Helium (MVE)

A hand-written C inference path for MobileNetV2 that targets Cortex-M55
with the Helium (MVE) extension, **without** the ExecuTorch C++ runtime
and **without** CMSIS-NN.  Outputs are bit-exact with the cortex_m
backend's Python reference (i.e., the kernels defined in
`backends/cortex_m/ops/operators.py`) — the same path that
`test_implementation_mv2` exercises through the ExecuTorch runtime.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the architecture
walkthrough with diagrams.

The whole compiled inference function (eight kernels plus the entry
point) lives in a single translation unit (`src/mv2_inference.c`) so
`-O3` can inline kernel bodies into the per-layer call sequence and
constant-fold every parameter literal into the loops.

## Layout

```
mv2_cortex_m_mve/
  tools/                    # AOT artifact dumper (Python)
    dump_mv2_artifacts.py     entrypoint
    extractors.py             per-op extraction from the lowered graph
    emitters.py               C array + struct + arena-offset emission
  include/                  # public headers
    mv2_inference.h           extern "C" mobilenet_v2_inference(...)
    mv2_layer_params.h        LayerParams struct typedefs
    mve_helpers.h             MVE requantize helper + scalar fallback
  src/
    mv2_inference.c           single TU: 8 kernels + call sequence
    arena.c                   static activation arena
    runner_host.c             stdin/stdout harness for host build
    runner_fvp.cpp            semihosted FVP runner
  fvp/
    toolchain-arm-none-eabi.cmake
    corstone-300.ld           customized linker script for the FVP
  generated/                # gitignored; produced by the dumper
  examples_generated/       # checked-in sample dumper output for inspection
```

See `examples_generated/full_mobilenet_v2/` for a checked-in snapshot
of every file the dumper produces for the full torchvision MV2.

The kernels emitted as `static` functions inside `mv2_inference.c`:

| Kernel | Used by full MV2 |
|---|---|
| `quantize_input`             | 1× (entry) |
| `conv2d_s8`                  | 35× |
| `dwconv2d_s8`                | 17× |
| `add_s8`                     | 10× |
| `avgpool_s8`                 |  1× |
| `gemv_s8` (Linear)           |  1× |

Each call site passes shape, padding, zero-points, multiplier/shift
pointers, and activation bounds via a per-layer `static const
LayerParams P_L<i>_*` that the dumper emits into
`generated/mv2_params.h`.

## AOT pipeline

The dumper takes the post-`CortexMPassManager` `ExportedProgram`,
populates `node.meta["spec"]`, runs `exir.memory_planning.greedy` via
`MemoryPlanningAlgorithmSuite([greedy])` (so we reuse the
ExecuTorch arena planner verbatim), walks the graph in topological
order, and emits per-layer artifacts:

| File | Content |
|---|---|
| `mv2_weights.{h,c}` | `static const int8_t` weights, `int32_t` biases / per-channel multipliers / shifts |
| `mv2_params.h`      | `static const LayerParams P_L<i>_*` per layer |
| `mv2_arena.h`       | `MV2_ARENA_BYTES`, `MV2_INPUT_NUM_ELEMENTS`, `MV2_OUTPUT_*`, `ACT_OFFSET_<tensor>` |
| `mv2_inference_body.h` | Body of `mobilenet_v2_inference()` — one call per layer |
| `input_fixture.h`   | Optional `mv2_fixture_input[]` (NHWC float32) |

## Build

Host (no MVE, for local correctness iteration):

```bash
python -m executorch.examples.models.mv2_cortex_m_mve.tools.dump_mv2_artifacts \
    <ExportedProgram-pkl-or-driver-script> generated/
cmake -S examples/models/mv2_cortex_m_mve -B build/host \
      -DMV2_GENERATED_DIR=$(pwd)/examples/models/mv2_cortex_m_mve/generated
cmake --build build/host
./build/host/mv2_runner_host < input.bin > output.bin
```

FVP (Corstone-300, Cortex-M55 + Helium MVE):

```bash
source examples/arm/arm-scratch/setup_path.sh
cmake -S examples/models/mv2_cortex_m_mve -B build/fvp \
      -DCMAKE_TOOLCHAIN_FILE=examples/models/mv2_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
      -DMV2_BUILD_FVP=ON \
      -DMV2_GENERATED_DIR=$(pwd)/examples/models/mv2_cortex_m_mve/generated
cmake --build build/fvp
FVP_Corstone_SSE-300_Ethos-U55 \
    -C ethosu.num_macs=128 \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file=- \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.semihosting-enable=1 \
    -C cpu0.semihosting-stack_base=0 \
    -C cpu0.semihosting-heap_limit=0 \
    -a build/fvp/mv2_runner_fvp.elf \
    --timelimit 1800
```

`Ethos-U55` is in the FVP binary name but the NPU is not used — the
runner only touches the Cortex-M55 + Helium pipeline.

## Tests

A phased pytest suite under `backends/cortex_m/test/models/`
exercises the same pipeline end-to-end and asserts parity with the
cortex_m runtime reference:

- `test_mv2_standalone_phaseA.py` — TinyLinear(8→4); validates the
  dumper, exir memory plan, FVP toolchain, and scalar requantize.
- `test_mv2_standalone_phaseB.py` — Conv + global AvgPool + Linear;
  validates `conv2d_s8` and `avgpool_s8`.
- `test_mv2_standalone_phaseC.py` — single inverted-residual block;
  validates `dwconv2d_s8`, `add_s8`, and residual-hold liveness.
- `test_mv2_standalone_mve.py` — full torchvision MobileNetV2.

Each test prefers the FVP path when `arm-none-eabi-gcc` and
`FVP_Corstone_SSE-300_Ethos-U55` are on PATH (set up via
`examples/arm/arm-scratch/setup_path.sh`), and falls back to a host
build otherwise.

Authored with assistance from Claude (claude.ai/code).
