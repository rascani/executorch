# Generated code samples

Snapshots of what `tools/dump_mv2_artifacts.py` emits, checked in so
readers can see the shape of the codegen without having to run the
dumper.

## `full_mobilenet_v2/`

Output for the lowered torchvision MobileNetV2 used by
`backends/cortex_m/test/models/test_mv2_standalone_mve.py`:
35 `quantized_conv2d` + 17 `quantized_depthwise_conv2d` + 10
`quantized_add` + 1 `quantized_avg_pool2d` + 1 `quantized_linear`,
bookended by `quantize_per_tensor` / `dequantize_per_tensor`.

| File | Bytes | What it shows |
|---|---|---|
| `mv2_arena.h`             |    4.5 KB | `MV2_ARENA_BYTES` (1.5 MB), `MV2_OUTPUT_*`, and `ACT_OFFSET_*` macros for every tensor placed by the exir greedy memory plan |
| `mv2_params.h`            |     30 KB | One `static const LayerParams` per layer with shape / zp / multiplier-shift pointers |
| `mv2_weights.{c,h}`       | 21 MB + 9.6 KB | int8 weights (formatted as `0xNN` per byte), int32 bias, per-channel multiplier / shift arrays |
| `mv2_inference_body.h`    |     13 KB | The body of `mobilenet_v2_inference()` — one call per layer, arena offsets and `LayerParams` pointers as integer literals |
| `input_fixture.h`         |    2.1 MB | NHWC float input (`torch.randn(1, 3, 224, 224)` for this snapshot) the runner embeds at build time |

`mv2_inference_body.h` is included verbatim by `src/mv2_inference.c`,
so the C compiler sees the kernel definitions and the call sequence
together in one translation unit and can inline + constant-fold them.

The weight bytes here come from a `torch.manual_seed(0)` +
`torchvision.models.mobilenet_v2(weights=None)` run — so they're random
init, not the ImageNet-trained weights.  A pretrained run produces the
same file *structure* with different byte values.

`mv2_weights.c` is the bulk of the directory (21 MB of `0xNN, ` hex
literals).  It compresses to roughly 4 MB in pack files but is the
single biggest reason this directory exists at all — open
`mv2_inference_body.h` and `mv2_params.h` first if you just want to see
what the codegen looks like.
