# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
"""
AOT artifact dumper for the standalone Cortex-M55 + MVE inference path.

Takes a post-CortexMPassManager EdgeProgramManager (or ExportedProgram),
runs SpecPropPass + MemoryPlanningAlgorithmSuite([greedy]) to assign arena
offsets, walks the graph in topological order, and emits:

  out_dir/mv2_weights.{c,h}    static const int8/int32 weight & bias arrays
  out_dir/mv2_params.h         static const LayerParams instances + ACT_OFFSET_* + MV2_ARENA_BYTES
  out_dir/input_fixture.h      static const float mv2_fixture_input[] + EXPECTED_TOP1

The inference function and MVE kernels are hand-written in src/mv2_inference.c.
"""

from pathlib import Path
from typing import Optional, Union

import torch
from executorch.exir import EdgeProgramManager
from executorch.exir.memory_planning import MemoryPlanningAlgorithmSuite, apply_algo, greedy
from executorch.exir.passes.spec_prop_pass import make_spec
from torch.export import ExportedProgram
from torch.utils import _pytree as pytree

from .extractors import (
    EXTRACTORS,
    NOOP_OPS,
    DequantOutputLayer,
    ProgramSchedule,
    op_target_string,
)
from .emitters import (
    emit_arena,
    emit_inference_body,
    emit_input_fixture,
    emit_params,
    emit_weights,
)


def _resolve_program(artifact) -> ExportedProgram:
    if isinstance(artifact, EdgeProgramManager):
        return artifact.exported_program()
    if isinstance(artifact, ExportedProgram):
        return artifact
    raise TypeError(f"Unsupported artifact type {type(artifact)}")


def _populate_specs(gm: torch.fx.GraphModule) -> None:
    """Populate node.meta['spec'] for every node in the graph.

    Mirrors SpecPropPass without retracing, so node identities (and thus
    the parameter lookup via the ExportedProgram graph signature) stay
    stable.
    """
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            meta_val = node.meta.get("val", None)
            if node.op == "output":
                node.meta["spec"] = pytree.tree_map(
                    lambda x: x.meta.get("spec", None) if hasattr(x, "meta") else None,
                    node.args[0],
                )
            else:
                node.meta["spec"] = pytree.tree_map(make_spec, meta_val)


def _plan_memory(program: ExportedProgram, alignment: int = 16) -> tuple[dict, dict, int]:
    """Populate specs + run MemoryPlanningAlgorithmSuite([greedy]) and return
    {node_name -> offset}, {node_name -> size}, arena_bytes."""
    gm = program.graph_module
    _populate_specs(gm)
    suite = MemoryPlanningAlgorithmSuite([greedy])
    bufsizes = apply_algo(
        suite,
        gm,
        alignment,
        program.graph_signature,
        alloc_graph_input=False,
        alloc_graph_output=False,
    )
    offsets: dict[str, int] = {}
    sizes: dict[str, int] = {}
    for node in gm.graph.nodes:
        spec = node.meta.get("spec", None)
        if spec is None or not hasattr(spec, "mem_offset") or spec.mem_offset is None:
            continue
        offsets[node.name] = int(spec.mem_offset)
        sizes[node.name] = int(spec.allocated_memory)
    # bufsizes is a list[int] indexed by mem_id; arena=mem_id 1 is the planned space.
    arena = int(bufsizes[1]) if len(bufsizes) > 1 else 0
    return offsets, sizes, arena


def build_schedule(program: ExportedProgram) -> ProgramSchedule:
    """Walk the graph and build a topologically-ordered list of layers."""
    offsets, sizes, arena_bytes = _plan_memory(program)
    gm = program.graph_module

    schedule = ProgramSchedule(arena_bytes=arena_bytes)
    placeholders: list[str] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            # Skip parameters/buffers; only user-supplied activations count.
            if str(node.name) not in program.graph_signature.user_inputs:
                continue
            placeholders.append(str(node.name))
            continue
        if node.op == "output":
            continue
        if node.op != "call_function":
            continue
        target = op_target_string(node)
        if target in NOOP_OPS:
            continue
        extractor = EXTRACTORS.get(target)
        if extractor is None:
            raise NotImplementedError(
                f"No extractor registered for op {target} (node {node.name}). "
                "Supported ops: " + ", ".join(sorted(EXTRACTORS))
            )
        layer = extractor(node, program, offsets, sizes)
        schedule.layers.append(layer)

    # The final dequantize, if present, captures the output scale/zp and the
    # int8 logits buffer becomes the program output.
    final_q_layer = None
    for layer in reversed(schedule.layers):
        if isinstance(layer, DequantOutputLayer):
            schedule.output_scale = layer.scale
            schedule.output_zero_point = layer.zero_point
            schedule.output_slot = layer.input
            break
        if hasattr(layer, "output") and layer.output is not None:
            final_q_layer = layer
            break

    if schedule.output_slot is None and final_q_layer is not None:
        schedule.output_slot = final_q_layer.output

    # Input slot: the first quantize_per_tensor's output is the first int8 buffer
    # that the C inference function writes to.
    for layer in schedule.layers:
        if hasattr(layer, "output") and layer.output is not None:
            schedule.input_slot = layer.output
            break

    return schedule


def dump(
    artifact: Union[EdgeProgramManager, ExportedProgram],
    out_dir: Union[str, Path],
    input_fixture: Optional[torch.Tensor] = None,
    expected_top1: int = -1,
) -> ProgramSchedule:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    program = _resolve_program(artifact)
    schedule = build_schedule(program)

    emit_weights(out_dir, schedule.layers)
    emit_arena(out_dir, schedule)
    emit_params(out_dir, schedule)
    emit_inference_body(out_dir, schedule)
    if input_fixture is not None:
        emit_input_fixture(out_dir, input_fixture, expected_top1)

    return schedule
