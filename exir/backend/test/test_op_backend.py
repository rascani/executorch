# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from executorch.exir import EdgeCompileConfig, to_edge
from executorch.exir.backend.op_backend import OpBackend, to_op_backend


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # atan is non-terminal, so retargeting it leaves the graph signature's
        # user outputs intact.
        return torch.relu(torch.atan(self.linear(x)))


def _edge_program() -> torch.export.ExportedProgram:
    exported = torch.export.export(_Model(), (torch.randn(1, 4),))
    return to_edge(exported).exported_program()


class TestToOpBackend(unittest.TestCase):
    def test_it_returns_what_the_backend_lowered(self) -> None:
        rewritten = _edge_program()

        class Backend(OpBackend):
            saw = None

            def lower(self, exported_program, method_name):
                self.saw = (method_name, exported_program)
                return rewritten

        backend = Backend()
        program = _edge_program()
        self.assertIs(to_op_backend(program, backend), rewritten)
        self.assertEqual(backend.saw, ("forward", program))

    def test_the_method_name_is_passed_through(self) -> None:
        class Backend(OpBackend):
            saw = None

            def lower(self, exported_program, method_name):
                self.saw = method_name
                return exported_program

        backend = Backend()
        to_op_backend(_edge_program(), backend, "decode")
        self.assertEqual(backend.saw, "decode")

    def test_a_backend_that_returns_nothing_is_named(self) -> None:
        # The common authoring slip is a missing return; without this the
        # None travels on and fails somewhere unrelated.
        class Forgetful(OpBackend):
            def lower(self, exported_program, method_name):
                pass

        with self.assertRaisesRegex(
            TypeError, r"Forgetful\.lower\(\) .* got <class 'NoneType'>"
        ):
            to_op_backend(_edge_program(), Forgetful())

    def test_an_unregistered_placeholder_is_named(self) -> None:
        class AddsRogueInput(OpBackend):
            def lower(self, exported_program, method_name):
                # Built fresh, not edited in place: lower() must not mutate its
                # argument, and it keeps the returned program distinguishable
                # from the one that was handed in.
                program = _edge_program()
                graph = program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("rogue")
                return program

        with self.assertRaisesRegex(
            ValueError,
            r"AddsRogueInput\.lower\(\) returned an inconsistent program: "
            r"Number of graph inputs \(4\)",
        ):
            to_op_backend(_edge_program(), AddsRogueInput())

    def test_a_backend_that_returns_a_graph_module_is_named(self) -> None:
        # Likelier than returning None: pass-manager habits produce a
        # GraphModule where the program was wanted.
        class ReturnsGraphModule(OpBackend):
            def lower(self, exported_program, method_name):
                return exported_program.graph_module

        with self.assertRaisesRegex(TypeError, "GraphModule"):
            to_op_backend(_edge_program(), ReturnsGraphModule())

    def test_a_constant_the_backend_adds_is_accepted(self) -> None:
        # The thing a real operator backend does: install a kernel and the
        # constant it needs. Nothing else here exercises a legal rewrite, so
        # without this the checks are never run against a changed program.
        from executorch.backends.transforms.utils import create_constant_placeholder
        from torch.export.graph_signature import InputKind

        class AddsAConstant(OpBackend):
            def lower(self, exported_program, method_name):
                exported_program = _edge_program()
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    create_constant_placeholder(
                        exp_program=exported_program,
                        graph=graph,
                        kind=InputKind.BUFFER,
                        name="op_backend_scale",
                        data=torch.ones(1),
                        persistent_buffer=True,
                    )
                return exported_program

        lowered = to_op_backend(_edge_program(), AddsAConstant())
        self.assertIn("op_backend_scale", lowered.state_dict)

    def test_a_spec_without_its_placeholder_is_rejected(self) -> None:
        # The mirror of the rogue-placeholder case. Reaching the emitter, this
        # one dies on a message-less assertion.
        from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

        class SpecOnly(OpBackend):
            def lower(self, exported_program, method_name):
                exported_program.graph_signature.input_specs.insert(
                    0,
                    InputSpec(InputKind.BUFFER, TensorArgument("ghost"), "ghost", True),
                )
                return exported_program

        with self.assertRaisesRegex(ValueError, "SpecOnly.lower"):
            to_op_backend(_edge_program(), SpecOnly())

    def test_a_registered_input_with_no_tensor_is_rejected(self) -> None:
        # Spec and placeholder agree, but nothing stores the tensor. This is
        # the leg the emitter reports as a bare KeyError.
        from torch.export.graph_signature import InputKind, InputSpec, TensorArgument

        class NoStorage(OpBackend):
            def lower(self, exported_program, method_name):
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("ghost")
                exported_program.graph_signature.input_specs.insert(
                    0,
                    InputSpec(InputKind.BUFFER, TensorArgument("ghost"), "ghost", True),
                )
                return exported_program

        with self.assertRaisesRegex(ValueError, "not in the state dict"):
            to_op_backend(_edge_program(), NoStorage())

    def test_an_exception_from_the_backend_propagates(self) -> None:
        # lower() is told to raise for an operator it cannot lower; swallowing
        # that would ship an unlowered program with no portable fallback.
        class Refuses(OpBackend):
            def lower(self, exported_program, method_name):
                raise NotImplementedError("no kernel for aten.atan")

        with self.assertRaisesRegex(NotImplementedError, "no kernel for aten.atan"):
            to_op_backend(_edge_program(), Refuses())

    def test_the_abc_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            OpBackend()


_kernels = torch.library.Library("op_backend_test", "FRAGMENT")
_kernels.define("atan(Tensor x) -> Tensor")
_kernels.impl("atan", lambda x: x.atan(), "CompositeExplicitAutograd")


class _Retargets(OpBackend):
    """Installs an operator the edge verifier does not know, which is what an
    operator backend is for."""

    def lower(self, exported_program, method_name):
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        graph_module = exported_program.graph_module
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and "atan" in str(node.target):
                with graph_module.graph.inserting_after(node):
                    replacement = graph_module.graph.call_function(
                        torch.ops.op_backend_test.atan.default, node.args
                    )
                replacement.meta.update(node.meta)
                node.replace_all_uses_with(replacement)
                graph_module.graph.erase_node(node)
                break
        graph_module.graph.lint()
        graph_module.recompile()
        return _transform(exported_program, ExportPass())


class _Counting(OpBackend):
    def __init__(self) -> None:
        self.seen: list = []

    def lower(self, exported_program, method_name):
        self.seen.append(method_name)
        return exported_program


def _manager(names=("forward",), **kwargs):
    programs = {
        name: torch.export.export(_Model(), (torch.randn(1, 4),)) for name in names
    }
    return to_edge(
        programs,
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
        **kwargs,
    )


class TestEdgeProgramManagerToOpBackend(unittest.TestCase):
    """The manager-level peer of ``EdgeProgramManager.to_backend``."""

    def test_one_backend_lowers_every_method_in_order(self) -> None:
        backend = _Counting()
        _manager(("zeta", "alpha")).to_op_backend(backend)
        self.assertEqual(backend.seen, ["alpha", "zeta"])

    def test_a_dict_lowers_only_the_methods_it_names(self) -> None:
        backend = _Counting()
        manager = _manager(("forward", "other")).to_op_backend({"forward": backend})
        self.assertEqual(backend.seen, ["forward"])
        self.assertEqual(sorted(manager.methods), ["forward", "other"])

    def test_the_input_manager_is_not_modified(self) -> None:
        manager = _manager()
        self.assertIsNot(manager.to_op_backend(_Counting()), manager)

    def test_constant_methods_and_etrecord_are_carried(self) -> None:
        manager = _manager(constant_methods={"get_max_seq_len": 128})
        manager._etrecord = "sentinel"
        lowered = manager.to_op_backend(_Counting())
        self.assertEqual(lowered._config_methods, {"get_max_seq_len": 128})
        self.assertEqual(lowered._etrecord, "sentinel")
        self.assertIsNot(lowered._config_methods, manager._config_methods)

    def test_a_backend_s_own_operators_survive(self) -> None:
        lowered = _manager().to_op_backend(_Retargets())
        targets = {
            str(n.target)
            for n in lowered.exported_program("forward").graph.nodes
            if n.op == "call_function"
        }
        self.assertIn("op_backend_test.atan.default", targets)

    def test_ir_validity_must_be_off_before_to_edge(self) -> None:
        # Programs keep the verifier they were built with, so this cannot be
        # rescued here: the backend's own _transform rejects its kernels before
        # this method sees them. Pinned because it is the caveat in OpBackend's
        # docstring, and the first thing a backend author trips over.
        strict = to_edge(
            {"forward": torch.export.export(_Model(), (torch.randn(1, 4),))}
        )
        self.assertTrue(strict.compile_config._check_ir_validity)
        with self.assertRaisesRegex(Exception, "is not an Edge operator"):
            strict.to_op_backend(_Retargets())

    def test_the_rest_of_the_compile_config_is_carried(self) -> None:
        # to_backend replaces the config wholesale, dropping preserve_ops with
        # it; carrying it is the one place this deliberately differs.
        manager = to_edge(
            {"forward": torch.export.export(_Model(), (torch.randn(1, 4),))},
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False, preserve_ops=[torch.ops.aten.linear.default]
            ),
        )
        config = manager.to_op_backend(_Counting()).compile_config
        self.assertIn(torch.ops.aten.linear.default, config.preserve_ops)
        self.assertEqual(config.preserve_ops, manager.compile_config.preserve_ops)
        self.assertFalse(config._check_ir_validity)

    def test_a_rewritten_program_still_reaches_a_pte(self) -> None:
        # The reason this method exists: to_executorch is only reachable from a
        # manager, so returning a bare program leaves a hand-staged export with
        # nowhere to put the result.
        lowered = _manager().to_op_backend(_Counting())
        self.assertGreater(len(lowered.to_executorch().buffer), 0)


if __name__ == "__main__":
    unittest.main()
