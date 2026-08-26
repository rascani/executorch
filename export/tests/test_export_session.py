# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import List
from unittest.mock import Mock

import torch
from executorch.exir.backend.op_backend import OpBackend
from executorch.export import ExportRecipe, ExportSession
from executorch.export.recipe import (
    AOQuantizationConfig,
    LoweringRecipe,
    QuantizationRecipe,
)
from executorch.export.stages import PipelineArtifact, Stage
from executorch.export.types import StageType


class SimpleTestModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestExportSessionCoreFlow(unittest.TestCase):
    """Test core export flow and pipeline execution."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    def _create_mock_stage(self, stage_type: StageType) -> Mock:
        mock_stage = Mock()
        mock_artifact = Mock(spec=PipelineArtifact)
        mock_artifact.data = Mock()
        mock_artifact.context = {}
        mock_stage.get_artifacts.return_value = mock_artifact
        mock_stage.stage_type = stage_type

        # Add the new properties required by the Stage interface
        if stage_type == StageType.SOURCE_TRANSFORM:
            mock_stage.valid_predecessor_stages = []
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.QUANTIZE:
            mock_stage.valid_predecessor_stages = [StageType.SOURCE_TRANSFORM]
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.TORCH_EXPORT:
            mock_stage.valid_predecessor_stages = [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
            ]
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.TO_EDGE_TRANSFORM_AND_LOWER:
            mock_stage.valid_predecessor_stages = [StageType.TORCH_EXPORT]
            mock_stage.can_start_pipeline = True
        elif stage_type == StageType.TO_EXECUTORCH:
            mock_stage.valid_predecessor_stages = [
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_BACKEND,
            ]
            mock_stage.can_start_pipeline = True
        else:
            mock_stage.valid_predecessor_stages = []
            mock_stage.can_start_pipeline = True

        return mock_stage

    def test_default_pipeline_execution_order(self) -> None:
        # Test that pipeline stages are executed in the correct order
        stage_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        mock_stages = [
            self._create_mock_stage(stage_type) for stage_type in stage_types
        ]

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        # Replace the stages in the registry with our mocked stages
        for stage_type, mock_stage in zip(stage_types, mock_stages):
            session.register_stage(stage_type, mock_stage)

        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 5)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

    def test_overriden_pipeline_execution_order(self) -> None:
        # Test when pipeline stages that are passed through recipe
        stage_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        mock_stages = [
            self._create_mock_stage(stage_type) for stage_type in stage_types
        ]

        self.recipe.pipeline_stages = stage_types
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        # Replace the stages in the registry with our mocked stages
        for stage_type, mock_stage in zip(stage_types, mock_stages):
            session.register_stage(stage_type, mock_stage)
        session.export()

        # Verify all stages were called
        for stage in mock_stages:
            stage.run.assert_called_once()

        # Verify artifacts were stored for each stage
        self.assertEqual(len(session._stage_to_artifacts), 4)
        self.assertEqual(set(session._stage_to_artifacts.keys()), set(stage_types))

    def test_model_standardization_single_to_dict(self) -> None:
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        self.assertIsInstance(session._model, dict)
        self.assertIn("forward", session._model)
        self.assertEqual(session._model["forward"], self.model)

        self.assertIsInstance(session._example_inputs, dict)
        self.assertIn("forward", session._example_inputs)
        self.assertEqual(session._example_inputs["forward"], self.example_inputs)

    def test_model_standardization_preserves_dict(self) -> None:
        # Test that dictionary models are preserved as-is.
        model_dict = {"method1": self.model, "method2": SimpleTestModel()}
        inputs_dict = {
            "method1": self.example_inputs,
            "method2": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,  # pyre-ignore[6]
            example_inputs=inputs_dict,
            export_recipe=self.recipe,
        )

        self.assertEqual(session._model, model_dict)
        self.assertEqual(session._example_inputs, inputs_dict)

    def test_context_propagation_through_pipeline(self) -> None:
        # Test that context is properly propagated through the pipeline
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
            name="test_session",
            constant_methods={"const_method": lambda: torch.tensor([1, 2, 3])},
        )

        # Check that initial context is set up correctly
        expected_context_keys = {
            "example_inputs",
            "dynamic_shapes",
            "constant_methods",
            "export_recipe",
            "session_name",
            "artifact_dir",
            "generate_etrecord",
        }
        self.assertEqual(set(session._run_context.keys()), expected_context_keys)
        self.assertEqual(session._run_context["session_name"], "test_session")
        self.assertIsNotNone(session._run_context["constant_methods"])

    def test_stage_registry_unknown_stage_type(self) -> None:
        # Test error handling for unknown stage types in pipeline
        unknown_stage_type = Mock()
        unknown_stage_type.name = "UNKNOWN_STAGE"
        recipe = ExportRecipe(name="test", pipeline_stages=[unknown_stage_type])

        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=self.model,
                example_inputs=self.example_inputs,
                export_recipe=recipe,
            )._run_pipeline()
        self.assertIn("not found in registry", str(cm.exception))

    def test_multi_method_model_export(self) -> None:
        # Test export with multi-method models
        model_dict = {
            "forward": self.model,
            "inference": SimpleTestModel(),
        }
        inputs_dict = {
            "forward": self.example_inputs,
            "inference": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,  # pyre-ignore[6]
            example_inputs=inputs_dict,
            export_recipe=ExportRecipe(name="multi_method_test"),
        )

        # Verify proper initialization
        self.assertEqual(session._model, model_dict)
        self.assertEqual(session._example_inputs, inputs_dict)

        # Test getting example inputs for different methods
        forward_input = session.get_example_input("forward")
        inference_input = session.get_example_input("inference")

        self.assertEqual(forward_input, self.example_inputs[0])
        self.assertEqual(inference_input, inputs_dict["inference"][0])


class TestPipelineValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    # pyre-ignore
    def _get_export_session(self, stages: List[StageType]):
        self.recipe.pipeline_stages = stages
        return ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

    def test_valid_pipeline_sequences(self) -> None:
        """Test various valid pipeline sequences."""
        valid_sequences = [
            # Full pipeline with to_edge_transform_lower
            [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Full pipeline with to_edge, to_backend
            [
                StageType.SOURCE_TRANSFORM,
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
            # Skip quantize
            [
                StageType.SOURCE_TRANSFORM,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Skip source transform and start with quantize
            [
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Start with torch export
            [
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Start with edge transform and lower (ExportedProgram input)
            [
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
            # Start with to_edge and to_backend
            [
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        ]

        for i, stages in enumerate(valid_sequences):
            with self.subTest(sequence=i, stages=[s.name for s in stages]):
                session = self._get_export_session(stages)
                # Should not raise any exception
                try:
                    session._validate_pipeline_sequence(stages)
                except Exception as e:
                    self.fail(f"Valid sequence {[s.name for s in stages]} raised {e}")

    def test_invalid_pipeline_start_stages(self) -> None:
        """Test stages that cannot start a pipeline."""
        invalid_stage_sequence = [
            # Executorch stage cannot start pipeline (requires edge stage first)
            [StageType.TO_EXECUTORCH],
            # Backend stage cannot start pipeline (requires TO_EDGE first)
            [StageType.TO_BACKEND],
            [StageType.TO_BACKEND, StageType.TO_EXECUTORCH],
        ]

        for i, stages in enumerate(invalid_stage_sequence):
            with self.subTest(sequence=i, stages=[s.name for s in stages]):
                session = self._get_export_session(stages)
                with self.assertRaises(ValueError) as cm:
                    session._validate_pipeline_sequence(stages)
                self.assertIn("cannot start a pipeline", str(cm.exception))

    def test_pipeline_transitions(self) -> None:
        """Test both valid and invalid pipeline transitions"""
        test_cases = [
            # Valid cases
            ([StageType.SOURCE_TRANSFORM, StageType.QUANTIZE], True),
            ([StageType.QUANTIZE, StageType.TORCH_EXPORT], True),
            ([StageType.SOURCE_TRANSFORM, StageType.TORCH_EXPORT], True),
            ([StageType.TORCH_EXPORT, StageType.TO_EDGE_TRANSFORM_AND_LOWER], True),
            # Invalid cases - transitions
            ([StageType.QUANTIZE, StageType.TO_EDGE_TRANSFORM_AND_LOWER], False),
            (
                [StageType.SOURCE_TRANSFORM, StageType.TO_EDGE_TRANSFORM_AND_LOWER],
                False,
            ),
            (
                [
                    StageType.TORCH_EXPORT,
                    StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                    StageType.QUANTIZE,
                ],
                False,
            ),
            ([StageType.TO_EXECUTORCH, StageType.TORCH_EXPORT], False),
        ]

        for i, (stages, should_pass) in enumerate(test_cases):
            with self.subTest(
                sequence=i, stages=[s.name for s in stages], should_pass=should_pass
            ):
                session = self._get_export_session(stages)
                if should_pass:
                    try:
                        session._validate_pipeline_sequence(stages)
                    except Exception as e:
                        self.fail(
                            f"Expected valid sequence {[s.name for s in stages]} but got {e}"
                        )
                else:
                    with self.assertRaises(ValueError):
                        session._validate_pipeline_sequence(stages)

    def test_empty_pipeline_sequence(self) -> None:
        """Test empty pipeline sequence."""
        session = self._get_export_session([])
        with self.assertRaises(ValueError) as cm:
            session._validate_pipeline_sequence([])
        self.assertIn("Pipeline stages cannot be empty", str(cm.exception))


class TestExportSessionErrorHandling(unittest.TestCase):
    """Test error handling in export session."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]
        self.recipe = ExportRecipe(name="test")

    def test_access_results_before_export(self) -> None:
        """Test that accessing results before export raises appropriate errors."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_executorch_program_manager()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_executorch_program()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_pte_buffer()
        self.assertIn(
            "Executorch program manager is not initialized", str(cm.exception)
        )

    def test_invalid_method_name_in_example_inputs(self) -> None:
        """Test error handling for invalid method names."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(KeyError) as cm:
            session.get_example_input("nonexistent_method")
        self.assertIn("Method name 'nonexistent_method' not found", str(cm.exception))

    def test_empty_example_inputs_list(self) -> None:
        """Test error handling for empty example inputs."""
        session = ExportSession(
            model={"forward": self.model},
            example_inputs={"forward": []},
            export_recipe=self.recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.get_example_input("forward")
        self.assertIn(
            "Example inputs list for method forward is empty", str(cm.exception)
        )

    def test_save_to_pte_invalid_name(self) -> None:
        """Test save_to_pte with invalid output name."""
        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=self.recipe,
        )

        with self.assertRaises(AssertionError):
            session.save_to_pte("")

        with self.assertRaises(AssertionError):
            session.save_to_pte(None)  # pyre-ignore


class TestExportSessionPipelineBuilding(unittest.TestCase):
    """Test pipeline building and stage configuration."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    def test_pipeline_building_with_all_recipes(self) -> None:
        """Test pipeline building with quantization and lowering recipes."""
        # Create comprehensive recipes
        quant_recipe = QuantizationRecipe(
            ao_quantization_configs=[AOQuantizationConfig(Mock())],
            quantizers=[Mock()],
        )
        lowering_recipe = LoweringRecipe(
            partitioners=[Mock()],
            edge_transform_passes=[Mock()],
            edge_compile_config=Mock(),
        )
        recipe = ExportRecipe(
            name="comprehensive_test",
            quantization_recipe=quant_recipe,
            lowering_recipe=lowering_recipe,
            executorch_backend_config=Mock(),
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        registered_stages = session.get_all_registered_stages()

        self.assertEqual(len(registered_stages), 5)
        expected_types = [
            StageType.SOURCE_TRANSFORM,
            StageType.QUANTIZE,
            StageType.TORCH_EXPORT,
            StageType.TO_EDGE_TRANSFORM_AND_LOWER,
            StageType.TO_EXECUTORCH,
        ]
        self.assertListEqual(list(registered_stages.keys()), expected_types)


class TestExportSessionExtendedInputTypes(unittest.TestCase):
    """Test extended input type support (GraphModule, ExportedProgram, etc.)"""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = (torch.randn(2, 10),)
        self.recipe = ExportRecipe(name="test")

    def test_nn_module_input_type_detection(self) -> None:
        """Test that nn.Module input is detected correctly."""
        session = ExportSession(
            model=self.model,
            example_inputs=[self.example_inputs],
            export_recipe=self.recipe,
        )

        self.assertEqual(session._input_model_type, "nn.Module")

        # Verify default pipeline includes quantization stages
        pipeline = session._get_default_pipeline()
        self.assertIn(StageType.SOURCE_TRANSFORM, pipeline)
        self.assertIn(StageType.QUANTIZE, pipeline)
        self.assertIn(StageType.TORCH_EXPORT, pipeline)
        self.assertIn(StageType.TO_EDGE_TRANSFORM_AND_LOWER, pipeline)
        self.assertIn(StageType.TO_EXECUTORCH, pipeline)

    def test_graph_module_input_type_detection(self) -> None:
        """Test that GraphModule input is detected correctly."""
        # Create a GraphModule using fx.symbolic_trace
        graph_module = torch.fx.symbolic_trace(self.model)

        session = ExportSession(
            model=graph_module,
            example_inputs=[self.example_inputs],
            export_recipe=self.recipe,
        )

        self.assertEqual(session._input_model_type, "GraphModule")

        # Verify default pipeline skips quantization stages
        pipeline = session._get_default_pipeline()
        self.assertNotIn(StageType.SOURCE_TRANSFORM, pipeline)
        self.assertNotIn(StageType.QUANTIZE, pipeline)
        self.assertIn(StageType.TORCH_EXPORT, pipeline)
        self.assertIn(StageType.TO_EDGE_TRANSFORM_AND_LOWER, pipeline)
        self.assertIn(StageType.TO_EXECUTORCH, pipeline)

    def test_exported_program_input_type_detection(self) -> None:
        """Test that ExportedProgram input is detected correctly."""
        # Create an ExportedProgram
        exported_program = torch.export.export(self.model, self.example_inputs)

        # ExportedProgram should not require example_inputs
        session = ExportSession(
            model=exported_program,
            export_recipe=self.recipe,
        )

        self.assertEqual(session._input_model_type, "ExportedProgram")

        # Verify default pipeline skips quantization and torch export stages
        pipeline = session._get_default_pipeline()
        self.assertNotIn(StageType.SOURCE_TRANSFORM, pipeline)
        self.assertNotIn(StageType.QUANTIZE, pipeline)
        self.assertNotIn(StageType.TORCH_EXPORT, pipeline)
        self.assertIn(StageType.TO_EDGE_TRANSFORM_AND_LOWER, pipeline)
        self.assertIn(StageType.TO_EXECUTORCH, pipeline)

    def test_dict_nn_module_input_type_detection(self) -> None:
        """Test that Dict[str, nn.Module] input is detected correctly."""
        model_dict = {
            "forward": self.model,
            "method2": SimpleTestModel(),
        }
        inputs_dict = {
            "forward": [self.example_inputs],
            "method2": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,
            example_inputs=inputs_dict,
            export_recipe=self.recipe,
        )

        # Should detect type based on first value
        self.assertEqual(session._input_model_type, "nn.Module")

    def test_dict_graph_module_input_type_detection(self) -> None:
        """Test that Dict[str, GraphModule] input is detected correctly."""
        graph_module1 = torch.fx.symbolic_trace(self.model)
        graph_module2 = torch.fx.symbolic_trace(SimpleTestModel())

        model_dict = {
            "forward": graph_module1,
            "method2": graph_module2,
        }
        inputs_dict = {
            "forward": [self.example_inputs],
            "method2": [(torch.randn(1, 10),)],
        }

        session = ExportSession(
            model=model_dict,
            example_inputs=inputs_dict,
            export_recipe=self.recipe,
        )

        # Should detect GraphModule type
        self.assertEqual(session._input_model_type, "GraphModule")

        # Verify pipeline skips quantization
        pipeline = session._get_default_pipeline()
        self.assertNotIn(StageType.QUANTIZE, pipeline)

    def test_dict_exported_program_input_type_detection(self) -> None:
        """Test that Dict[str, ExportedProgram] input is detected correctly."""
        ep1 = torch.export.export(self.model, self.example_inputs)
        ep2 = torch.export.export(SimpleTestModel(), (torch.randn(1, 10),))

        model_dict = {
            "forward": ep1,
            "method2": ep2,
        }

        session = ExportSession(
            model=model_dict,
            export_recipe=self.recipe,
        )

        # Should detect ExportedProgram type
        self.assertEqual(session._input_model_type, "ExportedProgram")

        # Verify pipeline skips export stages
        pipeline = session._get_default_pipeline()
        self.assertNotIn(StageType.TORCH_EXPORT, pipeline)

    def test_example_inputs_required_for_nn_module(self) -> None:
        """Test that example_inputs are required for nn.Module."""
        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=self.model,
                export_recipe=self.recipe,
            )
        self.assertIn("example_inputs are required", str(cm.exception))
        self.assertIn("nn.Module", str(cm.exception))

    def test_example_inputs_required_for_graph_module(self) -> None:
        """Test that example_inputs are required for GraphModule."""
        graph_module = torch.fx.symbolic_trace(self.model)

        with self.assertRaises(ValueError) as cm:
            ExportSession(
                model=graph_module,
                export_recipe=self.recipe,
            )
        self.assertIn("example_inputs are required", str(cm.exception))
        self.assertIn("GraphModule", str(cm.exception))

    def test_example_inputs_optional_for_exported_program(self) -> None:
        """Test that example_inputs are optional for ExportedProgram."""
        exported_program = torch.export.export(self.model, self.example_inputs)

        # Should not raise
        session = ExportSession(
            model=exported_program,
            export_recipe=self.recipe,
        )

        self.assertEqual(session._input_model_type, "ExportedProgram")

    def test_validation_graph_module_cannot_run_quantization(self) -> None:
        """Test that GraphModule input cannot run quantization stages."""
        graph_module = torch.fx.symbolic_trace(self.model)

        # Try to force quantization stages
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.QUANTIZE,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=graph_module,
            example_inputs=[self.example_inputs],
            export_recipe=recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.export()
        self.assertIn("Cannot run", str(cm.exception))
        self.assertIn("stage(s)", str(cm.exception))
        self.assertIn("QUANTIZE", str(cm.exception))
        self.assertIn("GraphModule", str(cm.exception))

    def test_validation_graph_module_cannot_run_source_transform(self) -> None:
        """Test that GraphModule input cannot run source transform stage."""
        graph_module = torch.fx.symbolic_trace(self.model)

        # Try to force source transform stage
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.SOURCE_TRANSFORM,
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=graph_module,
            example_inputs=[self.example_inputs],
            export_recipe=recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.export()
        self.assertIn("Cannot run", str(cm.exception))
        self.assertIn("stage(s)", str(cm.exception))
        self.assertIn("SOURCE_TRANSFORM", str(cm.exception))
        self.assertIn("GraphModule", str(cm.exception))

    def test_validation_exported_program_cannot_run_torch_export(self) -> None:
        """Test that ExportedProgram input cannot run torch export stage."""
        exported_program = torch.export.export(self.model, self.example_inputs)

        # Try to force torch export stage
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=exported_program,
            export_recipe=recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.export()
        self.assertIn("Cannot run", str(cm.exception))
        self.assertIn("stage(s)", str(cm.exception))
        self.assertIn("TORCH_EXPORT", str(cm.exception))
        self.assertIn("ExportedProgram", str(cm.exception))

    def test_validation_exported_program_cannot_run_quantization(self) -> None:
        """Test that ExportedProgram input cannot run quantization stages."""
        exported_program = torch.export.export(self.model, self.example_inputs)

        # Try to force quantization stages
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.QUANTIZE,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=exported_program,
            export_recipe=recipe,
        )

        with self.assertRaises(ValueError) as cm:
            session.export()
        self.assertIn("Cannot run", str(cm.exception))
        self.assertIn("stage(s)", str(cm.exception))
        self.assertIn("QUANTIZE", str(cm.exception))
        self.assertIn("ExportedProgram", str(cm.exception))

    def test_graph_module_valid_pipeline(self) -> None:
        """Test valid pipeline for GraphModule input."""
        graph_module = torch.fx.symbolic_trace(self.model)

        # Valid pipeline starting from torch export
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=graph_module,
            example_inputs=[self.example_inputs],
            export_recipe=recipe,
        )

        # Should not raise during validation
        session._validate_pipeline_sequence(recipe.pipeline_stages)

    def test_exported_program_valid_pipeline(self) -> None:
        """Test valid pipeline for ExportedProgram input."""
        exported_program = torch.export.export(self.model, self.example_inputs)

        # Valid pipeline starting from edge stages
        recipe = ExportRecipe(
            pipeline_stages=[
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ]
        )

        session = ExportSession(
            model=exported_program,
            export_recipe=recipe,
        )

        # Should not raise during validation
        session._validate_pipeline_sequence(recipe.pipeline_stages)


class TestIntermediateStateGetters(unittest.TestCase):
    """Test convenience getters for intermediate pipeline states."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.example_inputs = [(torch.randn(2, 10),)]

    def test_get_exported_program_after_torch_export(self) -> None:
        """Test that get_exported_program works after torch export stage."""
        recipe = ExportRecipe(
            name="test",
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        exported_program = session.get_exported_program()
        self.assertIsNotNone(exported_program)
        self.assertIsInstance(exported_program, torch.export.ExportedProgram)

    def test_get_exported_program_before_export_fails(self) -> None:
        """Test that get_exported_program fails before torch export stage."""
        recipe = ExportRecipe(name="test")

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        with self.assertRaises(RuntimeError) as cm:
            session.get_exported_program()
        self.assertIn("Exported program is not available", str(cm.exception))

    def test_get_exported_program_invalid_method_name(self) -> None:
        """Test that get_exported_program fails with invalid method name."""
        recipe = ExportRecipe(name="test")

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        with self.assertRaises(KeyError) as cm:
            session.get_exported_program("nonexistent_method")
        self.assertIn("Method name 'nonexistent_method' not found", str(cm.exception))

    def test_get_exported_program_multi_method(self) -> None:
        """Test get_exported_program with multi-method model."""
        model_dict = {
            "forward": self.model,
            "inference": SimpleTestModel(),
        }
        inputs_dict = {
            "forward": self.example_inputs,
            "inference": [(torch.randn(1, 10),)],
        }

        recipe = ExportRecipe(name="multi_method_test")

        session = ExportSession(
            model=model_dict,
            example_inputs=inputs_dict,
            export_recipe=recipe,
        )

        session.export()

        forward_ep = session.get_exported_program("forward")
        inference_ep = session.get_exported_program("inference")

        self.assertIsNotNone(forward_ep)
        self.assertIsNotNone(inference_ep)
        self.assertIsInstance(forward_ep, torch.export.ExportedProgram)
        self.assertIsInstance(inference_ep, torch.export.ExportedProgram)

    def test_get_edge_program_manager_with_transform_and_lower(self) -> None:
        """Test get_edge_program_manager with TO_EDGE_TRANSFORM_AND_LOWER stage."""
        recipe = ExportRecipe(
            name="test",
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.TO_EXECUTORCH,
            ],
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        edge_manager = session.get_edge_program_manager()
        self.assertIsNotNone(edge_manager)

    def test_get_edge_program_manager_with_separate_stages(self) -> None:
        """Test get_edge_program_manager with separate TO_EDGE and TO_BACKEND stages."""
        recipe = ExportRecipe(
            name="test",
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        edge_manager = session.get_edge_program_manager()
        self.assertIsNotNone(edge_manager)

    def test_get_edge_program_manager_before_edge_stage_fails(self) -> None:
        """Test that get_edge_program_manager fails before edge stages."""
        recipe = ExportRecipe(
            name="test",
            pipeline_stages=[StageType.TORCH_EXPORT],
        )

        session = ExportSession(
            model=self.model,
            example_inputs=self.example_inputs,
            export_recipe=recipe,
        )

        session.export()

        with self.assertRaises(RuntimeError) as cm:
            session.get_edge_program_manager()
        self.assertIn("Edge program manager is not available", str(cm.exception))


class TestStageArtifactsAreScopedToOneRun(unittest.TestCase):
    def test_rerunning_export_discards_the_previous_run(self) -> None:
        # A stage that raises must not leave the previous run's artifact behind
        # for the accessors to hand back as if it were current.
        model = SimpleTestModel()
        inputs = [(torch.randn(1, 10),)]
        session = ExportSession(
            model=model, example_inputs=inputs, export_recipe=ExportRecipe(name="t")
        )
        session.export()
        first = session.get_stage_artifacts()[StageType.TO_EXECUTORCH]

        artifacts = session.get_stage_artifacts()

        failing = Mock(spec=Stage)
        failing.run.side_effect = RuntimeError("boom")
        failing.stage_type = StageType.TO_EXECUTORCH
        failing.valid_predecessor_stages = [StageType.TO_EDGE_TRANSFORM_AND_LOWER]
        failing.can_start_pipeline = False
        session.register_stage(StageType.TO_EXECUTORCH, failing)

        with self.assertRaises(RuntimeError):
            session.export()
        self.assertNotIn(StageType.TO_EXECUTORCH, session.get_stage_artifacts())
        # Cleared in place, so a dict the caller captured before the re-run
        # reflects the clear too.
        self.assertNotIn(StageType.TO_EXECUTORCH, artifacts)
        self.assertIsNotNone(first)

    def test_a_rejected_pipeline_leaves_the_previous_run_intact(self) -> None:
        # Validation runs before anything is cleared: a pipeline that never
        # executes must not destroy results the caller still has.
        session = ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(name="t"),
        )
        session.export()
        before = session.get_executorch_program_manager()

        session._pipeline_stages = [StageType.TO_EXECUTORCH]
        with self.assertRaises(ValueError):
            session.export()
        self.assertIs(session.get_executorch_program_manager(), before)


class _RecordingOpBackend(OpBackend):
    """Records the methods it lowered; optionally applies a real pass."""

    def __init__(self, apply_pass: bool = False) -> None:
        self.seen: List[str] = []
        self._apply_pass = apply_pass

    def lower(self, exported_program, method_name):
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        self.seen.append(method_name)
        return (
            _transform(exported_program, ExportPass())
            if self._apply_pass
            else exported_program
        )


class TestOpBackendPipeline(unittest.TestCase):
    """A lowering recipe that declares op_backends has to get the
    stage that runs them, whatever the input model type."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.inputs = [(torch.randn(1, 10),)]

    def _recipe(self, real_pass: bool = False, **kwargs) -> ExportRecipe:
        self.backend = _RecordingOpBackend(apply_pass=real_pass)
        return ExportRecipe(
            name="ppt",
            lowering_recipe=LoweringRecipe(op_backends=[self.backend]),
            **kwargs,
        )

    def test_stage_added_to_default_pipeline(self) -> None:
        session = ExportSession(
            model=self.model, example_inputs=self.inputs, export_recipe=self._recipe()
        )
        stages = session._pipeline_stages
        self.assertGreater(
            stages.index(StageType.EDGE_PROGRAM_MANAGER_TRANSFORM),
            stages.index(StageType.TO_EDGE_TRANSFORM_AND_LOWER),
        )
        session.export()
        self.assertEqual(self.backend.seen, ["forward"])

    def test_stage_absent_without_op_backends(self) -> None:
        session = ExportSession(
            model=self.model,
            example_inputs=self.inputs,
            export_recipe=ExportRecipe(name="plain"),
        )
        self.assertNotIn(
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM, session._pipeline_stages
        )

    def test_accepts_exported_program_input(self) -> None:
        exported = torch.export.export(self.model, self.inputs[0])
        session = ExportSession(
            model=exported, example_inputs=self.inputs, export_recipe=self._recipe()
        )
        self.assertNotIn(StageType.TORCH_EXPORT, session._pipeline_stages)
        self.assertIn(
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM, session._pipeline_stages
        )
        session.export()
        self.assertEqual(self.backend.seen, ["forward"])

    def test_omitting_the_stage_from_pipeline_stages_is_rejected(self) -> None:
        # The default pipeline places the stage; a caller-supplied list that
        # leaves it out would silently drop the transforms.
        with self.assertRaisesRegex(ValueError, "EDGE_PROGRAM_MANAGER_TRANSFORM"):
            ExportSession(
                model=self.model,
                example_inputs=self.inputs,
                export_recipe=self._recipe(
                    pipeline_stages=[
                        StageType.TORCH_EXPORT,
                        StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                        StageType.TO_EXECUTORCH,
                    ]
                ),
            ).export()

    def test_edge_program_manager_getter_returns_the_lowered_program(self) -> None:
        session = ExportSession(
            model=self.model,
            example_inputs=self.inputs,
            export_recipe=self._recipe(real_pass=True),
        )
        session.export()
        artifacts = session.get_stage_artifacts()
        lowered = artifacts[StageType.TO_EDGE_TRANSFORM_AND_LOWER].data
        transformed = artifacts[StageType.EDGE_PROGRAM_MANAGER_TRANSFORM].data
        # Guard the guard: the stage always rebuilds the manager, so comparing
        # managers proves nothing -- it is the programs that must differ.
        self.assertIsNot(
            lowered.exported_program("forward"),
            transformed.exported_program("forward"),
        )
        self.assertIs(session.get_edge_program_manager(), transformed)


_kernels = torch.library.Library("op_backend_test", "FRAGMENT")
_kernels.define("scale(Tensor x) -> Tensor")
_kernels.impl("scale", lambda x: x * 2.0, "CompositeExplicitAutograd")
# to_executorch needs an out variant for every operator left in the graph.
_kernels.define("scale.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)")
_kernels.impl(
    "scale.out",
    lambda x, out: out.copy_(x * 2.0),
    "CompositeExplicitAutograd",
)


class _TwoOpModel(torch.nn.Module):
    """The retargeted operator has to be non-terminal, or replacing it would
    invalidate the graph signature's user outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.linear: torch.nn.Module = torch.nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # atan is not delegated by XNNPACK, so the backend has something left
        # to retarget in both the partitioned and unpartitioned cases.
        return torch.relu(torch.atan(self.linear(x)))


class _RetargetingOpBackend(OpBackend):
    """Does what the ABC describes: swaps an operator for its own kernel."""

    def lower(self, exported_program, method_name):
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        graph_module = exported_program.graph_module
        for node in list(graph_module.graph.nodes):
            if node.op == "call_function" and "atan" in str(node.target):
                with graph_module.graph.inserting_after(node):
                    replacement = graph_module.graph.call_function(
                        torch.ops.op_backend_test.scale.default, (node.args[0],)
                    )
                replacement.meta.update(node.meta)
                node.replace_all_uses_with(replacement)
                graph_module.graph.erase_node(node)
                break
        graph_module.graph.lint()
        graph_module.recompile()
        return _transform(exported_program, ExportPass())


class TestOpBackendMisuseIsReported(unittest.TestCase):
    def test_the_abc_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            OpBackend()

    def test_the_offending_backend_is_named_in_a_chain(self) -> None:
        # Checking after the whole chain instead of after each backend would
        # report whichever ran last, and only once the emitter tripped over it.
        class AddsRogueInput(OpBackend):
            def lower(self, exported_program, method_name):
                graph = exported_program.graph
                first = next(n for n in graph.nodes if n.op == "placeholder")
                with graph.inserting_before(first):
                    graph.placeholder("rogue")
                return exported_program

        recipe = ExportRecipe(
            name="chain",
            lowering_recipe=LoweringRecipe(
                op_backends=[AddsRogueInput(), _RecordingOpBackend()]
            ),
        )
        with self.assertRaisesRegex(ValueError, "AddsRogueInput.lower"):
            ExportSession(
                model=SimpleTestModel(),
                example_inputs=[(torch.randn(1, 10),)],
                export_recipe=recipe,
            ).export()


class TestOpBackendsMayLeaveTheEdgeDialect(unittest.TestCase):
    """The whole point of an operator backend is to install operators the edge
    verifier does not know, so the stage must not re-verify against it."""

    def _export(self, **lowering):
        session = ExportSession(
            model=_TwoOpModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(
                name="retarget",
                lowering_recipe=LoweringRecipe(
                    op_backends=[_RetargetingOpBackend()], **lowering
                ),
            ),
        )
        session.export()
        return session

    def _ops(self, session):
        program = session.get_edge_program_manager().exported_program("forward")
        return {str(n.target) for n in program.graph.nodes if n.op == "call_function"}

    def test_a_custom_operator_survives_the_rebuild(self) -> None:
        # Without partitioners nothing has cleared _check_ir_validity, so the
        # rebuild used to raise SpecViolationError on the backend's own kernel.
        session = self._export()
        self.assertTrue(
            any(op.startswith("op_backend_test.scale") for op in self._ops(session))
        )
        self.assertGreater(len(session.get_executorch_program_manager().buffer), 0)

    def test_it_also_survives_after_partitioning(self) -> None:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        session = self._export(partitioners=[XnnpackPartitioner()])
        ops = self._ops(session)
        self.assertIn("executorch_call_delegate", ops)
        self.assertTrue(any(op.startswith("op_backend_test.scale") for op in ops))
        self.assertGreater(len(session.get_executorch_program_manager().buffer), 0)


class TestTheRebuildCarriesManagerState(unittest.TestCase):
    """Rebuilding the manager drops anything not explicitly carried over."""

    def _session(self, *, constant_methods=None, etrecord=False, preserve=None):
        from executorch.exir import EdgeCompileConfig

        config = EdgeCompileConfig(preserve_ops=preserve) if preserve else None
        return ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            constant_methods=constant_methods,
            generate_etrecord=etrecord,
            export_recipe=ExportRecipe(
                name="carry",
                lowering_recipe=LoweringRecipe(
                    op_backends=[_RecordingOpBackend()], edge_compile_config=config
                ),
            ),
        )

    def test_constant_methods_survive(self) -> None:
        session = self._session(constant_methods={"get_max_seq_len": 128})
        session.export()
        manager = session.get_edge_program_manager()
        self.assertEqual(manager._config_methods, {"get_max_seq_len": 128})
        # Copied, so the rebuilt manager cannot write back into the one the
        # previous stage recorded.
        self.assertIsNot(
            manager._config_methods,
            session.get_stage_artifacts()[
                StageType.TO_EDGE_TRANSFORM_AND_LOWER
            ].data._config_methods,
        )

    def test_the_etrecord_survives(self) -> None:
        session = self._session(etrecord=True)
        session.export()
        self.assertIsNotNone(session.get_edge_program_manager()._etrecord)

    def test_the_compile_config_survives_except_for_validity(self) -> None:
        # The rebuild clears _check_ir_validity deliberately; everything else
        # the recipe asked for has to come through untouched.
        session = self._session(preserve=[torch.ops.aten.linear.default])
        session.export()
        config = session.get_edge_program_manager().compile_config
        self.assertEqual(config.preserve_ops, [torch.ops.aten.linear.default])
        self.assertFalse(config._check_ir_validity)


class TestEdgeManagerTransformPasses(unittest.TestCase):
    """The field predates op_backends and keeps working unchanged."""

    def _recipe(self, **kwargs):
        self.seen = []

        def collect(manager):
            self.seen.append(sorted(manager.methods))
            return []

        return ExportRecipe(
            name="passes",
            lowering_recipe=LoweringRecipe(
                edge_manager_transform_passes=[collect], **kwargs
            ),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        )

    def test_the_pipeline_shape_its_callers_use_still_works(self) -> None:
        # to_edge -> transform -> to_backend, which predates
        # TO_EDGE_TRANSFORM_AND_LOWER and is still the only way to reach this
        # field: the default pipeline does not insert the stage for it.
        session = ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=self._recipe(),
        )
        session.export()
        self.assertEqual(self.seen, [["forward"]])
        self.assertGreater(len(session.get_executorch_program_manager().buffer), 0)

    def test_it_does_not_get_the_stage_by_default(self) -> None:
        session = ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(
                name="passes",
                lowering_recipe=LoweringRecipe(
                    edge_manager_transform_passes=[lambda m: []]
                ),
            ),
        )
        self.assertNotIn(
            StageType.EDGE_PROGRAM_MANAGER_TRANSFORM, session._pipeline_stages
        )


class TestOpBackendsCompose(unittest.TestCase):
    """Several backends may share one program; the order they see it in is
    part of the contract, and each has to be handed what the last returned."""

    def test_backends_run_in_recipe_order(self) -> None:
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        order, handoff = [], {}

        class First(OpBackend):
            def lower(self, exported_program, method_name):
                order.append("first")
                handoff["out"] = _transform(exported_program, ExportPass())
                return handoff["out"]

        class Second(OpBackend):
            def lower(self, exported_program, method_name):
                order.append("second")
                handoff["in"] = exported_program
                return exported_program

        ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(
                name="composed",
                lowering_recipe=LoweringRecipe(op_backends=[First(), Second()]),
            ),
        ).export()

        self.assertEqual(order, ["first", "second"])
        self.assertIs(handoff["in"], handoff["out"])


class TestOpBackendsSeeDelegates(unittest.TestCase):
    """The premise of the stage: transforms run after the partitioner, so they
    see what was left outside the delegates."""

    def test_op_backend_observes_a_partitioned_program(self) -> None:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )
        from executorch.exir.pass_base import ExportPass
        from executorch.exir.program._program import _transform

        observed = {}

        class Observer(OpBackend):
            def lower(self, exported_program, method_name):
                observed["delegates"] = sum(
                    1
                    for n in exported_program.graph.nodes
                    if n.op == "call_function"
                    and n.target is torch.ops.higher_order.executorch_call_delegate
                )
                return _transform(exported_program, ExportPass())

        session = ExportSession(
            model=SimpleTestModel(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(
                name="delegated",
                lowering_recipe=LoweringRecipe(
                    partitioners=[XnnpackPartitioner()],
                    op_backends=[Observer()],
                ),
            ),
        )
        session.export()

        self.assertGreater(observed["delegates"], 0)
        self.assertGreater(len(session.get_executorch_program_manager().buffer), 0)

    def test_graph_module_input_gets_the_stage(self) -> None:
        backend = _RecordingOpBackend()
        exported = torch.export.export(SimpleTestModel(), (torch.randn(1, 10),))
        session = ExportSession(
            model=exported.module(),
            example_inputs=[(torch.randn(1, 10),)],
            export_recipe=ExportRecipe(
                name="gm", lowering_recipe=LoweringRecipe(op_backends=[backend])
            ),
        )
        self.assertNotIn(StageType.QUANTIZE, session._pipeline_stages)
        session.export()
        self.assertEqual(backend.seen, ["forward"])


class TestEdgeProgramManagerGetterArms(unittest.TestCase):
    """Each edge stage the getter recognises has to be the one it returns when
    that stage ran last."""

    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.inputs = [(torch.randn(1, 10),)]

    def _run(self, stages, **lowering):
        session = ExportSession(
            model=self.model,
            example_inputs=self.inputs,
            export_recipe=ExportRecipe(
                name="arms",
                lowering_recipe=LoweringRecipe(**lowering) if lowering else None,
                pipeline_stages=stages,
            ),
        )
        session.export()
        return session

    def test_to_backend_last(self) -> None:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        session = self._run(
            [
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
            partitioners=[XnnpackPartitioner()],
        )
        artifacts = session.get_stage_artifacts()
        # Without a partitioner these are the same object and the assertion
        # below would hold whichever arm the getter took.
        self.assertIsNot(
            artifacts[StageType.TO_EDGE].data, artifacts[StageType.TO_BACKEND].data
        )
        self.assertIs(
            session.get_edge_program_manager(),
            artifacts[StageType.TO_BACKEND].data,
        )

    def test_to_edge_last(self) -> None:
        session = self._run([StageType.TORCH_EXPORT, StageType.TO_EDGE])
        self.assertIs(
            session.get_edge_program_manager(),
            session.get_stage_artifacts()[StageType.TO_EDGE].data,
        )


class TestPipelinesThatWouldMisapplyTheRecipe(unittest.TestCase):
    def setUp(self) -> None:
        self.model = SimpleTestModel()
        self.inputs = [(torch.randn(1, 10),)]

    def _session(self, recipe) -> ExportSession:
        return ExportSession(
            model=self.model, example_inputs=self.inputs, export_recipe=recipe
        )

    def test_op_backends_may_follow_a_plain_to_edge(self) -> None:
        # Without partitioners there is nothing to run after, and this is the
        # pipeline shape the stage supported before op_backends existed.
        backend = _RecordingOpBackend()
        recipe = ExportRecipe(
            name="plain",
            lowering_recipe=LoweringRecipe(op_backends=[backend]),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_EXECUTORCH,
            ],
        )
        self._session(recipe).export()
        self.assertEqual(backend.seen, ["forward"])

    def test_op_backends_cannot_outrun_the_partitioners(self) -> None:
        # With partitioners set, lowering before they run would let the backend
        # claim operators a delegate was going to take.
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        recipe = ExportRecipe(
            name="pre",
            lowering_recipe=LoweringRecipe(
                partitioners=[XnnpackPartitioner()],
                op_backends=[_RecordingOpBackend()],
            ),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        )
        with self.assertRaisesRegex(ValueError, "before any partitioning stage"):
            self._session(recipe).export()

    def test_partitioners_cannot_run_twice(self) -> None:
        # TETAL -> EPMT -> TO_BACKEND only became legal with this stage's new
        # predecessors; before, the transition validator refused it and this
        # rule had nothing to catch.
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        recipe = ExportRecipe(
            name="twice",
            lowering_recipe=LoweringRecipe(
                partitioners=[XnnpackPartitioner()],
                op_backends=[_RecordingOpBackend()],
            ),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        )
        with self.assertRaisesRegex(ValueError, "partitioners would be applied"):
            self._session(recipe).export()

    def test_op_backends_cannot_run_twice(self) -> None:
        backend = _RecordingOpBackend()
        recipe = ExportRecipe(
            name="twice",
            lowering_recipe=LoweringRecipe(op_backends=[backend]),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_BACKEND,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_EXECUTORCH,
            ],
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            self._session(recipe).export()

    def test_a_truncated_pipeline_is_not_refused(self) -> None:
        # Stopping early to inspect an intermediate artifact is supported, so
        # a field the run never reaches is not an error.
        recipe = ExportRecipe(
            name="truncated",
            lowering_recipe=LoweringRecipe(op_backends=[_RecordingOpBackend()]),
            pipeline_stages=[StageType.TORCH_EXPORT],
        )
        self._session(recipe).export()

    def test_a_partitioning_stage_cannot_repeat(self) -> None:
        # TO_BACKEND accepts EDGE_PROGRAM_MANAGER_TRANSFORM as a predecessor,
        # so the new predecessor closes a loop back onto partitioning.
        from executorch.exir.backend.partitioner import Partitioner

        recipe = ExportRecipe(
            name="cycle",
            lowering_recipe=LoweringRecipe(
                partitioners=[Mock(spec=Partitioner)],
                op_backends=[_RecordingOpBackend()],
            ),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_BACKEND,
                StageType.TO_EXECUTORCH,
            ],
        )
        with self.assertRaisesRegex(ValueError, "more than once"):
            self._session(recipe).export()

    def test_op_backends_may_follow_a_separate_to_backend(self) -> None:
        # to_edge -> to_backend -> op backend is the staged shape a composed
        # NPU export needs; TO_BACKEND partitions, so it qualifies.
        backend = _RecordingOpBackend()
        recipe = ExportRecipe(
            name="staged",
            lowering_recipe=LoweringRecipe(op_backends=[backend]),
            pipeline_stages=[
                StageType.TORCH_EXPORT,
                StageType.TO_EDGE,
                StageType.TO_BACKEND,
                StageType.EDGE_PROGRAM_MANAGER_TRANSFORM,
                StageType.TO_EXECUTORCH,
            ],
        )
        self._session(recipe).export()
        self.assertEqual(backend.seen, ["forward"])
