# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.espressif.quantizer import EspQuantizer
from executorch.backends.test.harness import Tester
from executorch.backends.test.harness.stages import Quantize, StageType


class EspTester(Tester):
    def __init__(self, module, example_inputs, calibration_samples=None):
        stages = self.default_stage_classes()
        stages[StageType.QUANTIZE] = lambda: Quantize(
            EspQuantizer(), calibration_samples=calibration_samples
        )
        super().__init__(module, example_inputs, stages)
