# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.espressif.passes.channels_last_pass import EspChannelsLastPass
from executorch.exir.pass_manager import ExportedProgramPassManager


class EspPassManager(ExportedProgramPassManager):
    """Run Espressif transformations on an edge ExportedProgram."""

    def __init__(self) -> None:
        super().__init__(passes=[EspChannelsLastPass()])
