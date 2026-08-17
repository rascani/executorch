# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.transforms.channels_last_ops  # noqa: F401

from executorch.exir.dialects._ops import ops as exir_ops

LAYOUT_PERMUTE_COPY = exir_ops.edge.channels_last.permute_copy.default
