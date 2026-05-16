# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
#
# CMake toolchain file for cross-compiling the standalone KWT-1 (Keyword
# Spotting Transformer) inference example for the Corstone-300 FVP
# (Cortex-M55 + Helium MVE).  Mirrors the mv2_cortex_m_mve toolchain.
#
# Usage:
#   cmake -S examples/models/kwt_1_cortex_m_mve -B build/fvp \
#         -DCMAKE_TOOLCHAIN_FILE=examples/models/kwt_1_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
#         -DKWT_1_BUILD_FVP=ON -DKWT_1_GENERATED_DIR=...

set(CMAKE_SYSTEM_NAME       Generic)
set(CMAKE_SYSTEM_PROCESSOR  cortex-m55)

set(CMAKE_C_COMPILER        arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER      arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER      arm-none-eabi-gcc)
set(CMAKE_AR                arm-none-eabi-ar)
set(CMAKE_OBJCOPY           arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP           arm-none-eabi-objdump)
set(CMAKE_SIZE              arm-none-eabi-size)

set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set(KWT_1_ARCH_FLAGS
    "-mcpu=cortex-m55"
    "-mthumb"
    "-mfloat-abi=hard"
    "-mfpu=auto"
)

string(REPLACE ";" " " KWT_1_ARCH_FLAGS_STR "${KWT_1_ARCH_FLAGS}")
set(CMAKE_C_FLAGS_INIT      "${KWT_1_ARCH_FLAGS_STR}")
set(CMAKE_CXX_FLAGS_INIT    "${KWT_1_ARCH_FLAGS_STR}")
set(CMAKE_ASM_FLAGS_INIT    "${KWT_1_ARCH_FLAGS_STR}")

set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${KWT_1_ARCH_FLAGS_STR} --specs=rdimon.specs -Wl,--gc-sections"
)
