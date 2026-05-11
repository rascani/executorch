# Copyright 2026 Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored with assistance from Claude (claude.ai/code).
#
# CMake toolchain file for cross-compiling the standalone MobileNetV2
# inference example for the Corstone-300 FVP (Cortex-M55 + Helium MVE).
#
# Assumes arm-none-eabi-gcc is on PATH; setup via
#   source examples/arm/arm-scratch/setup_path.sh
#
# Usage:
#   cmake -S examples/models/mv2_cortex_m_mve -B build/fvp \
#         -DCMAKE_TOOLCHAIN_FILE=examples/models/mv2_cortex_m_mve/fvp/toolchain-arm-none-eabi.cmake \
#         -DMV2_BUILD_FVP=ON -DMV2_GENERATED_DIR=...

set(CMAKE_SYSTEM_NAME       Generic)
set(CMAKE_SYSTEM_PROCESSOR  cortex-m55)

set(CMAKE_C_COMPILER        arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER      arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER      arm-none-eabi-gcc)
set(CMAKE_AR                arm-none-eabi-ar)
set(CMAKE_OBJCOPY           arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP           arm-none-eabi-objdump)
set(CMAKE_SIZE              arm-none-eabi-size)

# Skip linker-required tests during compile-id probe (no startup at that point).
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Architecture flags: Cortex-M55 with Helium (MVE) integer + FP, hard-float ABI.
set(MV2_ARCH_FLAGS
    "-mcpu=cortex-m55"
    "-mthumb"
    "-mfloat-abi=hard"
    "-mfpu=auto"
)

string(REPLACE ";" " " MV2_ARCH_FLAGS_STR "${MV2_ARCH_FLAGS}")
set(CMAKE_C_FLAGS_INIT      "${MV2_ARCH_FLAGS_STR}")
set(CMAKE_CXX_FLAGS_INIT    "${MV2_ARCH_FLAGS_STR}")
set(CMAKE_ASM_FLAGS_INIT    "${MV2_ARCH_FLAGS_STR}")

# rdimon.specs gives newlib + semihosting (printf, exit, etc.).
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "${MV2_ARCH_FLAGS_STR} --specs=rdimon.specs -Wl,--gc-sections"
)
