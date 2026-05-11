/*
 * Copyright 2026 Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Authored with assistance from Claude (claude.ai/code).
 *
 * Linker script for the standalone MobileNetV2 inference example running on
 * the Corstone-300 FVP.  Adapted from
 *   examples/arm/arm-scratch/ethos-u/core_platform/targets/corstone-300/platform.ld
 * with three changes for the MV2 use case:
 *   - .rodata stays in DDR (it can reach 600+ KB for MV2 weights and would
 *     overflow DTCM if copied there).
 *   - .text in ITCM (fits easily; fast execution).
 *   - .bss.tensor_arena unconditionally placed in DDR.
 */

__STACK_SIZE = 0x00010000;
__HEAP_SIZE  = 0x00010000;

MEMORY
{
  ITCM  (rx)  : ORIGIN = 0x10000000, LENGTH = 0x00080000
  BRAM  (rw)  : ORIGIN = 0x11000000, LENGTH = 0x00100000
  DTCM  (rw)  : ORIGIN = 0x30000000, LENGTH = 0x00080000
  SRAM  (rw)  : ORIGIN = 0x31000000, LENGTH = 0x00200000
  DDR   (rwx) : ORIGIN = 0x70000000, LENGTH = 0x10000000
}

ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    _vectors = .;
    KEEP(*(.vectors))
    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    KEEP(*(.eh_frame*))
  } > ITCM

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > ITCM

  __exidx_start = .;
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > ITCM
  __exidx_end = .;

  .copy.table :
  {
    . = ALIGN(4);
    __copy_table_start__ = .;
    LONG (LOADADDR(.data))
    LONG (ADDR(.data))
    LONG (SIZEOF(.data) / 4)
    __copy_table_end__ = .;
  } > ITCM

  .zero.table :
  {
    . = ALIGN(4);
    __zero_table_start__ = .;
    __zero_table_end__ = .;
  } > ITCM

  PROVIDE(__etext = LOADADDR(.data));

  .rodata :
  {
    __rodata_start__ = .;
    *(.rodata)
    *(.rodata.*)
    . = ALIGN(4);
    __rodata_end__ = .;
  } > DDR

  .data : ALIGN(4)
  {
    __data_start__ = .;
    *(vtable)
    *(.data)
    *(.data.*)

    . = ALIGN(4);
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);

    . = ALIGN(4);
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);

    KEEP(*(.jcr*))
    . = ALIGN(4);
    __data_end__ = .;
  } > DTCM AT > ITCM

  .arena_ddr (NOLOAD) :
  {
    . = ALIGN(16);
    *(.bss.tensor_arena)
  } > DDR

  .bss (NOLOAD) :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss)
    *(.bss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > DTCM

  .heap (COPY) :
  {
    . = ALIGN(8);
    __end__ = .;
    PROVIDE(end = .);
    . = . + __HEAP_SIZE;
    . = ALIGN(8);
    __HeapLimit = .;
  } > DTCM

  .stack (ORIGIN(DTCM) + LENGTH(DTCM) - __STACK_SIZE) (COPY) :
  {
    . = ALIGN(8);
    __StackLimit = .;
    . = . + __STACK_SIZE;
    . = ALIGN(8);
    __StackTop = .;
  } > DTCM
  PROVIDE(__stack = __StackTop);
  PROVIDE(__INITIAL_SP = __StackTop);
  PROVIDE(__STACK_LIMIT = __StackLimit);

  ASSERT(__StackLimit >= __HeapLimit, "DTCM overflowed: stack/heap don't fit")
}
