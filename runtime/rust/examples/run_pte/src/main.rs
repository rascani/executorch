// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

use std::env;
use std::fs;
use std::process;

use et_core::allocator::{BumpAllocator, HierarchicalAllocator, MemoryManager};
use et_core::evalue::EValue;
use et_core::scalar_type::ScalarType;
use et_core::tensor::{TensorImpl, TensorShapeDynamism};
use et_runtime::data_loader::BufferDataLoader;
use et_runtime::method::Method;
use et_runtime::program::Program;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <model.pte>", args[0]);
        process::exit(1);
    }

    let pte_path = &args[1];
    eprintln!("Loading PTE: {}", pte_path);

    let pte_data = fs::read(pte_path).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", pte_path, e);
        process::exit(1);
    });

    let loader = BufferDataLoader::new(&pte_data);
    let program = Program::load(loader).unwrap_or_else(|e| {
        eprintln!("Failed to load program: {}", e);
        process::exit(1);
    });

    let num_methods = program.num_methods().unwrap_or(0);
    eprintln!("Program loaded: {} method(s)", num_methods);

    for i in 0..num_methods {
        match program.get_method_name(i) {
            Ok(name) => eprintln!("  method[{}]: {}", i, name),
            Err(e) => eprintln!("  method[{}]: <error: {}>", i, e),
        }
    }

    let method_name = "forward";
    let meta = program.method_meta(method_name).unwrap_or_else(|e| {
        eprintln!("Failed to get method meta for '{}': {}", method_name, e);
        process::exit(1);
    });

    eprintln!(
        "Method '{}': {} inputs, {} outputs",
        method_name,
        meta.num_inputs(),
        meta.num_outputs(),
    );

    // Allocate memory for the method
    let num_planned_buffers = meta.num_memory_planned_buffers();
    eprintln!("Planned buffers: {}", num_planned_buffers);

    // Allocate planned memory buffers
    // We need to keep the Vec<Vec<u8>> alive, then create &mut [&mut [u8]] from it.
    let mut planned_bufs: Vec<Vec<u8>> = Vec::with_capacity(num_planned_buffers);
    for i in 0..num_planned_buffers {
        let size = meta.memory_planned_buffer_size(i).unwrap_or(0);
        eprintln!("  buffer[{}]: {} bytes", i, size);
        planned_bufs.push(vec![0u8; size]);
    }

    // Build the &mut [&mut [u8]] for HierarchicalAllocator.
    // This requires some careful borrow juggling: we need a Vec<&mut [u8]>
    // pointing into the Vec<Vec<u8>>.
    let mut planned_slices: Vec<&mut [u8]> = planned_bufs
        .iter_mut()
        .map(|v| v.as_mut_slice())
        .collect();

    let mut hierarchical = HierarchicalAllocator::new(&mut planned_slices);

    // Method allocator — 4 MB should be plenty for most models
    let mut method_alloc_buf = vec![0u8; 4 * 1024 * 1024];
    let mut method_allocator = BumpAllocator::new(&mut method_alloc_buf);

    // Temp allocator — 1 MB
    let mut temp_alloc_buf = vec![0u8; 1024 * 1024];
    let mut temp_allocator = BumpAllocator::new(&mut temp_alloc_buf);

    let mut memory = MemoryManager::new(
        &mut method_allocator,
        Some(&mut hierarchical),
        Some(&mut temp_allocator),
    );

    // Load the method — this resolves all kernels and builds the execution plan
    eprintln!("Loading method '{}'...", method_name);

    // We need to re-parse the flatbuffer plan to pass to Method::init.
    // This is safe because Program keeps the data alive.
    let plan = {
        let prog_data = program.program_data();
        let fb_prog =
            flatbuffers::root::<et_flatbuffer::executorch_flatbuffer::Program>(prog_data)
                .unwrap_or_else(|e| {
                    eprintln!("Failed to parse flatbuffer: {:?}", e);
                    process::exit(1);
                });
        let plans = fb_prog.execution_plan().unwrap_or_else(|| {
            eprintln!("No execution plans in program");
            process::exit(1);
        });
        // Find the plan matching method_name
        let mut found = None;
        for i in 0..plans.len() {
            if plans.get(i).name() == Some(method_name) {
                found = Some(i);
                break;
            }
        }
        found.unwrap_or_else(|| {
            eprintln!("Method '{}' not found in program", method_name);
            process::exit(1);
        })
    };

    // Re-parse to get the plan (the previous borrow was dropped)
    let prog_data = program.program_data();
    let fb_prog =
        flatbuffers::root::<et_flatbuffer::executorch_flatbuffer::Program>(prog_data).unwrap();
    let fb_plan = fb_prog.execution_plan().unwrap().get(plan);

    let mut method = unsafe {
        Method::init(&program, fb_plan, &mut memory).unwrap_or_else(|e| {
            eprintln!("Failed to init method: {}", e);
            process::exit(1);
        })
    };

    eprintln!(
        "Method loaded: {} inputs, {} outputs",
        method.num_inputs(),
        method.num_outputs()
    );

    // For a simple add model (forward(x, y) = x + y), set up float tensor inputs.
    // This assumes the model expects two 2x2 float tensors.
    let mut sizes = [2i32, 2];
    let mut dim_order = [0u8, 1];
    let mut strides = [2i32, 1];

    let mut input_data_a = [1.0f32, 2.0, 3.0, 4.0];
    let mut input_data_b = [10.0f32, 20.0, 30.0, 40.0];

    let mut impl_a = TensorImpl::new(
        ScalarType::Float,
        2,
        sizes.as_mut_ptr(),
        input_data_a.as_mut_ptr() as *mut u8,
        dim_order.as_mut_ptr(),
        strides.as_mut_ptr(),
        TensorShapeDynamism::Static,
    );
    let mut impl_b = TensorImpl::new(
        ScalarType::Float,
        2,
        sizes.as_mut_ptr(),
        input_data_b.as_mut_ptr() as *mut u8,
        dim_order.as_mut_ptr(),
        strides.as_mut_ptr(),
        TensorShapeDynamism::Static,
    );

    let ev_a = EValue::from_tensor(&mut impl_a as *mut TensorImpl);
    let ev_b = EValue::from_tensor(&mut impl_b as *mut TensorImpl);

    method.set_input(0, &ev_a).unwrap_or_else(|e| {
        eprintln!("Failed to set input 0: {}", e);
        process::exit(1);
    });
    method.set_input(1, &ev_b).unwrap_or_else(|e| {
        eprintln!("Failed to set input 1: {}", e);
        process::exit(1);
    });

    eprintln!("Executing...");
    method.execute().unwrap_or_else(|e| {
        eprintln!("Execution failed: {}", e);
        process::exit(1);
    });

    eprintln!("Execution complete. Reading output...");
    let output = method.get_output(0).unwrap_or_else(|e| {
        eprintln!("Failed to get output: {}", e);
        process::exit(1);
    });

    if output.is_tensor() {
        let tensor = output.to_tensor().unwrap();
        let numel = tensor.numel() as usize;
        eprintln!("Output tensor: {:?}, numel={}", tensor.sizes(), numel);

        if tensor.scalar_type() == ScalarType::Float {
            let data_ptr = tensor.data_ptr::<f32>().unwrap();
            eprint!("  data: [");
            for i in 0..numel {
                let val = unsafe { *data_ptr.add(i) };
                if i > 0 {
                    eprint!(", ");
                }
                eprint!("{}", val);
            }
            eprintln!("]");
        }
    } else if output.is_int() {
        eprintln!("Output int: {}", output.to_int().unwrap());
    } else if output.is_double() {
        eprintln!("Output double: {}", output.to_double().unwrap());
    } else {
        eprintln!("Output tag: {:?}", output.tag);
    }

    eprintln!("Done.");
}

extern crate flatbuffers;
