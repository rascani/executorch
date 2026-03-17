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
        let fb_prog = unsafe {
            flatbuffers::root_unchecked::<et_flatbuffer::executorch_flatbuffer::Program>(prog_data)
        };
        let plans = fb_prog.execution_plan().unwrap_or_else(|| {
            eprintln!("No execution plans in program");
            process::exit(1);
        });
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
    let fb_prog = unsafe {
        flatbuffers::root_unchecked::<et_flatbuffer::executorch_flatbuffer::Program>(prog_data)
    };
    let fb_plan = fb_prog.execution_plan().unwrap().get(plan);

    let mut method = unsafe {
        Method::init(&program, fb_plan, &mut memory).unwrap_or_else(|e| {
            eprintln!("Failed to init method: {}", e);
            process::exit(1);
        })
    };

    let n_inputs = method.num_inputs();
    let n_outputs = method.num_outputs();
    eprintln!("Method loaded: {} inputs, {} outputs", n_inputs, n_outputs);

    // Build inputs from metadata: fill each tensor with sequential floats.
    struct InputStorage {
        data: Vec<f32>,
        sizes: Vec<i32>,
        dim_order: Vec<u8>,
        strides: Vec<i32>,
    }

    let mut storages: Vec<InputStorage> = Vec::with_capacity(n_inputs);
    for i in 0..n_inputs {
        let ti = meta.input_tensor_meta(i).unwrap_or_else(|e| {
            eprintln!("Failed to get input tensor meta {}: {}", i, e);
            process::exit(1);
        });
        let sizes: Vec<i32> = ti.sizes().to_vec();
        let numel: usize = sizes.iter().map(|&s| s as usize).product();
        let ndim = sizes.len();

        let data: Vec<f32> = (0..numel).map(|j| (j + 1) as f32).collect();

        // Row-major dim order and strides
        let dim_order: Vec<u8> = (0..ndim as u8).collect();
        let mut strides = vec![1i32; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * sizes[d + 1];
        }

        eprintln!(
            "  input[{}]: {:?} {:?} ({} elements)",
            i, ti.scalar_type, sizes, numel
        );
        storages.push(InputStorage {
            data,
            sizes,
            dim_order,
            strides,
        });
    }

    // Create TensorImpl and EValue for each input (borrows from storages).
    let mut tensor_impls: Vec<TensorImpl> = storages
        .iter_mut()
        .map(|s| {
            TensorImpl::new(
                ScalarType::Float,
                s.sizes.len() as isize,
                s.sizes.as_mut_ptr(),
                s.data.as_mut_ptr() as *mut u8,
                s.dim_order.as_mut_ptr(),
                s.strides.as_mut_ptr(),
                TensorShapeDynamism::Static,
            )
        })
        .collect();

    let evalues: Vec<EValue> = tensor_impls
        .iter_mut()
        .map(|ti| EValue::from_tensor(ti as *mut TensorImpl))
        .collect();

    for (i, ev) in evalues.iter().enumerate() {
        method.set_input(i, ev).unwrap_or_else(|e| {
            eprintln!("Failed to set input {}: {}", i, e);
            process::exit(1);
        });
    }

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
