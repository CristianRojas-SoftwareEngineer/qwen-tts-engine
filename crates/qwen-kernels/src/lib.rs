// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Primitivas numéricas en CPU, buffers alineados a caché, pool de hilos,
//! cuantización y despacho SIMD/BLAS en runtime para `qwen-tts-engine`.

pub mod aligned;
pub mod blas;
pub mod caps;
pub mod dispatch;
pub mod ftz;
pub mod pool;
pub mod quant;
pub mod selftest;
mod simd;

pub use aligned::AlignedBuf;
pub use blas::{set_initial_openblas_threads, FfiBlas, Transpose};
pub use caps::CpuCaps;
pub use dispatch::{ComputeRuntime, KernelChoice, KernelDispatch};
pub use ftz::ftz_on;
pub use pool::{default_num_threads, resolve_num_threads, spin_budget_for, ComputePool};
pub use quant::Q4_0Block;
