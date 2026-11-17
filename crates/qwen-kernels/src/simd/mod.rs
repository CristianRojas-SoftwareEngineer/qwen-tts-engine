// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Backends SIMD y escalar de `qwen-kernels` seleccionados en runtime por `KernelDispatch`.

pub(crate) mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;
