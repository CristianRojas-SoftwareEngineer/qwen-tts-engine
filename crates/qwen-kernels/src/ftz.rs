// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

/// Activa Flush-to-Zero (FTZ) y Denormals-Are-Zero (DAZ) en el hilo actual.
/// Las activaciones INT8 descuantizadas entran con frecuencia en rango subnormal
/// (~100x de penalización sin FTZ/DAZ). Debe invocarse una vez en el hilo principal
/// y en cada trabajador de `ComputePool`.
#[inline]
pub fn ftz_on() {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut mxcsr: u32 = 0;
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        if (mxcsr & 0x8040) != 0x8040 {
            mxcsr |= 0x8040;
            core::arch::asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack, preserves_flags));
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        let mut fpcr: u64;
        core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nomem, nostack, preserves_flags));
        if (fpcr & (1 << 24)) == 0 {
            fpcr |= 1 << 24;
            core::arch::asm!("msr fpcr, {}", in(reg) fpcr, options(nomem, nostack, preserves_flags));
        }
    }
}
