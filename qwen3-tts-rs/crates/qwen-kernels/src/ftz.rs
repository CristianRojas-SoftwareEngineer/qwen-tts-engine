/// Enables Flush-to-Zero (FTZ) and Denormals-Are-Zero (DAZ) on the current thread.
/// INT8 dequantized activations frequently enter the subnormal range (~100x penalty
/// without FTZ/DAZ). Must be called once on the main thread and on every pool worker.
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
