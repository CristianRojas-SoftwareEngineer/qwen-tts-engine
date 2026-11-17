// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Write;

#[derive(Debug, Clone, Copy, Default)]
pub struct CpuCaps {
    pub arch: &'static str,
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub neon: bool,
    pub dotprod: bool,
}

impl CpuCaps {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                arch: "x86-64",
                sse2: std::arch::is_x86_feature_detected!("sse2"),
                avx: std::arch::is_x86_feature_detected!("avx"),
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
                avx512bf16: std::arch::is_x86_feature_detected!("avx512bf16"),
                neon: false,
                dotprod: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                arch: "arm64",
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
                ..Default::default()
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                arch: "(other)",
                ..Default::default()
            }
        }
    }

    #[inline]
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    pub fn format_report(&self, threads: usize, blas_name: &str) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "qwen-tts-engine runtime capabilities:");
        let _ = writeln!(s, "  arch:             {}", self.arch);
        if self.has_avx2_fma() {
            let _ = writeln!(s, "  matvec + attn:    AVX2 (2-row fused, FMA)");
            let _ = writeln!(
                s,
                "  int8 dot:         AVX2 _mm256_maddubs_epi16 (integer, runtime dispatch)"
            );
            let _ = writeln!(s, "  bf16 dot:         bf16->f32 conversion + FMA");
            let _ = writeln!(s, "  rms/bf16-conv:    AVX2");
        } else {
            let _ = writeln!(s, "  matvec + attn:    scalar");
            let _ = writeln!(
                s,
                "  int8 dot:         scalar (integer loop, int32 accumulation)"
            );
            let _ = writeln!(s, "  rms/bf16-conv:    scalar");
        }

        let pool_kind = if cfg!(target_os = "windows") {
            "Win32/Rust ComputePool"
        } else {
            "Rust ComputePool"
        };
        let _ = writeln!(s, "  matvec threads:   {} ({} threads)", pool_kind, threads);
        let _ = writeln!(s, "  BLAS (prefill):   {}", blas_name);

        if self.arch == "x86-64" {
            let mut feats = Vec::new();
            if self.sse2 { feats.push("sse2"); }
            if self.avx { feats.push("avx"); }
            if self.avx2 { feats.push("avx2"); }
            if self.fma { feats.push("fma"); }
            if self.avx512f { feats.push("avx512f"); }
            if self.avx512bw { feats.push("avx512bw"); }
            if self.avx512vnni { feats.push("avx512vnni"); }
            if self.avx512bf16 { feats.push("avx512bf16"); }
            let _ = writeln!(s, "  runtime cpu:      {}", feats.join(" "));
            let _ = writeln!(
                s,
                "  lever (x86):      {}",
                if self.has_avx2_fma() {
                    "AVX2 maddubs integer dot for int8 and q4_0"
                } else {
                    "scalar integer dot"
                }
            );
        } else if self.arch == "arm64" {
            let _ = writeln!(
                s,
                "  runtime cpu:      NEON{}",
                if self.dotprod { " dotprod/SDOT" } else { "" }
            );
        }
        s
    }
}
