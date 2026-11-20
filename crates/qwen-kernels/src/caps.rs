// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fmt::Write;

use crate::dispatch::KernelChoice;

/// Capacidades SIMD y de arquitectura detectadas en tiempo de ejecución.
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
                arch: "(otra)",
                ..Default::default()
            }
        }
    }

    #[inline]
    pub fn has_avx2_fma(&self) -> bool {
        self.avx2 && self.fma
    }

    /// Informe de `--caps`. `kernels` es el nombre de la ruta en uso (`KernelDispatch::name`): la
    /// escalar si `QWEN_KERNELS=scalar` la fuerza.
    pub fn format_report(
        &self,
        threads: usize,
        blas_name: &str,
        choice: KernelChoice,
        kernels: &str,
    ) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "Capacidades de qwen-tts-engine en tiempo de ejecución:");
        let _ = writeln!(s, "  arquitectura:     {}", self.arch);
        if choice == KernelChoice::Scalar {
            let _ = writeln!(s, "  kernels:          scalar (forzado por QWEN_KERNELS=scalar)");
        } else {
            let _ = writeln!(s, "  kernels:          {kernels}");
        }
        match kernels {
            "avx2" => {
                let _ = writeln!(s, "  matvec + attn:    AVX2 (2 filas fusionadas, FMA)");
                let _ = writeln!(
                    s,
                    "  int8 dot:         AVX2 _mm256_maddubs_epi16 (enteros, despacho en tiempo de ejecución)"
                );
                let _ = writeln!(s, "  bf16 dot:         conversión bf16->f32 + FMA");
                let _ = writeln!(s, "  rms/bf16-conv:    AVX2");
            }
            "neon" | "neon-dotprod" => {
                let dot = if kernels == "neon-dotprod" {
                    "NEON SDOT (dotprod)"
                } else {
                    "NEON SMULL widening"
                };
                let _ = writeln!(s, "  q4_0 + int8 dot:  {dot}, orden de suma de AVX2");
                let _ = writeln!(s, "  bf16 + attn:      NEON (FMA, orden de suma de AVX2)");
                let _ = writeln!(s, "  rms/bf16-conv:    NEON");
                let _ = writeln!(s, "  snake (decoder):  sin polinómico en NEON");
            }
            _ => {
                let _ = writeln!(s, "  matvec + attn:    scalar");
                let _ = writeln!(
                    s,
                    "  int8 dot:         scalar (bucle entero, acumulación en int32)"
                );
                let _ = writeln!(s, "  rms/bf16-conv:    scalar");
            }
        }

        let pool_kind = if cfg!(target_os = "windows") {
            "Win32/Rust ComputePool"
        } else {
            "Rust ComputePool"
        };
        let _ = writeln!(s, "  hilos de matvec:  {} ({} hilos)", pool_kind, threads);
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
                "  palanca (x86):    {}",
                if kernels == "avx2" {
                    "dot entero AVX2 maddubs para int8 y q4_0"
                } else {
                    "dot entero scalar"
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
