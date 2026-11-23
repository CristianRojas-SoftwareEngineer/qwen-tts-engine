// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::{DEFAULT_SEED, DEFAULT_TEMPERATURE};

/// Parámetros de muestreo de una petición. Única fuente de los valores por defecto.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub seed: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: DEFAULT_TEMPERATURE,
            top_k: 50,
            top_p: 1.0,
            rep_penalty: 1.05,
            seed: DEFAULT_SEED,
        }
    }
}

/// Origen de la semilla pedida por el usuario.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedChoice {
    /// El usuario no fijó semilla: se usa `DEFAULT_SEED`.
    Default,
    /// Semilla explícita.
    Fixed(u32),
    /// Semilla temporal (reloj del sistema).
    Clock,
}

/// Semilla derivada del reloj del sistema.
pub fn time_based_seed() -> u32 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    ((d.as_secs() ^ (d.subsec_micros() as u64)) & 0xFFFF_FFFF) as u32
}

#[derive(Debug, Clone)]
pub struct LcgRng {
    pub seed: u32,
}

impl LcgRng {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Muestra uniforme determinista mediante LCG en `[0.0, 1.0)`.
    #[inline]
    fn next_uniform(&mut self) -> f32 {
        self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.seed >> 16) & 0x7FFF) as f32 / 32768.0f32
    }
}

pub struct Sampler {
    topk_tmp: Vec<f32>,
    topp_idx: Vec<usize>,
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler {
    pub fn new() -> Self {
        Self {
            topk_tmp: Vec::new(),
            topp_idx: Vec::new(),
        }
    }

    fn ensure_buffers(&mut self, n: usize) {
        if self.topk_tmp.len() < n {
            self.topk_tmp.resize(n, 0.0);
            self.topp_idx.resize(n, 0);
        }
    }

    fn softmax(logits: &mut [f32], temp: f32) {
        let mut max_val = f32::NEG_INFINITY;
        for &v in logits.iter() {
            if v.is_finite() && v > max_val {
                max_val = v;
            }
        }
        if !max_val.is_finite() {
            let uniform = 1.0f32 / (logits.len() as f32);
            logits.fill(uniform);
            return;
        }
        let mut sum = 0.0f32;
        let inv_temp = 1.0f32 / temp;
        for v in logits.iter_mut() {
            let p = if v.is_finite() {
                ((*v - max_val) * inv_temp).exp()
            } else if *v > 0.0 {
                1.0f32
            } else {
                0.0f32
            };
            *v = p;
            sum += p;
        }
        if sum > 0.0 && sum.is_finite() {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        } else {
            let uniform = 1.0f32 / (logits.len() as f32);
            logits.fill(uniform);
        }
    }

    fn quickselect_kth_largest(arr: &mut [f32], k: usize) -> f32 {
        let mut lo = 0isize;
        let mut hi = (arr.len() as isize) - 1;
        let target = (k as isize) - 1;
        while lo < hi {
            let pivot = arr[(lo + (hi - lo) / 2) as usize];
            let mut i = lo;
            let mut j = hi;
            let mut p = lo;
            while i <= j {
                if arr[i as usize] > pivot {
                    arr.swap(p as usize, i as usize);
                    p += 1;
                    i += 1;
                } else if arr[i as usize] < pivot {
                    arr.swap(i as usize, j as usize);
                    j -= 1;
                } else {
                    i += 1;
                }
            }
            if target < p {
                hi = p - 1;
            } else if target > j {
                lo = j + 1;
            } else {
                return pivot;
            }
        }
        arr[lo as usize]
    }

    fn topk_filter(&mut self, logits: &mut [f32], k: usize) {
        let n = logits.len();
        if k == 0 || k >= n {
            return;
        }
        self.topk_tmp[..n].copy_from_slice(logits);
        let threshold = Self::quickselect_kth_largest(&mut self.topk_tmp[..n], k);
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = 0.0;
            }
        }
    }

    fn topp_filter(&mut self, logits: &mut [f32], p: f32) {
        let n = logits.len();
        if p >= 1.0 {
            return;
        }
        let idx = &mut self.topp_idx[..n];
        let mut nz = 0usize;
        for (i, &val) in logits.iter().enumerate() {
            if val > 0.0 {
                idx[nz] = i;
                nz += 1;
            }
        }
        idx[..nz].sort_unstable_by(|&a, &b| {
            logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b))
        });
        let mut cumsum = 0.0f32;
        let mut cutoff = nz;
        for (i, &k) in idx[..nz].iter().enumerate() {
            cumsum += logits[k];
            if cumsum > p {
                cutoff = i + 1;
                break;
            }
        }
        for &k in &idx[cutoff..nz] {
            logits[k] = 0.0;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &mut self,
        logits: &mut [f32],
        temp: f32,
        top_k: i32,
        top_p: f32,
        rep_penalty: f32,
        prev_tokens: &[i32],
        rng: &mut LcgRng,
    ) -> usize {
        let vocab_size = logits.len();
        if vocab_size == 0 {
            return 0;
        }
        self.ensure_buffers(vocab_size);

        if rep_penalty != 1.0 && !prev_tokens.is_empty() {
            for &tok in prev_tokens {
                if tok >= 0 && (tok as usize) < vocab_size {
                    let v = &mut logits[tok as usize];
                    if *v > 0.0 {
                        *v /= rep_penalty;
                    } else {
                        *v *= rep_penalty;
                    }
                }
            }
        }

        if temp < 1e-6 {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if !v.is_nan() && v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best;
        }

        Self::softmax(logits, temp);

        if top_k > 0 && (top_k as usize) < vocab_size {
            self.topk_filter(logits, top_k as usize);
        }

        if top_p > 0.0 && top_p < 1.0 {
            self.topp_filter(logits, top_p);
        }

        let sum: f32 = logits.iter().sum();
        if sum > 0.0 {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        }

        let r = rng.next_uniform();
        let mut cumsum = 0.0f32;
        for (i, &p) in logits.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        vocab_size - 1
    }
}
