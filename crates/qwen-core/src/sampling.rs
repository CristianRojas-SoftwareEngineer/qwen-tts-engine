// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#[derive(Debug, Clone)]
pub struct LcgRng {
    pub seed: u32,
}

impl LcgRng {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    /// Deterministic LCG uniform sample in `[0.0, 1.0)`.
    #[inline]
    pub fn next_uniform(&mut self) -> f32 {
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
        let mut max_val = logits[0];
        for &v in &logits[1..] {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum = 0.0f32;
        let inv_temp = 1.0f32 / temp;
        for v in logits.iter_mut() {
            *v = ((*v - max_val) * inv_temp).exp();
            sum += *v;
        }
        for v in logits.iter_mut() {
            *v /= sum;
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
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = i;
        }
        let mut cumsum = 0.0f32;
        let mut cutoff = n;
        for i in 0..n {
            let mut max_idx = i;
            for j in (i + 1)..n {
                if logits[idx[j]] > logits[idx[max_idx]] {
                    max_idx = j;
                }
            }
            idx.swap(i, max_idx);
            cumsum += logits[idx[i]];
            if cumsum > p {
                cutoff = i + 1;
                break;
            }
        }
        for &k in &idx[cutoff..n] {
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
            let mut best_v = logits[0];
            for (i, &v) in logits.iter().enumerate().skip(1) {
                if v > best_v {
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
