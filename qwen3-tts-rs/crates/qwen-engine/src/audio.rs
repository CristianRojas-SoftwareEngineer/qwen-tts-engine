use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::EngineError;

#[inline]
pub fn f32_to_s16le_clamped(mut s: f32) -> i16 {
    if s.is_nan() {
        s = 0.0;
    }
    if s < -1.0 {
        s = -1.0;
    }
    if s > 1.0 {
        s = 1.0;
    }
    (s * 32767.0f32) as i16
}

pub fn apply_gain(samples: &mut [f32], gain: f32) {
    if (gain - 1.0).abs() < f32::EPSILON {
        return;
    }
    for s in samples.iter_mut() {
        let mut v = *s * gain;
        if v < -1.0 {
            v = -1.0;
        }
        if v > 1.0 {
            v = 1.0;
        }
        *s = v;
    }
}

/// Pitch-preserving time-stretch via WSOLA (`--rate`), matching `qwen_tts_audio.c`.
pub fn time_stretch(input: &[f32], rate: f32) -> Vec<f32> {
    let rate = if rate <= 0.0 { 1.0 } else { rate };
    let n_in = input.len();
    if (rate - 1.0).abs() < f32::EPSILON || n_in == 0 {
        return input.to_vec();
    }
    let n = 1024usize.min(n_in);
    if n < 16 {
        return input.to_vec();
    }
    let hs = n / 2;
    let tol = n / 4;
    let mut win = vec![0.0f32; n];
    for i in 0..n {
        win[i] = 0.5f32
            * (1.0f32 - (2.0f32 * std::f32::consts::PI * i as f32 / (n - 1) as f32).cos());
    }
    let cap = ((n_in as f64) / (rate as f64)) as usize + 2 * n + 16;
    let mut o = vec![0.0f32; cap];
    let mut wsum = vec![0.0f32; cap];

    let ha = (hs as f64) * (rate as f64);
    let mut xs = 0usize;
    let mut oy = 0usize;
    let mut nominal = 0.0f64;

    while xs + n <= n_in && oy + n <= cap {
        for i in 0..n {
            o[oy + i] += input[xs + i] * win[i];
            wsum[oy + i] += win[i];
        }
        let tstart = xs + hs;
        nominal += ha;
        let center = (nominal + 0.5) as isize;
        let next: isize = if tstart + n > n_in {
            center
        } else {
            let mut best_d = 0isize;
            let mut best = -1e300f64;
            for d in -(tol as isize)..=(tol as isize) {
                let s = center + d;
                if s < 0 || (s as usize) + n > n_in {
                    continue;
                }
                let su = s as usize;
                let mut num = 0.0f64;
                let mut en = 0.0f64;
                for i in 0..n {
                    let a = input[su + i] as f64;
                    num += a * (input[tstart + i] as f64);
                    en += a * a;
                }
                let score = num / (en.sqrt() + 1e-9);
                if score > best {
                    best = score;
                    best_d = d;
                }
            }
            center + best_d
        };
        let next = next.max(0) as usize;
        if next + n > n_in {
            break;
        }
        xs = next;
        oy += hs;
    }

    let total = (oy + n).min(cap);
    for i in 0..total {
        if wsum[i] > 1e-6 {
            o[i] /= wsum[i];
        }
    }
    o.truncate(total);
    o
}

pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let n_samples = samples.len();
    let data_size = (n_samples * 2) as u32;
    let file_size = 36u32 + data_size;
    let byte_rate = sample_rate * 2;
    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&file_size.to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    let mut fade_in = (sample_rate as usize) / 200; // 5 ms
    let mut fade_out = (sample_rate as usize) * 40 / 1000; // 40 ms
    if fade_in > n_samples / 2 {
        fade_in = n_samples / 2;
    }
    if fade_out > n_samples / 2 {
        fade_out = n_samples / 2;
    }
    for (i, &orig) in samples.iter().enumerate() {
        let mut s = orig;
        if fade_in > 0 && i < fade_in {
            s *= (i as f32) / (fade_in as f32);
        }
        if fade_out > 0 && i >= n_samples - fade_out {
            s *= ((n_samples - 1 - i) as f32) / (fade_out as f32);
        }
        out.extend_from_slice(&f32_to_s16le_clamped(s).to_le_bytes());
    }
    out
}

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), EngineError> {
    let bytes = encode_wav_bytes(samples, sample_rate);
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&bytes)?;
    w.flush()?;
    Ok(())
}

pub fn write_wav_header<W: Write>(w: &mut W, sample_rate: u32, data_bytes: u32) -> Result<(), EngineError> {
    let file_size = 36u32.saturating_add(data_bytes);
    let byte_rate = sample_rate * 2;
    w.write_all(b"RIFF")?;
    w.write_all(&file_size.to_le_bytes())?;
    w.write_all(b"WAVEfmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&2u16.to_le_bytes())?;
    w.write_all(&16u16.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    Ok(())
}

pub trait AudioSink: Send {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError>;
    fn finish(&mut self) -> Result<(), EngineError>;
}

pub struct MemorySink {
    pub samples: Vec<f32>,
}

impl MemorySink {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
        }
    }
}

impl AudioSink for MemorySink {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        self.samples.extend_from_slice(samples);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EngineError> {
        Ok(())
    }
}

pub struct WavStreamSink {
    file: BufWriter<File>,
    sample_rate: u32,
    volume: f32,
    data_bytes: u32,
}

impl WavStreamSink {
    pub fn create(path: &Path, sample_rate: u32, volume: f32) -> Result<Self, EngineError> {
        let f = File::create(path)?;
        let mut bw = BufWriter::new(f);
        write_wav_header(&mut bw, sample_rate, 0)?;
        bw.flush()?;
        Ok(Self {
            file: bw,
            sample_rate,
            volume,
            data_bytes: 0,
        })
    }
}

impl AudioSink for WavStreamSink {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = s * self.volume;
            buf.extend_from_slice(&f32_to_s16le_clamped(v).to_le_bytes());
        }
        self.file.write_all(&buf)?;
        self.file.flush()?;
        self.data_bytes = self.data_bytes.saturating_add(buf.len() as u32);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EngineError> {
        self.file.flush()?;
        let inner = self.file.get_mut();
        let file_size = 36u32.saturating_add(self.data_bytes);
        inner.seek(SeekFrom::Start(4))?;
        inner.write_all(&file_size.to_le_bytes())?;
        inner.seek(SeekFrom::Start(40))?;
        inner.write_all(&self.data_bytes.to_le_bytes())?;
        inner.flush()?;
        let _ = self.sample_rate;
        Ok(())
    }
}

pub struct StdoutRawSink {
    volume: f32,
}

impl StdoutRawSink {
    pub fn new(volume: f32) -> Self {
        Self { volume }
    }
}

impl AudioSink for StdoutRawSink {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        let mut buf = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let v = s * self.volume;
            buf.extend_from_slice(&f32_to_s16le_clamped(v).to_le_bytes());
        }
        let mut out = std::io::stdout().lock();
        out.write_all(&buf)?;
        out.flush()?;
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EngineError> {
        std::io::stdout().lock().flush()?;
        Ok(())
    }
}

pub fn first_onset(s: &[f32], sample_rate: usize) -> usize {
    let win = (sample_rate / 200).max(1);
    let floor_sq = (3.16e-3f64) * (3.16e-3f64);
    let mut i = 0;
    while i + win <= s.len() {
        let e: f64 = s[i..i + win].iter().map(|&v| (v as f64) * (v as f64)).sum();
        if e / (win as f64) > floor_sq {
            return i;
        }
        i += win;
    }
    0
}

pub fn onset_fade(s: &mut [f32], sample_rate: usize, fade_ms: usize) {
    if fade_ms == 0 || s.is_empty() {
        return;
    }
    let onset = first_onset(s, sample_rate);
    let f = (sample_rate * fade_ms / 1000).min(s.len() - onset);
    for k in 0..f {
        s[onset + k] *= (k as f32) / (f as f32);
    }
}

pub fn tail_glitch_score(s: &[f32], sample_rate: usize) -> (f32, usize) {
    let fr = (sample_rate / 100).max(1);
    let nf = s.len() / fr;
    if nf < 8 {
        return (0.0, s.len());
    }
    let floor_rms = 5.62e-3f32;
    let mut zcr = vec![0.0f32; nf];
    let mut rms = vec![0.0f32; nf];
    let mut loud = vec![false; nf];
    for i in 0..nf {
        let b = &s[i * fr..(i + 1) * fr];
        let mut e = 0.0f64;
        let mut zc = 0usize;
        for k in 0..fr {
            e += (b[k] as f64) * (b[k] as f64);
            if k > 0 && ((b[k] >= 0.0) != (b[k - 1] >= 0.0)) {
                zc += 1;
            }
        }
        rms[i] = (e / fr as f64).sqrt() as f32;
        loud[i] = rms[i] > floor_rms;
        zcr[i] = (zc as f32) / (fr as f32);
    }
    let Some(last) = (0..nf).rev().find(|&i| loud[i]) else {
        return (0.0, s.len());
    };
    let Some(first) = (0..nf).find(|&i| loud[i]) else {
        return (0.0, s.len());
    };
    if last - first < 6 {
        return (0.0, s.len());
    }
    let body_end = first + ((last - first) as f32 * 0.7f32) as usize;
    let mut zb = Vec::new();
    let mut rb = Vec::new();
    for i in first..=body_end {
        if loud[i] {
            zb.push(zcr[i]);
            rb.push(rms[i]);
        }
    }
    if zb.len() < 3 {
        return (0.0, s.len());
    }
    zb.sort_by(|a, b| a.total_cmp(b));
    rb.sort_by(|a, b| a.total_cmp(b));
    let zthr = (zb[zb.len() / 2] * 2.5f32).max(0.20f32);
    let ethr = (rb[rb.len() / 2] * 0.6f32).max(0.10f32);
    let mut run = 0usize;
    let mut cur = 0usize;
    let mut best_len = 0usize;
    let mut best_start = 0usize;
    for i in (body_end + 1)..=last {
        if loud[i] && zcr[i] > zthr && rms[i] > ethr {
            if run == 0 {
                cur = i;
            }
            run += 1;
            if run > best_len {
                best_len = run;
                best_start = cur;
            }
        } else {
            run = 0;
        }
    }
    if best_len < 8 {
        return (0.0, s.len());
    }
    let flagged = (last + 1) - best_start;
    let ms = (flagged as f32) * (fr as f32) * 1000.0f32 / (sample_rate as f32);
    ((ms / 300.0f32).min(1.0), best_start * fr)
}

pub fn tail_trim(s: &mut Vec<f32>, sample_rate: usize, min_score: f32) -> usize {
    let (sc, trim_at) = tail_glitch_score(s, sample_rate);
    if sc < min_score || trim_at >= s.len() {
        return 0;
    }
    let guard = sample_rate * 15 / 1000;
    let new_n = (trim_at + guard).min(s.len());
    let trimmed = s.len() - new_n;
    s.truncate(new_n);
    trimmed
}
