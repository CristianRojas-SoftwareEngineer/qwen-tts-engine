// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use qwen_core::config::SAMPLE_RATE;

use crate::error::EngineError;

#[inline]
pub fn f32_to_s16le_clamped(s: f32) -> i16 {
    let clamped = if s.is_nan() { 0.0 } else { s.clamp(-1.0, 1.0) };
    (clamped * 32767.0f32) as i16
}

pub fn apply_gain(samples: &mut [f32], gain: f32) {
    if (gain - 1.0).abs() < f32::EPSILON {
        return;
    }
    for s in samples.iter_mut() {
        *s = (*s * gain).clamp(-1.0, 1.0);
    }
}

/// Estiramiento temporal preservando el pitch mediante WSOLA (`--rate`).
pub fn time_stretch(input: &[f32], rate: f32) -> Vec<f32> {
    let rate = if !rate.is_finite() || rate <= 0.0 {
        1.0
    } else {
        rate.clamp(0.25, 4.0)
    };
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
    for (i, w) in win.iter_mut().enumerate() {
        *w = 0.5f32
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
    let data_bytes_usize = n_samples.saturating_mul(2);
    let data_size = u32::try_from(data_bytes_usize).unwrap_or(u32::MAX);
    let file_size = 36u32.saturating_add(data_size);
    let byte_rate = sample_rate.saturating_mul(2);
    let mut out = Vec::with_capacity(44 + data_bytes_usize);
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

fn write_wav_header<W: Write>(w: &mut W, sample_rate: u32, data_bytes: u32) -> Result<(), EngineError> {
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

fn first_onset(s: &[f32], sample_rate: usize) -> usize {
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

const NON_WAV_EXTS: &[&str] = &[
    "mp4", "m4a", "mp3", "ogg", "opus", "flac", "aac", "wma", "webm", "mkv", "avi", "mov",
];

fn ffmpeg_convert_hint(path: &Path) -> String {
    format!(
        "Voice cloning requires 24 kHz WAV (PCM, 16-bit, mono).\nConvert first:\n  ffmpeg -i \"{}\" -ar 24000 -ac 1 output.wav",
        path.display()
    )
}

/// Validación rápida previa a la carga sobre la ruta de audio de referencia y cabecera `RIFF`/`WAVE` (con guía `ffmpeg`).
pub fn validate_wav_path_or_header(path: &Path) -> Result<(), EngineError> {
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let lower = ext.to_ascii_lowercase();
        if NON_WAV_EXTS.contains(&lower.as_str()) {
            return Err(EngineError::InvalidArg(format!(
                "Error: {} is not a WAV file (detected .{lower} format)\n{}",
                path.display(),
                ffmpeg_convert_hint(path)
            )));
        }
    }

    let mut f = File::open(path).map_err(|e| {
        EngineError::InvalidArg(format!("Error: cannot open file {}: {e}", path.display()))
    })?;
    let mut hdr = [0u8; 12];
    let n = f.read(&mut hdr)?;
    if n < 4 {
        return Err(EngineError::InvalidArg(format!(
            "Error: file too small to be audio: {}",
            path.display()
        )));
    }
    if (n >= 4 && hdr[..3] == [0, 0, 0]) || (n >= 8 && &hdr[4..8] == b"ftyp") {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is an MP4/M4A file, not a WAV file\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if n >= 3 && hdr[0] == 0xFF && (hdr[1] & 0xE0) == 0xE0 {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is an MP3 file, not a WAV file\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if n >= 4 && &hdr[..4] == b"OggS" {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is an OGG/Opus file, not a WAV file\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if n >= 4 && &hdr[..4] == b"fLaC" {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is a FLAC file, not a WAV file\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if n >= 3 && &hdr[..3] == b"ID3" {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is an MP3 file (ID3 tagged), not a WAV file\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if &hdr[..4] != b"RIFF" {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} is not a WAV file (unrecognized format)\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    if n < 12 || &hdr[8..12] != b"WAVE" {
        return Err(EngineError::InvalidArg(format!(
            "Error: {} has RIFF header but is not a WAV file (no WAVE marker)\n{}",
            path.display(),
            ffmpeg_convert_hint(path)
        )));
    }
    Ok(())
}

/// Lee un archivo WAV PCM (entero de 16 o 32 bits) y lo normaliza a muestras mono `f32` en `[-1.0, 1.0]`.
pub fn read_wav_pcm(path: &Path) -> Result<(Vec<f32>, u32), EngineError> {
    validate_wav_path_or_header(path)?;
    let f = File::open(path)?;
    let mut r = BufReader::new(f);
    let mut riff_hdr = [0u8; 12];
    r.read_exact(&mut riff_hdr)?;

    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut num_channels = 0u16;
    let mut audio_format = 0u16;
    let mut got_fmt = false;
    let mut raw_data: Option<Vec<u8>> = None;

    loop {
        let mut chunk_id = [0u8; 4];
        if r.read_exact(&mut chunk_id).is_err() {
            break;
        }
        let mut sz_buf = [0u8; 4];
        if r.read_exact(&mut sz_buf).is_err() {
            break;
        }
        let chunk_size = u32::from_le_bytes(sz_buf);

        if &chunk_id == b"fmt " {
            if chunk_size < 16 {
                return Err(EngineError::InvalidArg(format!(
                    "Error: truncated fmt chunk in {}",
                    path.display()
                )));
            }
            let mut fmt_buf = [0u8; 16];
            r.read_exact(&mut fmt_buf).map_err(|_| {
                EngineError::InvalidArg(format!("Error: truncated fmt chunk in {}", path.display()))
            })?;
            audio_format = u16::from_le_bytes([fmt_buf[0], fmt_buf[1]]);
            num_channels = u16::from_le_bytes([fmt_buf[2], fmt_buf[3]]);
            sample_rate = u32::from_le_bytes([fmt_buf[4], fmt_buf[5], fmt_buf[6], fmt_buf[7]]);
            bits_per_sample = u16::from_le_bytes([fmt_buf[14], fmt_buf[15]]);
            got_fmt = true;
            let extra = (chunk_size as usize) - 16 + ((chunk_size & 1) as usize);
            if extra > 0 {
                r.seek(SeekFrom::Current(extra as i64))?;
            }
        } else if &chunk_id == b"data" {
            let mut buf = Vec::new();
            let mut limited = (&mut r).take((chunk_size as u64).min(256 * 1024 * 1024));
            limited.read_to_end(&mut buf)?;
            raw_data = Some(buf);
            break;
        } else {
            let skip = (chunk_size as i64) + ((chunk_size & 1) as i64);
            if r.seek(SeekFrom::Current(skip)).is_err() {
                break;
            }
        }
    }

    let raw = raw_data.ok_or_else(|| {
        EngineError::InvalidArg(format!("Error: no data chunk in WAV file: {}", path.display()))
    })?;
    if raw.is_empty() {
        return Err(EngineError::InvalidArg(format!(
            "Error: empty data chunk in WAV file: {}",
            path.display()
        )));
    }
    if !got_fmt || num_channels == 0 || sample_rate == 0 {
        return Err(EngineError::InvalidArg(format!(
            "Error: missing or invalid fmt chunk in WAV file: {}",
            path.display()
        )));
    }
    if audio_format != 1 {
        return Err(EngineError::InvalidArg(format!(
            "Error: unsupported WAV format {audio_format} (only PCM=1 supported): {}",
            path.display()
        )));
    }
    if bits_per_sample != 16 && bits_per_sample != 32 {
        return Err(EngineError::InvalidArg(format!(
            "Error: unsupported bits per sample {bits_per_sample} (only 16/32 supported): {}",
            path.display()
        )));
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let n_ch = num_channels as usize;
    let total_samples = raw.len() / (bytes_per_sample * n_ch);
    let mut samples = Vec::with_capacity(total_samples);

    for i in 0..total_samples {
        if n_ch == 1 {
            let val = if bits_per_sample == 16 {
                let off = i * 2;
                i16::from_le_bytes([raw[off], raw[off + 1]]) as f32 / 32768.0f32
            } else {
                let off = i * 4;
                i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as f32
                    / 2_147_483_648.0f32
            };
            samples.push(val);
        } else {
            let mut sum = 0.0f32;
            for c in 0..n_ch {
                let off = (i * n_ch + c) * bytes_per_sample;
                sum += if bits_per_sample == 16 {
                    i16::from_le_bytes([raw[off], raw[off + 1]]) as f32 / 32768.0f32
                } else {
                    i32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]) as f32
                        / 2_147_483_648.0f32
                };
            }
            samples.push(sum / (n_ch as f32));
        }
    }

    Ok((samples, sample_rate))
}

/// Lector estricto de WAV a 24 kHz: rechaza cualquier frecuencia distinta de 24000 Hz con sugerencia de conversión `ffmpeg`.
pub fn read_wav_24khz(path: &Path) -> Result<Vec<f32>, EngineError> {
    let (samples, sr) = read_wav_pcm(path)?;
    if sr as usize != SAMPLE_RATE {
        return Err(EngineError::InvalidArg(format!(
            "Error: reference audio must be {SAMPLE_RATE} Hz (got {sr} Hz)\nConvert with: ffmpeg -i \"{}\" -ar 24000 -ac 1 output.wav",
            path.display()
        )));
    }
    Ok(samples)
}

/// Recorte conservador de silencio o *fade-out* final en el audio de referencia.
/// Se desactiva definiendo `QWEN_NO_REF_TRIM`.
pub fn trim_trailing_silence(audio: &mut Vec<f32>, sample_rate: usize, silent: bool) {
    if std::env::var_os("QWEN_NO_REF_TRIM").is_some() {
        return;
    }
    let n = audio.len();
    if n < sample_rate * 3 {
        return;
    }
    let sum: f64 = audio.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let global_rms = (sum / (n as f64)).sqrt() as f32;
    if global_rms <= 0.0 {
        return;
    }
    let thresh = (0.18f32 * global_rms).max(0.004f32);
    let win = (sample_rate / 50).max(1);
    let mut last_voiced_end: Option<usize> = None;
    let mut i = n;
    while i >= win {
        let s: f64 = audio[i - win..i]
            .iter()
            .map(|&v| (v as f64) * (v as f64))
            .sum();
        if ((s / (win as f64)).sqrt() as f32) >= thresh {
            last_voiced_end = Some(i);
            break;
        }
        i -= win;
    }
    let Some(end) = last_voiced_end else {
        return;
    };
    let mut new_n = (end + sample_rate / 20).min(n);
    let max_trim = ((n as f32) * 0.40f32) as usize;
    if new_n < n - max_trim {
        new_n = n - max_trim;
    }
    let min_keep = sample_rate * 2;
    if new_n < min_keep {
        new_n = n.min(min_keep);
    }
    if new_n < n {
        if !silent {
            eprintln!(
                "  Voice clone: trimmed {:.2}s trailing silence/fade ({:.1}s -> {:.1}s)",
                (n - new_n) as f32 / (sample_rate as f32),
                n as f32 / (sample_rate as f32),
                new_n as f32 / (sample_rate as f32)
            );
        }
        audio.truncate(new_n);
    }
}
