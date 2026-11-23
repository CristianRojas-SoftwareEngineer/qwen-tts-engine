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

/// Estiramiento temporal preservando el tono mediante WSOLA (`--rate`).
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

pub fn encode_wav_bytes(samples: &[f32]) -> Vec<u8> {
    let sample_rate = SAMPLE_RATE as u32;
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

    let (fade_in, fade_out) = edge_fade_lens(n_samples);
    for (i, &orig) in samples.iter().enumerate() {
        let s = edge_fade_out(edge_fade_in(orig, i, fade_in), i, n_samples, fade_out);
        out.extend_from_slice(&f32_to_s16le_clamped(s).to_le_bytes());
    }
    out
}

/// Longitudes de las rampas de borde de una síntesis de `n_samples` muestras: entrada de 5 ms y
/// salida de 40 ms, cada una acotada a la mitad del audio.
fn edge_fade_lens(n_samples: usize) -> (usize, usize) {
    let fade_in = (SAMPLE_RATE / 200).min(n_samples / 2);
    let fade_out = (SAMPLE_RATE * 40 / 1000).min(n_samples / 2);
    (fade_in, fade_out)
}

#[inline]
fn edge_fade_in(s: f32, i: usize, fade_in: usize) -> f32 {
    if fade_in > 0 && i < fade_in {
        s * ((i as f32) / (fade_in as f32))
    } else {
        s
    }
}

#[inline]
fn edge_fade_out(s: f32, i: usize, n_samples: usize, fade_out: usize) -> f32 {
    if fade_out > 0 && i >= n_samples - fade_out {
        s * (((n_samples - 1 - i) as f32) / (fade_out as f32))
    } else {
        s
    }
}

/// Rampas de borde para salidas en *streaming*, con el mismo resultado bit a bit que
/// `encode_wav_bytes` sobre el audio completo (la ganancia se aplica antes, fuera de aquí).
///
/// La rampa de salida necesita saber dónde acaba el audio: se retienen los últimos 40 ms (y todo
/// hasta acumular 80 ms, para que las longitudes de rampa sean las definitivas) y se emiten con la
/// llamada siguiente o en `finish`.
struct StreamEdgeRamp {
    pending: Vec<f32>,
    emitted: usize,
}

impl StreamEdgeRamp {
    fn new() -> Self {
        Self {
            pending: Vec::new(),
            emitted: 0,
        }
    }

    /// Añade `samples` y escribe en `out` (PCM s16le) las muestras que ya no pueden caer en la
    /// rampa de salida.
    fn push(&mut self, samples: &[f32], out: &mut Vec<u8>) {
        self.pending.extend_from_slice(samples);
        let (fade_in_max, hold) = edge_fade_lens(usize::MAX);
        // Con al menos 2 × `hold` muestras en total, las rampas definitivas son las máximas y
        // ninguna muestra fuera de las últimas `hold` cae en la de salida.
        if self.emitted + self.pending.len() < 2 * hold {
            return;
        }
        let ready = self.pending.len() - hold;
        out.reserve(ready * 2);
        for (j, &s) in self.pending[..ready].iter().enumerate() {
            let v = edge_fade_in(s, self.emitted + j, fade_in_max);
            out.extend_from_slice(&f32_to_s16le_clamped(v).to_le_bytes());
        }
        self.pending.drain(..ready);
        self.emitted += ready;
    }

    /// Escribe en `out` las muestras retenidas con la rampa de salida definitiva.
    fn finish(&mut self, out: &mut Vec<u8>) {
        let n_samples = self.emitted + self.pending.len();
        let (fade_in, fade_out) = edge_fade_lens(n_samples);
        out.reserve(self.pending.len() * 2);
        for (j, &s) in self.pending.iter().enumerate() {
            let i = self.emitted + j;
            let v = edge_fade_out(edge_fade_in(s, i, fade_in), i, n_samples, fade_out);
            out.extend_from_slice(&f32_to_s16le_clamped(v).to_le_bytes());
        }
        self.emitted = n_samples;
        self.pending.clear();
    }
}

pub fn write_wav(path: &Path, samples: &[f32]) -> Result<(), EngineError> {
    let bytes = encode_wav_bytes(samples);
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(&bytes)?;
    w.flush()?;
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

/// Destino de bytes PCM s16le ya procesados (ganancia y rampas de borde aplicadas).
pub trait PcmWrite {
    fn write_pcm(&mut self, b: &[u8]) -> std::io::Result<()>;
    /// Cierra el destino tras el último `write_pcm`.
    fn finish(&mut self) -> std::io::Result<()>;
}

/// Único destino de audio del camino de síntesis en *streaming*: aplica las rampas de borde
/// (entrada de 5 ms, salida de 40 ms, con retención de la cola) y entrega el PCM a `W`.
/// No aplica ganancia: el llamador la aplica justo antes (`apply_gain`).
pub struct EdgeRampSink<W: PcmWrite> {
    ramp: StreamEdgeRamp,
    writer: W,
    total_samples: usize,
}

impl<W: PcmWrite> EdgeRampSink<W> {
    pub fn new(writer: W) -> Self {
        Self {
            ramp: StreamEdgeRamp::new(),
            writer,
            total_samples: 0,
        }
    }

    /// Muestras recibidas hasta ahora (antes de retener la cola).
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }
}

impl<W: PcmWrite + Send> AudioSink for EdgeRampSink<W> {
    fn write_samples(&mut self, samples: &[f32]) -> Result<(), EngineError> {
        if samples.is_empty() {
            return Ok(());
        }
        let mut buf = Vec::new();
        self.ramp.push(samples, &mut buf);
        self.total_samples += samples.len();
        if !buf.is_empty() {
            self.writer.write_pcm(&buf)?;
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EngineError> {
        let mut buf = Vec::new();
        self.ramp.finish(&mut buf);
        if !buf.is_empty() {
            self.writer.write_pcm(&buf)?;
        }
        self.writer.finish()?;
        Ok(())
    }
}

impl PcmWrite for Vec<u8> {
    fn write_pcm(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.extend_from_slice(b);
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Escritor de PCM s16le en bruto por la salida estándar.
pub struct StdoutWriter;

impl PcmWrite for StdoutWriter {
    fn write_pcm(&mut self, buf: &[u8]) -> std::io::Result<()> {
        let mut out = std::io::stdout().lock();
        out.write_all(buf)?;
        out.flush()
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn ffmpeg_convert_hint(path: &Path) -> String {
    format!(
        "La clonación de voz requiere un WAV de 24 kHz (PCM, 16 bits, mono).\nConvierte primero:\n  ffmpeg -i \"{}\" -ar 24000 -ac 1 output.wav",
        path.display()
    )
}

/// Único lector WAV de la clonación: solo WAV PCM mono de 16 bits a 24 kHz; cualquier otra cosa da un error con la sugerencia de `ffmpeg`.
pub fn read_wav_24khz(path: &Path) -> Result<Vec<f32>, EngineError> {
    let hint = ffmpeg_convert_hint(path);
    let invalid = |msg: String| EngineError::InvalidArg(format!("{msg}\n{hint}"));

    let f = File::open(path).map_err(|e| {
        EngineError::InvalidArg(format!(
            "Error: no se puede abrir el archivo {}: {e}",
            path.display()
        ))
    })?;
    let mut r = BufReader::new(f);
    let mut riff_hdr = [0u8; 12];
    let n = r.read(&mut riff_hdr)?;
    if n < 12 || &riff_hdr[..4] != b"RIFF" || &riff_hdr[8..12] != b"WAVE" {
        return Err(invalid(format!(
            "Error: {} no es un archivo WAV",
            path.display()
        )));
    }

    // (formato, canales, frecuencia, bits por muestra)
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
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
            let truncated = || invalid(format!("Error: chunk fmt truncado en {}", path.display()));
            if chunk_size < 16 {
                return Err(truncated());
            }
            let mut fmt_buf = [0u8; 16];
            r.read_exact(&mut fmt_buf).map_err(|_| truncated())?;
            fmt = Some((
                u16::from_le_bytes([fmt_buf[0], fmt_buf[1]]),
                u16::from_le_bytes([fmt_buf[2], fmt_buf[3]]),
                u32::from_le_bytes([fmt_buf[4], fmt_buf[5], fmt_buf[6], fmt_buf[7]]),
                u16::from_le_bytes([fmt_buf[14], fmt_buf[15]]),
            ));
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

    let Some((audio_format, num_channels, sample_rate, bits_per_sample)) = fmt else {
        return Err(invalid(format!(
            "Error: el archivo WAV no tiene chunk fmt: {}",
            path.display()
        )));
    };
    if audio_format != 1
        || num_channels != 1
        || bits_per_sample != 16
        || sample_rate as usize != SAMPLE_RATE
    {
        return Err(invalid(format!(
            "Error: {} debe ser WAV PCM mono de 16 bits a {SAMPLE_RATE} Hz (se obtuvo: formato {audio_format}, {num_channels} canales, {bits_per_sample} bits, {sample_rate} Hz)",
            path.display()
        )));
    }

    let raw = raw_data.ok_or_else(|| {
        invalid(format!(
            "Error: el archivo WAV no tiene chunk data: {}",
            path.display()
        ))
    })?;
    if raw.is_empty() {
        return Err(invalid(format!(
            "Error: chunk data vacío en el archivo WAV: {}",
            path.display()
        )));
    }

    Ok(raw
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0f32)
        .collect())
}

/// Recorte conservador de silencio o *fade-out* final en el audio de referencia.
pub fn trim_trailing_silence(audio: &mut Vec<f32>, silent: bool) {
    let sample_rate = SAMPLE_RATE;
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
                "  Clonación de voz: recortados {:.2}s de silencio o fundido final ({:.1}s -> {:.1}s)",
                (n - new_n) as f32 / (sample_rate as f32),
                n as f32 / (sample_rate as f32),
                new_n as f32 / (sample_rate as f32)
            );
        }
        audio.truncate(new_n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PCM s16le de referencia: ganancia y rampas sobre el audio completo (camino sin streaming).
    fn reference_pcm(samples: &[f32], volume: f32) -> Vec<u8> {
        let mut s = samples.to_vec();
        apply_gain(&mut s, volume);
        encode_wav_bytes(&s)[44..].to_vec()
    }

    fn streamed_pcm(samples: &[f32], volume: f32, chunk: usize) -> Vec<u8> {
        let mut sink = EdgeRampSink::new(Vec::<u8>::new());
        for c in samples.chunks(chunk) {
            let mut g = c.to_vec();
            apply_gain(&mut g, volume);
            sink.write_samples(&g).unwrap();
        }
        sink.finish().unwrap();
        sink.writer
    }

    #[test]
    fn stream_edge_ramp_matches_full_wav_bit_exact() {
        let mut seed = 0x1234_5678u32;
        for &n in &[0usize, 1, 100, 1919, 1920, 1921, 2880, 5000, 48_000] {
            let samples: Vec<f32> = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 2.4
                })
                .collect();
            for &volume in &[1.0f32, 0.7, 2.0] {
                let expected = reference_pcm(&samples, volume);
                for &chunk in &[1usize, 7, 960, 1920, 3840, 100_000] {
                    assert_eq!(
                        streamed_pcm(&samples, volume, chunk),
                        expected,
                        "n={n} volume={volume} chunk={chunk}"
                    );
                }
            }
        }
    }

    #[test]
    fn stream_edge_ramp_holds_back_only_the_fade_out() {
        let mut sink = EdgeRampSink::new(Vec::<u8>::new());
        sink.write_samples(&vec![0.5f32; 1000]).unwrap();
        assert!(sink.writer.is_empty(), "con menos de 80 ms no se emite nada");
        sink.write_samples(&vec![0.5f32; 2840]).unwrap();
        let out = &sink.writer;
        // 3 840 muestras recibidas (2 frames): se emiten todas salvo las 960 de la rampa de salida.
        assert_eq!(out.len(), (3840 - 960) * 2);
    }
}
