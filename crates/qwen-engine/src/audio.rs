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
fn f32_to_s16le_clamped(s: f32) -> i16 {
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

/// Filtro pasa-altos Butterworth de 2.º orden a 70 Hz (24 kHz) para eliminar *DC offset*
/// y retumbo sub-grave antes de medir energía o calcular el espectrograma mel.
fn highpass_70hz(samples: &mut [f32]) {
    let w0 = 2.0f64 * std::f64::consts::PI * 70.0 / (SAMPLE_RATE as f64);
    let cos_w0 = w0.cos();
    let alpha = w0.sin() / std::f64::consts::SQRT_2;
    let a0 = 1.0 + alpha;
    let b0 = ((1.0 + cos_w0) * 0.5) / a0;
    let b1 = -(1.0 + cos_w0) / a0;
    let b2 = b0;
    let a1 = (-2.0 * cos_w0) / a0;
    let a2 = (1.0 - alpha) / a0;

    let mut z1 = 0.0f64;
    let mut z2 = 0.0f64;
    for s in samples.iter_mut() {
        let x = *s as f64;
        let y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        *s = y as f32;
    }
}

/// Recorta silencios inicial y final y compacta pausas internas largas (> 260 ms → 100 ms)
/// con un fundido cruzado de 5 ms en cada unión.
fn compact_silences(audio: &mut Vec<f32>, silent: bool) {
    let n = audio.len();
    let win = (SAMPLE_RATE / 50).max(1); // 20 ms = 480 muestras
    if n < SAMPLE_RATE || n < win * 4 {
        return;
    }
    let sum: f64 = audio.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let global_rms = (sum / (n as f64)).sqrt() as f32;
    if global_rms <= 0.0 {
        return;
    }
    let thresh = (0.18f32 * global_rms).max(0.004f32);
    let n_wins = n / win;
    let active: Vec<bool> = (0..n_wins)
        .map(|w| {
            let start = w * win;
            let s: f64 = audio[start..start + win]
                .iter()
                .map(|&v| (v as f64) * (v as f64))
                .sum();
            ((s / (win as f64)).sqrt() as f32) >= thresh
        })
        .collect();

    let Some(first_active) = active.iter().position(|&a| a) else {
        return;
    };
    let last_active = active.iter().rposition(|&a| a).unwrap_or(first_active);

    // Margen de 40 ms al inicio y 60 ms al final.
    let start_win = first_active.saturating_sub(2);
    let end_win = (last_active + 3).min(n_wins);

    const MAX_INTERNAL_SILENCE_WINS: usize = 13; // 260 ms
    const KEEP_HEAD_WINS: usize = 2; // 40 ms
    const KEEP_TAIL_WINS: usize = 3; // 60 ms
    let xfade = (SAMPLE_RATE / 200).min(win / 2); // 5 ms = 120 muestras

    let mut out = Vec::with_capacity(n);
    let mut w = start_win;
    while w < end_win {
        if active[w] {
            let end_sample = if w + 1 == n_wins { n } else { (w + 1) * win };
            out.extend_from_slice(&audio[w * win..end_sample]);
            w += 1;
            continue;
        }
        let run_start = w;
        while w < end_win && !active[w] {
            w += 1;
        }
        let run_len = w - run_start;
        if run_len <= MAX_INTERNAL_SILENCE_WINS {
            let end_sample = if w == n_wins { n } else { w * win };
            out.extend_from_slice(&audio[run_start * win..end_sample]);
        } else {
            // Conservar 40 ms tras el tramo previo, fundir 5 ms y conservar 60 ms antes del siguiente.
            let head_end = (run_start + KEEP_HEAD_WINS) * win;
            out.extend_from_slice(&audio[run_start * win..head_end]);
            let tail_start = (w - KEEP_TAIL_WINS) * win;
            let tail_end = if w == n_wins { n } else { w * win };
            let fade_len = xfade.min(out.len()).min(tail_end - tail_start);
            let out_base = out.len() - fade_len;
            for i in 0..fade_len {
                let t = (i as f32) / (fade_len as f32);
                out[out_base + i] = out[out_base + i] * (1.0 - t) + audio[tail_start + i] * t;
            }
            out.extend_from_slice(&audio[tail_start + fade_len..tail_end]);
        }
    }

    if out.len() >= SAMPLE_RATE && out.len() < n {
        if !silent {
            eprintln!(
                "  Clonación de voz: silencios limpiados ({:.2}s -> {:.2}s)",
                n as f32 / (SAMPLE_RATE as f32),
                out.len() as f32 / (SAMPLE_RATE as f32)
            );
        }
        *audio = out;
    }
}

/// Normaliza el RMS del audio de referencia hacia -20 dBFS (`0.1`) respetando un techo de pico
/// de -1 dBFS (`0.89`) y un límite de ganancia de +26 dB (`20.0`).
fn normalize_ref_rms(audio: &mut [f32], silent: bool) {
    if audio.is_empty() {
        return;
    }
    let sum: f64 = audio.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let rms = (sum / (audio.len() as f64)).sqrt() as f32;
    let peak = audio.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if rms < 1e-6 || peak < 1e-6 {
        return;
    }
    const TARGET_RMS: f32 = 0.1;
    const MAX_PEAK: f32 = 0.89;
    const MAX_GAIN: f32 = 20.0;
    let gain = (TARGET_RMS / rms).min(MAX_PEAK / peak).min(MAX_GAIN);
    if (gain - 1.0).abs() <= 0.02 {
        return;
    }
    apply_gain(audio, gain);
    if !silent {
        eprintln!(
            "  Clonación de voz: nivel normalizado (ganancia x{gain:.2}, RMS {rms:.4} -> {:.4})",
            rms * gain
        );
    }
}

/// Preprocesa el audio de referencia (24 kHz mono) antes de extraer el x-vector:
/// 1. Filtro pasa-altos a 70 Hz (elimina *DC offset* y retumbo sub-grave).
/// 2. Recorte de silencios inicial/final y compactación de pausas internas largas (> 260 ms).
/// 3. Normalización de nivel RMS hacia -20 dBFS con límite de pico a -1 dBFS.
pub fn preprocess_ref_audio(audio: &mut Vec<f32>, silent: bool) {
    highpass_70hz(audio);
    compact_silences(audio, silent);
    normalize_ref_rms(audio, silent);
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

    /// Senoide de 220 Hz y amplitud 0,5 durante `secs` segundos a 24 kHz.
    fn sine_220(secs: usize) -> Vec<f32> {
        let sr = SAMPLE_RATE;
        (0..secs * sr)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin())
            .collect()
    }

    /// Cruces por cero por muestra: proporcional a la frecuencia de una senoide.
    fn zero_crossing_rate(x: &[f32]) -> f64 {
        let n = x.windows(2).filter(|w| (w[0] < 0.0) != (w[1] < 0.0)).count();
        n as f64 / x.len() as f64
    }

    #[test]
    fn time_stretch_leaves_audio_unchanged_without_a_valid_rate() {
        let input = sine_220(1);
        for rate in [1.0f32, 0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(time_stretch(&input, rate), input, "rate {rate}");
        }
        assert!(time_stretch(&[], 0.5).is_empty());
        // Con menos de 16 muestras no cabe una ventana de análisis.
        let short = [0.1f32; 15];
        assert_eq!(time_stretch(&short, 0.5), short.to_vec());
    }

    #[test]
    fn time_stretch_clamps_the_rate_to_its_range() {
        let input = sine_220(1);
        assert_eq!(time_stretch(&input, 10.0), time_stretch(&input, 4.0));
        assert_eq!(time_stretch(&input, 0.1), time_stretch(&input, 0.25));
    }

    #[test]
    fn time_stretch_scales_duration_and_keeps_pitch() {
        let input = sine_220(3);
        let n_in = input.len() as f64;
        // Ventana de análisis y salto de síntesis del WSOLA para una entrada de más de 1 024 muestras.
        let (win, hop) = (1024.0f64, 512.0f64);
        let rate_zc = zero_crossing_rate(&input);
        for rate in [0.25f32, 0.5, 0.8, 1.25, 2.0, 4.0] {
            let out = time_stretch(&input, rate);
            // Cada salto avanza `hop * rate` muestras de entrada y escribe `hop` de salida, y la
            // última ventana completa añade `win`: la salida dura (n_in - win) / rate + win, con
            // el error de un salto porque el bucle termina en saltos enteros.
            let expected = (n_in - win) / rate as f64 + win;
            let len = out.len() as f64;
            assert!(
                (len - expected).abs() <= hop,
                "rate {rate}: {len} muestras, se esperaban {expected:.0} ± {hop}"
            );
            // El tono se conserva: cambiar la velocidad remuestreando movería la frecuencia en el
            // factor `rate` (20 % o más en estos casos); un 1 % separa los dos comportamientos.
            // Se excluye una ventana en cada extremo, donde el solapamiento está incompleto.
            let middle = &out[1024..out.len() - 1024];
            let ratio = zero_crossing_rate(middle) / rate_zc;
            assert!((ratio - 1.0).abs() < 0.01, "rate {rate}: la frecuencia cambió en {ratio}");
            // La suma solapada normalizada es una media ponderada de muestras de la entrada.
            let peak = out.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            assert!(peak.is_finite() && peak <= 0.5 + 1e-6, "rate {rate}: pico {peak}");
        }
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
