// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use qwen_kernels::quant::{bf16_to_f32, f32_to_bf16_round};

use crate::config::{ENC_DIM, HIDDEN_SIZE, NUM_CODEBOOKS, TEXT_HIDDEN_SIZE};
use crate::error::CoreError;

pub const QVCE_MAGIC: &[u8; 4] = b"QVCE";

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceMeta {
    pub lang_id: u32,
    pub lang_name: String,
    pub model_size: u32,
    pub enc_dim: u32,
    pub ref_dur: f32,
    pub voice_name: String,
    pub flags: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceTpad {
    pub hidden: u32,
    pub tts_pad_embed: Vec<f32>,
    pub tts_bos_embed: Vec<f32>,
    pub tts_eos_embed: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceWovr {
    pub wh: u32,
    pub wth: u32,
    pub wcv: u32,
    pub fc1_bf16: Vec<u16>,
    pub fc1_bias: Vec<f32>,
    pub fc2_bf16: Vec<u16>,
    pub fc2_bias: Vec<f32>,
    pub codec_emb_bf16: Vec<u16>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceWeightTensor {
    pub name: String,
    pub data_bytes: u32,
    pub dtype_flag: u8,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceWeightsOverride {
    pub target_hidden: u32,
    pub tensors: Vec<QVoiceWeightTensor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceFile {
    pub version: u32,
    pub enc_dim: u32,
    pub speaker_embedding: Vec<f32>,
    pub ref_text: String,
    pub ref_codes: Vec<i32>,
    pub meta: Option<QVoiceMeta>,
    pub tpad: Option<QVoiceTpad>,
    pub wovr: Option<QVoiceWovr>,
    pub weights: Option<QVoiceWeightsOverride>,
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8, CoreError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, CoreError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, CoreError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32, CoreError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<f32>, CoreError> {
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_u16_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<u16>, CoreError> {
    let mut raw = vec![0u8; n * 2];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn read_i32_vec<R: Read>(r: &mut R, n: usize) -> Result<Vec<i32>, CoreError> {
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_exact_or_eof<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<usize, CoreError> {
    let mut total = 0usize;
    while total < buf.len() {
        let n = r.read(&mut buf[total..])?;
        if n == 0 {
            break;
        }
        total += n;
    }
    Ok(total)
}

fn fixed_cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

fn write_fixed_cstr<W: Write>(w: &mut W, s: &str, len: usize) -> Result<(), CoreError> {
    let mut buf = vec![0u8; len];
    let sb = s.as_bytes();
    let copy_len = sb.len().min(len.saturating_sub(1));
    buf[..copy_len].copy_from_slice(&sb[..copy_len]);
    w.write_all(&buf)?;
    Ok(())
}

impl QVoiceFile {
    pub fn read_from_file(path: &Path, icl_only: bool) -> Result<Self, CoreError> {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != QVCE_MAGIC {
            return Err(CoreError::QVoice(format!(
                "{} is not a valid .qvoice file (bad magic)",
                path.display()
            )));
        }
        let version = read_u32(&mut r)?;
        if version != 3 {
            return Err(CoreError::QVoice(format!(
                "Unsupported .qvoice version {version} (expected v3)"
            )));
        }
        let enc_dim = read_u32(&mut r)?;
        if enc_dim as usize != ENC_DIM {
            return Err(CoreError::QVoice(format!(
                "Incompatible .qvoice enc_dim={enc_dim} (expected {ENC_DIM} for 0.6B)"
            )));
        }
        let speaker_embedding = read_f32_vec(&mut r, enc_dim as usize)?;
        let ref_text_len = read_u32(&mut r)? as usize;
        if ref_text_len > 1_000_000 {
            return Err(CoreError::QVoice("Corrupt ref_text_len".to_string()));
        }
        let mut ref_text_bytes = vec![0u8; ref_text_len];
        if ref_text_len > 0 {
            r.read_exact(&mut ref_text_bytes)?;
        }
        let ref_text = String::from_utf8_lossy(&ref_text_bytes).into_owned();

        let n_ref_frames = read_u32(&mut r)? as usize;
        if n_ref_frames > 1_000_000 {
            return Err(CoreError::QVoice("Corrupt n_ref_frames".to_string()));
        }
        let ref_codes = if n_ref_frames > 0 {
            read_i32_vec(&mut r, n_ref_frames * NUM_CODEBOOKS)?
        } else {
            Vec::new()
        };

        let mut meta = None;
        let mut tpad = None;
        let mut wovr = None;
        let mut weights = None;

        let mut tag4 = [0u8; 4];
        let mut pending: Option<[u8; 4]> = None;

        let n = read_exact_or_eof(&mut r, &mut tag4)?;
        if n == 4 {
            if &tag4 == b"META" {
                let lang_id = read_u32(&mut r)?;
                let mut lang_raw = [0u8; 16];
                r.read_exact(&mut lang_raw)?;
                let model_size = read_u32(&mut r)?;
                let meta_enc_dim = read_u32(&mut r)?;
                let ref_dur = read_f32(&mut r)?;
                let mut vname_raw = [0u8; 64];
                r.read_exact(&mut vname_raw)?;
                let flags = read_u32(&mut r)?;
                meta = Some(QVoiceMeta {
                    lang_id,
                    lang_name: fixed_cstr(&lang_raw),
                    model_size,
                    enc_dim: meta_enc_dim,
                    ref_dur,
                    voice_name: fixed_cstr(&vname_raw),
                    flags,
                });
            } else {
                pending = Some(tag4);
            }
        }

        let next4 = if let Some(p) = pending.take() {
            Some(p)
        } else {
            let n = read_exact_or_eof(&mut r, &mut tag4)?;
            if n == 4 { Some(tag4) } else { None }
        };

        if let Some(t4) = next4 {
            if &t4 == b"TPAD" {
                let hidden = read_u32(&mut r)?;
                if hidden as usize != HIDDEN_SIZE {
                    return Err(CoreError::QVoice(format!(
                        "Incompatible TPAD hidden={hidden} (expected {HIDDEN_SIZE})"
                    )));
                }
                let tts_pad_embed = read_f32_vec(&mut r, HIDDEN_SIZE)?;
                let tts_bos_embed = read_f32_vec(&mut r, HIDDEN_SIZE)?;
                let tts_eos_embed = read_f32_vec(&mut r, HIDDEN_SIZE)?;
                tpad = Some(QVoiceTpad {
                    hidden,
                    tts_pad_embed,
                    tts_bos_embed,
                    tts_eos_embed,
                });
            } else {
                pending = Some(t4);
            }
        }

        let next4 = if let Some(p) = pending.take() {
            Some(p)
        } else {
            let n = read_exact_or_eof(&mut r, &mut tag4)?;
            if n == 4 { Some(tag4) } else { None }
        };

        if let Some(t4) = next4 {
            if &t4 == b"WOVR" {
                let wh = read_u32(&mut r)?;
                let wth = read_u32(&mut r)?;
                let wcv = read_u32(&mut r)?;
                if wh as usize != HIDDEN_SIZE || wth as usize != TEXT_HIDDEN_SIZE {
                    return Err(CoreError::QVoice(format!(
                        "Incompatible WOVR dimensions wh={wh} wth={wth} (expected {HIDDEN_SIZE}/{TEXT_HIDDEN_SIZE})"
                    )));
                }
                let fc1_bf16 = read_u16_vec(&mut r, wth as usize * wth as usize)?;
                let fc1_bias = read_f32_vec(&mut r, wth as usize)?;
                let fc2_bf16 = read_u16_vec(&mut r, wh as usize * wth as usize)?;
                let fc2_bias = read_f32_vec(&mut r, wh as usize)?;
                let codec_emb_bf16 = read_u16_vec(&mut r, wcv as usize * wh as usize)?;
                wovr = Some(QVoiceWovr {
                    wh,
                    wth,
                    wcv,
                    fc1_bf16,
                    fc1_bias,
                    fc2_bf16,
                    fc2_bias,
                    codec_emb_bf16,
                });
            } else {
                pending = Some(t4);
            }
        }

        if !icl_only {
            let next4 = if let Some(p) = pending.take() {
                Some(p)
            } else {
                let n = read_exact_or_eof(&mut r, &mut tag4)?;
                if n == 4 { Some(tag4) } else { None }
            };

            if let Some(t4) = next4 {
                if &t4 == b"WDLT" {
                    let target_hidden = read_u32(&mut r)?;
                    if target_hidden as usize != HIDDEN_SIZE {
                        return Err(CoreError::QVoice(format!(
                            "Incompatible WDELTA target_hidden={target_hidden} (expected {HIDDEN_SIZE})"
                        )));
                    }
                    let n_tensors = read_u32(&mut r)?;
                    let mut tensors = Vec::with_capacity(n_tensors as usize);
                    for _ in 0..n_tensors {
                        let name_len = read_u16(&mut r)? as usize;
                        let mut name_buf = vec![0u8; name_len];
                        r.read_exact(&mut name_buf)?;
                        let name = String::from_utf8_lossy(&name_buf).into_owned();
                        let data_bytes = read_u32(&mut r)?;
                        let dtype_flag = read_u8(&mut r)?;
                        let compressed_size = read_u32(&mut r)? as usize;
                        let mut payload = vec![0u8; compressed_size];
                        r.read_exact(&mut payload)?;
                        tensors.push(QVoiceWeightTensor {
                            name,
                            data_bytes,
                            dtype_flag,
                            payload,
                        });
                    }
                    weights = Some(QVoiceWeightsOverride {
                        target_hidden,
                        tensors,
                    });
                }
            }
        }

        Ok(Self {
            version,
            enc_dim,
            speaker_embedding,
            ref_text,
            ref_codes,
            meta,
            tpad,
            wovr,
            weights,
        })
    }

    pub fn write_to_file(&self, path: &Path) -> Result<(), CoreError> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);

        w.write_all(QVCE_MAGIC)?;
        w.write_all(&3u32.to_le_bytes())?;
        w.write_all(&self.enc_dim.to_le_bytes())?;
        for &v in &self.speaker_embedding {
            w.write_all(&v.to_le_bytes())?;
        }
        let rt_bytes = self.ref_text.as_bytes();
        w.write_all(&(rt_bytes.len() as u32).to_le_bytes())?;
        w.write_all(rt_bytes)?;

        let n_ref_frames = (self.ref_codes.len() / NUM_CODEBOOKS) as u32;
        w.write_all(&n_ref_frames.to_le_bytes())?;
        for &c in &self.ref_codes {
            w.write_all(&c.to_le_bytes())?;
        }

        if let Some(m) = &self.meta {
            w.write_all(b"META")?;
            w.write_all(&m.lang_id.to_le_bytes())?;
            write_fixed_cstr(&mut w, &m.lang_name, 16)?;
            w.write_all(&m.model_size.to_le_bytes())?;
            w.write_all(&m.enc_dim.to_le_bytes())?;
            w.write_all(&m.ref_dur.to_le_bytes())?;
            write_fixed_cstr(&mut w, &m.voice_name, 64)?;
            w.write_all(&m.flags.to_le_bytes())?;
        }
        if let Some(t) = &self.tpad {
            w.write_all(b"TPAD")?;
            w.write_all(&t.hidden.to_le_bytes())?;
            for &v in &t.tts_pad_embed {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &t.tts_bos_embed {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in &t.tts_eos_embed {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        if let Some(wv) = &self.wovr {
            w.write_all(b"WOVR")?;
            w.write_all(&wv.wh.to_le_bytes())?;
            w.write_all(&wv.wth.to_le_bytes())?;
            w.write_all(&wv.wcv.to_le_bytes())?;
            for &u in &wv.fc1_bf16 {
                w.write_all(&u.to_le_bytes())?;
            }
            for &v in &wv.fc1_bias {
                w.write_all(&v.to_le_bytes())?;
            }
            for &u in &wv.fc2_bf16 {
                w.write_all(&u.to_le_bytes())?;
            }
            for &v in &wv.fc2_bias {
                w.write_all(&v.to_le_bytes())?;
            }
            for &u in &wv.codec_emb_bf16 {
                w.write_all(&u.to_le_bytes())?;
            }
        }
        if let Some(wo) = &self.weights {
            w.write_all(b"WDLT")?;
            w.write_all(&wo.target_hidden.to_le_bytes())?;
            w.write_all(&(wo.tensors.len() as u32).to_le_bytes())?;
            for t in &wo.tensors {
                let nb = t.name.as_bytes();
                w.write_all(&(nb.len() as u16).to_le_bytes())?;
                w.write_all(nb)?;
                w.write_all(&t.data_bytes.to_le_bytes())?;
                w.write_all(&[t.dtype_flag])?;
                w.write_all(&(t.payload.len() as u32).to_le_bytes())?;
                w.write_all(&t.payload)?;
            }
        }

        w.flush()?;
        Ok(())
    }
}

pub fn qvoice_tensor_alpha(tname: &str, voice_strength: f32, vs_l0: i32, vs_l1: i32) -> f32 {
    if voice_strength >= 0.9999 {
        return 1.0;
    }
    if let Some(rest) = tname.strip_prefix("talker.model.layers.") {
        if let Some((num_s, suffix)) = rest.split_once('.') {
            if let Ok(li) = num_s.parse::<i32>() {
                if li >= vs_l0
                    && li <= vs_l1
                    && (suffix.starts_with("self_attn.") || suffix.starts_with("mlp."))
                {
                    return voice_strength.clamp(0.0, 1.0);
                }
            }
        }
    }
    1.0
}

pub fn decode_wdelta_bf16(
    cv_orig: &[u16],
    tensor: &QVoiceWeightTensor,
    voice_strength: f32,
    vs_l0: i32,
    vs_l1: i32,
) -> Result<Vec<u16>, CoreError> {
    match tensor.dtype_flag {
        4 => {
            let n16 = (tensor.data_bytes as usize) / 2;
            if cv_orig.len() != n16 {
                return Err(CoreError::QVoice(format!(
                    "WDELTA tensor '{}' length mismatch: cv={} delta={}",
                    tensor.name,
                    cv_orig.len(),
                    n16
                )));
            }
            let raw = lz4_flex::block::decompress(&tensor.payload, tensor.data_bytes as usize)
                .map_err(|e| CoreError::QVoice(format!("LZ4 decompress '{}': {e}", tensor.name)))?;
            let alpha = qvoice_tensor_alpha(&tensor.name, voice_strength, vs_l0, vs_l1);
            let mut out = Vec::with_capacity(n16);
            if (alpha - 1.0).abs() < f32::EPSILON {
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    let d = i16::from_le_bytes([c[0], c[1]]) as i32;
                    out.push((cv_orig[i] as i32 + d) as u16);
                }
            } else {
                for (i, c) in raw.chunks_exact(2).enumerate() {
                    let d = i16::from_le_bytes([c[0], c[1]]) as i32;
                    let vc = bf16_to_f32(cv_orig[i]);
                    let vcl = bf16_to_f32((cv_orig[i] as i32 + d) as u16);
                    out.push(f32_to_bf16_round(vc + alpha * (vcl - vc)));
                }
            }
            Ok(out)
        }
        0 => {
            if !tensor.payload.len().is_multiple_of(2) {
                return Err(CoreError::QVoice(format!(
                    "Raw .qvoice tensor '{}' odd byte count",
                    tensor.name
                )));
            }
            Ok(tensor
                .payload
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect())
        }
        other => Err(CoreError::QVoice(format!(
            "Unsupported .qvoice tensor dtype_flag={other} for '{}'",
            tensor.name
        ))),
    }
}

pub fn encode_wdelta_bf16(
    name: &str,
    base_bf16: &[u16],
    cv_bf16: &[u16],
) -> Result<QVoiceWeightTensor, CoreError> {
    if base_bf16.len() != cv_bf16.len() {
        return Err(CoreError::QVoice(format!(
            "WDELTA encode '{name}' length mismatch"
        )));
    }
    let mut delta_bytes = Vec::with_capacity(base_bf16.len() * 2);
    for (&b, &c) in base_bf16.iter().zip(cv_bf16.iter()) {
        let d = (b as i32 - c as i32) as i16;
        delta_bytes.extend_from_slice(&d.to_le_bytes());
    }
    let payload = lz4_flex::block::compress(&delta_bytes);
    Ok(QVoiceWeightTensor {
        name: name.to_string(),
        data_bytes: delta_bytes.len() as u32,
        dtype_flag: 4,
        payload,
    })
}

pub fn read_bin_embedding(path: &Path) -> Result<Vec<f32>, CoreError> {
    let raw = std::fs::read(path)?;
    if raw.len() != ENC_DIM * 4 {
        return Err(CoreError::QVoice(format!(
            "{} must be {} bytes ({} x f32), got {} bytes",
            path.display(),
            ENC_DIM * 4,
            ENC_DIM,
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn write_bin_embedding(path: &Path, emb: &[f32]) -> Result<(), CoreError> {
    if emb.len() != ENC_DIM {
        return Err(CoreError::QVoice(format!(
            "Expected {ENC_DIM} floats for .bin embedding, got {}",
            emb.len()
        )));
    }
    let mut raw = Vec::with_capacity(ENC_DIM * 4);
    for &v in emb {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, raw)?;
    Ok(())
}
