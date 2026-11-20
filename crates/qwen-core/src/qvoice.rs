// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Formato de voz `.qvoice` versión 4 (`QVCE`), el único formato de voz del motor.
//!
//! Sin secciones con etiqueta: todos los campos son obligatorios y van en este
//! orden, little-endian: `magic` (4 B), `version` (`u32` = 4), `enc_dim` (`u32` = 1024),
//! x-vector (`enc_dim × f32`), `voice_name_len` (`u32`) y nombre UTF-8, `lang_id` (`u32`,
//! 0 = sin idioma), `ref_text_len` (`u32`) y texto UTF-8, `n_ref_frames` (`u32`) y
//! `n_ref_frames × 16 × i32` códigos ICL.

use std::path::Path;

use crate::config::{language_name, ENC_DIM, NUM_CODEBOOKS};
use crate::error::CoreError;

pub const QVCE_MAGIC: &[u8; 4] = b"QVCE";
pub const QVOICE_VERSION: u32 = 4;
/// Tamaño máximo del nombre de voz, en bytes.
pub const MAX_VOICE_NAME_LEN: usize = 256;
/// Tamaño máximo del texto de referencia, en bytes.
pub const MAX_REF_TEXT_LEN: usize = 1_000_000;
/// Número máximo de fotogramas de referencia ICL.
pub const MAX_REF_FRAMES: usize = 1_000_000;

/// Representación en memoria de un perfil de voz `.qvoice` v4.
#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceFile {
    /// x-vector del locutor (`ENC_DIM` valores).
    pub speaker_embedding: Vec<f32>,
    /// Nombre de la voz (puede estar vacío).
    pub voice_name: String,
    /// Id de idioma (`lookup_language`), o 0 si la voz no tiene idioma.
    pub lang_id: u32,
    /// Transcripción del audio de referencia (vacía si no hay ICL).
    pub ref_text: String,
    /// Códigos del audio de referencia, `n_ref_frames × 16` (vacío si no hay ICL).
    pub ref_codes: Vec<i32>,
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let s = self.buf.get(self.pos..end)?;
        self.pos = end;
        Some(s)
    }

    fn u32(&mut self) -> Option<u32> {
        self.take(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

fn valid_lang_id(lang_id: u32) -> bool {
    lang_id == 0
        || i32::try_from(lang_id)
            .ok()
            .and_then(language_name)
            .is_some()
}

impl QVoiceFile {
    /// Indica si la voz lleva ICL (texto y códigos de referencia).
    pub fn has_icl(&self) -> bool {
        !self.ref_codes.is_empty() && !self.ref_text.is_empty()
    }

    /// Número de fotogramas de referencia ICL.
    pub fn n_ref_frames(&self) -> usize {
        self.ref_codes.len() / NUM_CODEBOOKS
    }

    /// `true` si el archivo empieza por la firma `QVCE`, sea cual sea su versión.
    pub fn has_magic(path: &Path) -> bool {
        use std::io::Read;
        let mut magic = [0u8; 4];
        std::fs::File::open(path)
            .and_then(|mut f| f.read_exact(&mut magic))
            .is_ok()
            && &magic == QVCE_MAGIC
    }

    pub fn read_from_file(path: &Path) -> Result<Self, CoreError> {
        let bytes = std::fs::read(path)
            .map_err(|e| CoreError::QVoice(format!("No se puede leer {}: {e}", path.display())))?;
        Self::parse(&bytes, path)
    }

    fn parse(bytes: &[u8], path: &Path) -> Result<Self, CoreError> {
        let p = path.display();
        let trunc = || CoreError::QVoice(format!("{p} está truncado"));
        let corrupt = |what: &str| CoreError::QVoice(format!("{p}: {what}"));
        let mut r = Reader { buf: bytes, pos: 0 };

        let magic = r.take(4).ok_or_else(trunc)?;
        if magic != QVCE_MAGIC {
            return Err(CoreError::QVoice(format!(
                "{p} no es un archivo .qvoice válido (magic incorrecto)"
            )));
        }
        let version = r.u32().ok_or_else(trunc)?;
        if version != QVOICE_VERSION {
            return Err(CoreError::QVoice(format!(
                "{p}: .qvoice no soportado (version {version}; se requiere version {QVOICE_VERSION}): clona la voz de nuevo"
            )));
        }
        let enc_dim = r.u32().ok_or_else(trunc)? as usize;
        if enc_dim != ENC_DIM {
            return Err(CoreError::QVoice(format!(
                "{p}: .qvoice incompatible, enc_dim={enc_dim} (se esperaba {ENC_DIM})"
            )));
        }
        let emb_raw = r.take(enc_dim * 4).ok_or_else(trunc)?;
        let speaker_embedding: Vec<f32> = emb_raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let name_len = r.u32().ok_or_else(trunc)? as usize;
        if name_len > MAX_VOICE_NAME_LEN {
            return Err(corrupt("voice_name_len demasiado grande"));
        }
        let name_raw = r.take(name_len).ok_or_else(trunc)?;
        let voice_name = std::str::from_utf8(name_raw)
            .map_err(|_| corrupt("voice_name no es UTF-8 válido"))?
            .to_string();

        let lang_id = r.u32().ok_or_else(trunc)?;
        if !valid_lang_id(lang_id) {
            return Err(corrupt(&format!("lang_id {lang_id} inválido")));
        }

        let text_len = r.u32().ok_or_else(trunc)? as usize;
        if text_len > MAX_REF_TEXT_LEN {
            return Err(corrupt("ref_text_len demasiado grande"));
        }
        let text_raw = r.take(text_len).ok_or_else(trunc)?;
        let ref_text = std::str::from_utf8(text_raw)
            .map_err(|_| corrupt("ref_text no es UTF-8 válido"))?
            .to_string();

        let n_frames = r.u32().ok_or_else(trunc)? as usize;
        if n_frames > MAX_REF_FRAMES {
            return Err(corrupt("n_ref_frames demasiado grande"));
        }
        let codes_raw = r.take(n_frames * NUM_CODEBOOKS * 4).ok_or_else(trunc)?;
        let ref_codes: Vec<i32> = codes_raw
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        if r.pos != bytes.len() {
            return Err(corrupt("sobran bytes tras el último campo"));
        }
        if ref_text.is_empty() != (n_frames == 0) {
            return Err(corrupt(
                "inconsistent ICL data (ref_text and ref_codes must both be present or both empty)",
            ));
        }

        Ok(Self {
            speaker_embedding,
            voice_name,
            lang_id,
            ref_text,
            ref_codes,
        })
    }

    pub fn write_to_file(&self, path: &Path) -> Result<(), CoreError> {
        let err = |what: &str| CoreError::QVoice(format!("{}: {what}", path.display()));
        if self.speaker_embedding.len() != ENC_DIM {
            return Err(err(&format!(
                "speaker embedding must have {ENC_DIM} values, got {}",
                self.speaker_embedding.len()
            )));
        }
        if self.voice_name.len() > MAX_VOICE_NAME_LEN {
            return Err(err("el nombre de la voz excede 256 bytes"));
        }
        if !valid_lang_id(self.lang_id) {
            return Err(err(&format!("lang_id {} inválido", self.lang_id)));
        }
        if self.ref_text.len() > MAX_REF_TEXT_LEN {
            return Err(err("ref_text excede 1000000 bytes"));
        }
        if self.n_ref_frames() * NUM_CODEBOOKS != self.ref_codes.len() {
            return Err(err("la longitud de ref_codes no es múltiplo de 16"));
        }
        let n_frames = self.n_ref_frames();
        if n_frames > MAX_REF_FRAMES {
            return Err(err("demasiados frames de referencia"));
        }
        if self.ref_text.is_empty() != (n_frames == 0) {
            return Err(err(
                "inconsistent ICL data (ref_text and ref_codes must both be present or both empty)",
            ));
        }

        let mut out = Vec::with_capacity(
            4124 + self.voice_name.len() + self.ref_text.len() + self.ref_codes.len() * 4,
        );
        out.extend_from_slice(QVCE_MAGIC);
        out.extend_from_slice(&QVOICE_VERSION.to_le_bytes());
        out.extend_from_slice(&(ENC_DIM as u32).to_le_bytes());
        for &v in &self.speaker_embedding {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.voice_name.len() as u32).to_le_bytes());
        out.extend_from_slice(self.voice_name.as_bytes());
        out.extend_from_slice(&self.lang_id.to_le_bytes());
        out.extend_from_slice(&(self.ref_text.len() as u32).to_le_bytes());
        out.extend_from_slice(self.ref_text.as_bytes());
        out.extend_from_slice(&(n_frames as u32).to_le_bytes());
        for &c in &self.ref_codes {
            out.extend_from_slice(&c.to_le_bytes());
        }
        std::fs::write(path, out)?;
        Ok(())
    }
}
