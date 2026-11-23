// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Formato de voz `.qvoice` versión 6 (`QVCE`), el único formato de voz del motor.
//!
//! Sin secciones con etiqueta: todos los campos son obligatorios y van en este
//! orden, little-endian: `magic` (4 B), `version` (`u32` = 6), `enc_dim` (`u32` = 1024),
//! x-vector (`enc_dim × f32`) y `lang_id` (`u32`, 0 = sin idioma): 4112 bytes justos. El
//! nombre de la voz es el nombre del archivo.

use std::path::Path;

use crate::config::{language_name, ENC_DIM};
use crate::error::CoreError;

pub const QVCE_MAGIC: &[u8; 4] = b"QVCE";
pub const QVOICE_VERSION: u32 = 6;

/// Representación en memoria de un perfil de voz `.qvoice` v6.
#[derive(Debug, Clone, PartialEq)]
pub struct QVoiceFile {
    /// x-vector del locutor (`ENC_DIM` valores).
    pub speaker_embedding: Vec<f32>,
    /// Id de idioma (`lookup_language`), o 0 si la voz no tiene idioma.
    pub lang_id: u32,
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

        let lang_id = r.u32().ok_or_else(trunc)?;
        if !valid_lang_id(lang_id) {
            return Err(corrupt(&format!("lang_id {lang_id} inválido")));
        }

        if r.pos != bytes.len() {
            return Err(corrupt("sobran bytes tras el último campo"));
        }

        Ok(Self {
            speaker_embedding,
            lang_id,
        })
    }

    pub fn write_to_file(&self, path: &Path) -> Result<(), CoreError> {
        let err = |what: &str| CoreError::QVoice(format!("{}: {what}", path.display()));
        if self.speaker_embedding.len() != ENC_DIM {
            return Err(err(&format!(
                "el x-vector debe tener {ENC_DIM} valores, tiene {}",
                self.speaker_embedding.len()
            )));
        }
        if !valid_lang_id(self.lang_id) {
            return Err(err(&format!("lang_id {} inválido", self.lang_id)));
        }

        let mut out = Vec::with_capacity(4112);
        out.extend_from_slice(QVCE_MAGIC);
        out.extend_from_slice(&QVOICE_VERSION.to_le_bytes());
        out.extend_from_slice(&(ENC_DIM as u32).to_le_bytes());
        for &v in &self.speaker_embedding {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&self.lang_id.to_le_bytes());
        std::fs::write(path, out)?;
        Ok(())
    }
}
