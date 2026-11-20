// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use crate::error::CoreError;

struct ByteTables {
    byte_to_cp: [u32; 256],
    cp_to_byte: [u8; 512],
}

fn byte_tables() -> &'static ByteTables {
    static TABLES: OnceLock<ByteTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        let mut byte_to_cp = [0u32; 256];
        let mut cp_to_byte = [0u8; 512];
        let mut is_good = [false; 256];

        for b in b'!'..=b'~' {
            byte_to_cp[b as usize] = b as u32;
            is_good[b as usize] = true;
        }
        for b in 0xA1u8..=0xACu8 {
            byte_to_cp[b as usize] = b as u32;
            is_good[b as usize] = true;
        }
        for b in 0xAEu8..=0xFFu8 {
            byte_to_cp[b as usize] = b as u32;
            is_good[b as usize] = true;
        }

        let mut extra = 0u32;
        for b in 0..256usize {
            if !is_good[b] {
                byte_to_cp[b] = 256 + extra;
                extra += 1;
            }
        }
        for (b, &cp_u32) in byte_to_cp.iter().enumerate() {
            let cp = cp_u32 as usize;
            if cp < cp_to_byte.len() {
                cp_to_byte[cp] = b as u8;
            }
        }
        ByteTables {
            byte_to_cp,
            cp_to_byte,
        }
    })
}

fn cp_to_utf8(cp: u32, out: &mut String) {
    if let Some(ch) = char::from_u32(cp) {
        out.push(ch);
    }
}

#[inline]
fn is_letter(c: u8) -> bool {
    c.is_ascii_alphabetic()
}

#[inline]
fn is_utf8_start(c: u8) -> bool {
    c >= 0xC0
}

#[inline]
fn utf8_char_len(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if c < 0xE0 {
        2
    } else if c < 0xF0 {
        3
    } else {
        4
    }
}

#[inline]
fn scan_utf8_letters(bytes: &[u8], mut i: usize) -> usize {
    let len = bytes.len();
    while i < len {
        let cc = bytes[i];
        if is_letter(cc) {
            i += 1;
        } else if is_utf8_start(cc) {
            let clen = utf8_char_len(cc);
            if i + clen > len {
                i += 1;
                break;
            }
            let valid = (1..clen).all(|k| (0x80..=0xBF).contains(&bytes[i + k]));
            if !valid {
                i += 1;
                break;
            }
            i += clen;
        } else {
            break;
        }
    }
    i
}

/// Pre-tokenizador a nivel de bytes compatible con BPE de Qwen2/Qwen3,
/// incluyendo validación de continuaciones UTF-8 y guardia de avance.
fn pre_tokenize(bytes: &[u8]) -> Vec<&[u8]> {
    let mut chunks = Vec::new();
    let len = bytes.len();
    let mut i = 0usize;

    while i < len {
        let c = bytes[i];

        // 1. Espacios en blanco: une un espacio simple al bloque siguiente si no hay más espacios
        if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
            if c == b' ' && i + 1 < len {
                let next = bytes[i + 1];
                if !matches!(next, b' ' | b'\t' | b'\n' | b'\r') {
                    let start = i;
                    i += 1;
                    let nc = bytes[i];
                    if is_letter(nc) || is_utf8_start(nc) {
                        i = scan_utf8_letters(bytes, i);
                    } else if nc.is_ascii_digit() {
                        let mut dcount = 0;
                        while i < len && bytes[i].is_ascii_digit() && dcount < 3 {
                            i += 1;
                            dcount += 1;
                        }
                    } else {
                        while i < len {
                            let cc = bytes[i];
                            if is_letter(cc)
                                || cc.is_ascii_digit()
                                || matches!(cc, b' ' | b'\t' | b'\n' | b'\r')
                                || is_utf8_start(cc)
                            {
                                break;
                            }
                            i += 1;
                        }
                    }
                    chunks.push(&bytes[start..i]);
                    continue;
                }
            }
            let start = i;
            while i < len && matches!(bytes[i], b' ' | b'\t' | b'\n' | b'\r') {
                i += 1;
            }
            chunks.push(&bytes[start..i]);
            continue;
        }

        // 2. Contracciones en inglés: 's, 't, 'm, 'd, 're, 've, 'll
        if c == b'\'' && i + 1 < len {
            let next = bytes[i + 1];
            let clen = if matches!(next, b's' | b't' | b'm' | b'd' | b'S' | b'T' | b'M' | b'D') {
                2
            } else if i + 2 < len {
                let p0 = next.to_ascii_lowercase();
                let p1 = bytes[i + 2].to_ascii_lowercase();
                if ((p0 == b'r' || p0 == b'v') && p1 == b'e') || (p0 == b'l' && p1 == b'l') {
                    3
                } else {
                    0
                }
            } else {
                0
            };
            if clen > 0 {
                chunks.push(&bytes[i..i + clen]);
                i += clen;
                continue;
            }
        }

        // 3. Patrón Qwen2: [^\r\n\p{L}\p{N}]?\p{L}+
        let mut next_is_letter = false;
        let mut lookahead = i;
        if !is_letter(c)
            && !c.is_ascii_digit()
            && c != b'\n'
            && c != b'\r'
            && i + 1 < len
        {
            let nc = bytes[i + 1];
            if is_letter(nc) || is_utf8_start(nc) {
                next_is_letter = true;
                lookahead = i;
            }
        }
        if is_letter(c) || is_utf8_start(c) {
            next_is_letter = true;
            lookahead = i;
        }
        if next_is_letter {
            let start = lookahead;
            if !is_letter(bytes[lookahead]) && !is_utf8_start(bytes[lookahead]) {
                lookahead += 1;
            }
            let loop_start = lookahead;
            lookahead = scan_utf8_letters(bytes, lookahead);
            if lookahead == loop_start {
                lookahead += 1;
            }
            if lookahead == start {
                lookahead = start + 1;
            }
            chunks.push(&bytes[start..lookahead]);
            i = lookahead;
            continue;
        }

        // 4. Dígitos: agrupados de 1 a 3
        if c.is_ascii_digit() {
            let start = i;
            let mut dcount = 0;
            while i < len && bytes[i].is_ascii_digit() && dcount < 3 {
                i += 1;
                dcount += 1;
            }
            chunks.push(&bytes[start..i]);
            continue;
        }

        // 5. Puntuación / símbolos + \r\n finales
        let start = i;
        while i < len {
            let cc = bytes[i];
            if is_letter(cc)
                || cc.is_ascii_digit()
                || matches!(cc, b' ' | b'\t' | b'\n' | b'\r')
                || is_utf8_start(cc)
            {
                break;
            }
            i += 1;
        }
        while i < len && (bytes[i] == b'\r' || bytes[i] == b'\n') {
            i += 1;
        }
        if i == start {
            i += 1;
        }
        chunks.push(&bytes[start..i]);
    }

    chunks
}

pub struct BpeTokenizer {
    vocab: HashMap<String, i32>,
    merges: HashMap<String, i32>,
    id_to_token: Vec<Option<String>>,
    max_id: i32,
}

impl BpeTokenizer {
    pub fn load_from_dir(dir: &Path) -> Result<Self, CoreError> {
        Self::load_files(&dir.join("vocab.json"), &dir.join("merges.txt"))
    }

    pub fn load_files(vocab_path: &Path, merges_path: &Path) -> Result<Self, CoreError> {
        let vocab_str = std::fs::read_to_string(vocab_path).map_err(|e| {
            CoreError::Tokenizer(format!("No se puede leer {}: {e}", vocab_path.display()))
        })?;
        let vocab: HashMap<String, i32> = serde_json::from_str(&vocab_str)?;
        let max_id = vocab.values().copied().max().unwrap_or(0).max(0);
        let table_len = (max_id as usize)
            .checked_add(1)
            .filter(|&n| n <= 16_777_216)
            .ok_or_else(|| {
                CoreError::Tokenizer(format!("max_id={max_id} del vocabulario fuera de rango"))
            })?;
        let mut id_to_token = vec![None; table_len];
        for (k, &id) in &vocab {
            if id >= 0 && (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = Some(k.clone());
            }
        }

        let merges_str = std::fs::read_to_string(merges_path).map_err(|e| {
            CoreError::Tokenizer(format!("No se puede leer {}: {e}", merges_path.display()))
        })?;
        let mut merges = HashMap::new();
        let mut rank = 0i32;
        for line in merges_str.lines() {
            let line = line.strip_suffix('\r').unwrap_or(line);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((a, b)) = line.split_once(' ') {
                if !a.is_empty() && !b.is_empty() {
                    let mut key = String::with_capacity(a.len() + 1 + b.len());
                    key.push_str(a);
                    key.push('\u{FFFF}');
                    key.push_str(b);
                    merges.insert(key, rank);
                    rank += 1;
                }
            }
        }

        Ok(Self {
            vocab,
            merges,
            id_to_token,
            max_id,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        self.encode_bytes(text.as_bytes())
    }

    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<i32> {
        let tables = byte_tables();
        let chunks = pre_tokenize(bytes);
        let mut ids = Vec::with_capacity(chunks.len() * 2 + 8);
        let mut merge_key = String::with_capacity(64);

        for chunk in chunks {
            let mut pieces: Vec<String> = Vec::with_capacity(chunk.len());
            for &b in chunk {
                let mut s = String::with_capacity(4);
                cp_to_utf8(tables.byte_to_cp[b as usize], &mut s);
                pieces.push(s);
            }

            while pieces.len() >= 2 {
                let mut best_rank = i32::MAX;
                let mut best_idx: Option<usize> = None;
                for i in 0..pieces.len() - 1 {
                    merge_key.clear();
                    merge_key.push_str(&pieces[i]);
                    merge_key.push('\u{FFFF}');
                    merge_key.push_str(&pieces[i + 1]);
                    if let Some(&r) = self.merges.get(&merge_key) {
                        if r < best_rank {
                            best_rank = r;
                            best_idx = Some(i);
                        }
                    }
                }
                let Some(idx) = best_idx else {
                    break;
                };
                let next = pieces.remove(idx + 1);
                pieces[idx].push_str(&next);
            }

            for piece in &pieces {
                if let Some(&id) = self.vocab.get(piece) {
                    ids.push(id);
                } else {
                    for &b in piece.as_bytes() {
                        let mut single = String::with_capacity(4);
                        cp_to_utf8(tables.byte_to_cp[b as usize], &mut single);
                        if let Some(&sid) = self.vocab.get(&single) {
                            ids.push(sid);
                        }
                    }
                }
            }
        }
        ids
    }

    pub fn encode_para(&self, text: &str) -> Vec<i32> {
        const PARA_TAGS: &[(&str, i32)] = &[
            ("[laugh]", 151646),
            ("[sigh]", 151647),
            ("[cough]", 151648),
            ("[breath]", 151649),
            ("[sniff]", 151650),
            ("[sneeze]", 151651),
            ("[yawn]", 151652),
            ("[uhm]", 151653),
            ("[cry]", 151654),
            ("[surprise]", 151655),
            ("[confirm]", 151656),
            ("[question]", 151657),
            ("[grumble]", 151658),
            ("[shh]", 151659),
            ("[gasp]", 151660),
            ("[groan]", 151661),
        ];

        let bytes = text.as_bytes();
        let mut ids = Vec::new();
        let mut p = 0usize;
        let mut seg = 0usize;

        while p < bytes.len() {
            let mut matched: Option<(usize, i32)> = None;
            if bytes[p] == b'[' {
                for &(tag, id) in PARA_TAGS {
                    if bytes[p..].starts_with(tag.as_bytes()) {
                        matched = Some((tag.len(), id));
                        break;
                    }
                }
            }
            if let Some((mlen, tag_id)) = matched {
                if p > seg {
                    ids.extend_from_slice(&self.encode_bytes(&bytes[seg..p]));
                }
                ids.push(tag_id);
                p += mlen.max(1);
                seg = p;
            } else {
                let uc = bytes[p];
                if uc >= 0xC0 {
                    let clen = utf8_char_len(uc);
                    let valid = p + clen <= bytes.len()
                        && (1..clen).all(|k| (0x80..=0xBF).contains(&bytes[p + k]));
                    p += if valid && clen > 1 { clen } else { 1 };
                } else {
                    p += 1;
                }
            }
        }
        if p > seg {
            ids.extend_from_slice(&self.encode_bytes(&bytes[seg..p]));
        }
        ids
    }

    pub fn decode_bytes(&self, tokens: &[i32]) -> Vec<u8> {
        let tables = byte_tables();
        let mut gpt2_str = String::new();
        for &id in tokens {
            if id >= 0 && id <= self.max_id {
                if let Some(Some(t)) = self.id_to_token.get(id as usize) {
                    gpt2_str.push_str(t);
                }
            }
        }
        let mut out = Vec::with_capacity(gpt2_str.len());
        for ch in gpt2_str.chars() {
            let cp = ch as u32;
            if (cp as usize) < tables.cp_to_byte.len() {
                out.push(tables.cp_to_byte[cp as usize]);
            } else {
                let mut buf = [0u8; 4];
                let s = ch.encode_utf8(&mut buf);
                out.extend_from_slice(s.as_bytes());
            }
        }
        out
    }

    pub fn decode(&self, tokens: &[i32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(tokens)).into_owned()
    }
}
