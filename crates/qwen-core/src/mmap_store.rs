// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::error::CoreError;

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Vista `Send + Sync + 'static` sin copias sobre un tensor BF16 inmutable mapeado en memoria.
#[derive(Clone)]
pub struct Bf16Ref {
    shard: Arc<Mmap>,
    byte_offset: usize,
    len: usize,
}

impl Bf16Ref {
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        let bytes = &self.shard[self.byte_offset..self.byte_offset + self.len * 2];
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, self.len) }
    }
}

impl Deref for Bf16Ref {
    type Target = [u16];
    #[inline]
    fn deref(&self) -> &[u16] {
        self.as_slice()
    }
}

/// Vista `Send + Sync + 'static` sobre un tensor F32 inmutable (sin copias sobre `mmap` si está alineado a 4 bytes).
#[derive(Clone)]
pub enum F32Ref {
    Mmap {
        shard: Arc<Mmap>,
        byte_offset: usize,
        len: usize,
    },
    Owned(Arc<Vec<f32>>),
}

impl F32Ref {
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match self {
            F32Ref::Mmap {
                shard,
                byte_offset,
                len,
            } => {
                let bytes = &shard[*byte_offset..*byte_offset + *len * 4];
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, *len) }
            }
            F32Ref::Owned(v) => v.as_slice(),
        }
    }
}

impl Deref for F32Ref {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

struct ShardEntry {
    mmap: Arc<Mmap>,
}

struct TensorLoc {
    shard_id: usize,
    dtype: Dtype,
    byte_offset: usize,
    byte_len: usize,
}

enum F32DataView<'a> {
    AlignedMmap {
        shard: &'a Arc<Mmap>,
        byte_offset: usize,
        len: usize,
        data: &'a [u8],
    },
    Owned(Vec<f32>),
}

/// Almacén `safetensors` de solo lectura mapeado en memoria, compatible con
/// archivo único `model.safetensors` y sharded `model.safetensors.index.json`.
///
/// Seguridad: el único código `unsafe` de `qwen-core` reside en este módulo (`Mmap::map`
/// y vistas alineadas verificadas sobre bytes inmutables mapeados).
pub struct MmapStore {
    shards: Vec<ShardEntry>,
    index: HashMap<String, TensorLoc>,
}

impl MmapStore {
    pub fn open_model(dir: &Path) -> Result<Self, CoreError> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Self::open_files(&[single]);
        }
        let idx_path = dir.join("model.safetensors.index.json");
        if idx_path.exists() {
            let raw = std::fs::read_to_string(&idx_path)?;
            let parsed: SafetensorsIndex = serde_json::from_str(&raw)?;
            let unique: BTreeSet<String> = parsed.weight_map.into_values().collect();
            let paths: Vec<_> = unique.into_iter().map(|f| dir.join(f)).collect();
            return Self::open_files(&paths);
        }
        Err(CoreError::Safetensors(format!(
            "No model.safetensors or model.safetensors.index.json in {}",
            dir.display()
        )))
    }

    pub fn open_files<P: AsRef<Path>>(paths: &[P]) -> Result<Self, CoreError> {
        let mut shards = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();

        for (shard_id, p) in paths.iter().enumerate() {
            let file = File::open(p.as_ref()).map_err(|e| {
                CoreError::Safetensors(format!("Cannot open {}: {e}", p.as_ref().display()))
            })?;
            let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| {
                CoreError::Safetensors(format!("Mmap failed for {}: {e}", p.as_ref().display()))
            })?);
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                CoreError::Safetensors(format!(
                    "Invalid safetensors {}: {e}",
                    p.as_ref().display()
                ))
            })?;
            let base_ptr = mmap.as_ptr() as usize;
            for (name, view) in st.tensors() {
                let data = view.data();
                let byte_offset = if data.is_empty() {
                    0
                } else {
                    (data.as_ptr() as usize)
                        .checked_sub(base_ptr)
                        .ok_or_else(|| {
                            CoreError::Safetensors(format!("Tensor '{name}' pointer outside shard"))
                        })?
                };
                if byte_offset.saturating_add(data.len()) > mmap.len() {
                    return Err(CoreError::Safetensors(format!(
                        "Tensor '{name}' range exceeds shard length"
                    )));
                }
                index.insert(
                    name,
                    TensorLoc {
                        shard_id,
                        dtype: view.dtype(),
                        byte_offset,
                        byte_len: data.len(),
                    },
                );
            }
            shards.push(ShardEntry { mmap });
        }

        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn locate_bf16(&self, name: &str) -> Result<(&Arc<Mmap>, usize, usize), CoreError> {
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Missing tensor '{name}'")))?;
        if loc.dtype != Dtype::BF16 {
            return Err(CoreError::Safetensors(format!(
                "Tensor '{name}' dtype is {:?}, expected BF16",
                loc.dtype
            )));
        }
        let shard = &self.shards[loc.shard_id].mmap;
        let data = &shard[loc.byte_offset..loc.byte_offset + loc.byte_len];
        if !data.len().is_multiple_of(2) || !(data.as_ptr() as usize).is_multiple_of(2) {
            return Err(CoreError::Safetensors(format!(
                "Tensor '{name}' BF16 data is not 2-byte aligned"
            )));
        }
        Ok((shard, loc.byte_offset, loc.byte_len / 2))
    }

    /// Vista sin copias de un tensor BF16 como `&[u16]` (host little-endian).
    /// Exige offset alineado a 2 bytes dentro del archivo safetensors (garantizado en exports estándar de HF).
    pub fn tensor_bf16(&self, name: &str) -> Result<&[u16], CoreError> {
        let (shard, byte_offset, len) = self.locate_bf16(name)?;
        let data = &shard[byte_offset..byte_offset + len * 2];
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u16, len) };
        Ok(slice)
    }

    pub fn tensor_bf16_ref(&self, name: &str) -> Result<Bf16Ref, CoreError> {
        let (shard, byte_offset, len) = self.locate_bf16(name)?;
        Ok(Bf16Ref {
            shard: Arc::clone(shard),
            byte_offset,
            len,
        })
    }

    fn locate_f32(&self, name: &str) -> Result<F32DataView<'_>, CoreError> {
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Missing tensor '{name}'")))?;
        let shard = &self.shards[loc.shard_id].mmap;
        let data = &shard[loc.byte_offset..loc.byte_offset + loc.byte_len];
        match loc.dtype {
            Dtype::F32 => {
                if !data.len().is_multiple_of(4) {
                    return Err(CoreError::Safetensors(format!(
                        "Tensor '{name}' F32 byte length invalid"
                    )));
                }
                if (data.as_ptr() as usize).is_multiple_of(4) {
                    Ok(F32DataView::AlignedMmap {
                        shard,
                        byte_offset: loc.byte_offset,
                        len: data.len() / 4,
                        data,
                    })
                } else {
                    let out: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    Ok(F32DataView::Owned(out))
                }
            }
            Dtype::BF16 => {
                if !data.len().is_multiple_of(2) {
                    return Err(CoreError::Safetensors(format!(
                        "Tensor '{name}' BF16 byte length invalid"
                    )));
                }
                let out: Vec<f32> = data
                    .chunks_exact(2)
                    .map(|c| {
                        let u = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((u as u32) << 16)
                    })
                    .collect();
                Ok(F32DataView::Owned(out))
            }
            other => Err(CoreError::Safetensors(format!(
                "Tensor '{name}' dtype {other:?} cannot be read as F32"
            ))),
        }
    }

    pub fn tensor_f32_ref(&self, name: &str) -> Result<F32Ref, CoreError> {
        match self.locate_f32(name)? {
            F32DataView::AlignedMmap {
                shard,
                byte_offset,
                len,
                ..
            } => Ok(F32Ref::Mmap {
                shard: Arc::clone(shard),
                byte_offset,
                len,
            }),
            F32DataView::Owned(out) => Ok(F32Ref::Owned(Arc::new(out))),
        }
    }

    /// Lee un tensor F32 (o BF16 convertido a F32) como `std::borrow::Cow<[f32]>`. Toma referencia sin copias cuando es F32 alineado a 4 bytes.
    pub fn tensor_f32(&self, name: &str) -> Result<std::borrow::Cow<'_, [f32]>, CoreError> {
        match self.locate_f32(name)? {
            F32DataView::AlignedMmap { data, len, .. } => {
                let slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, len) };
                Ok(std::borrow::Cow::Borrowed(slice))
            }
            F32DataView::Owned(out) => Ok(std::borrow::Cow::Owned(out)),
        }
    }

    pub fn tensor_bf16_opt(&self, name: &str) -> Result<Option<&[u16]>, CoreError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.tensor_bf16(name).map(Some)
    }

    pub fn tensor_f32_ref_opt(&self, name: &str) -> Result<Option<F32Ref>, CoreError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.tensor_f32_ref(name).map(Some)
    }
}
