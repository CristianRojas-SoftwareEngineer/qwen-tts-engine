// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::collections::HashMap;
use std::fs::File;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::error::CoreError;

/// Vista `Send + Sync + 'static` sin copias sobre un tensor BF16 inmutable mapeado en memoria.
#[derive(Clone)]
pub struct Bf16Ref {
    mmap: Arc<Mmap>,
    byte_offset: usize,
    len: usize,
}

impl Bf16Ref {
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        let bytes = &self.mmap[self.byte_offset..self.byte_offset + self.len * 2];
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

/// Vista `Send + Sync + 'static` sobre un tensor leído como F32: sin copias sobre `mmap` si es F32; copia convertida si es BF16.
#[derive(Clone)]
pub enum F32Ref {
    Mmap {
        mmap: Arc<Mmap>,
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
                mmap,
                byte_offset,
                len,
            } => {
                let bytes = &mmap[*byte_offset..*byte_offset + *len * 4];
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

struct TensorLoc {
    dtype: Dtype,
    byte_offset: usize,
    byte_len: usize,
}

enum F32DataView<'a> {
    AlignedMmap {
        mmap: &'a Arc<Mmap>,
        byte_offset: usize,
        len: usize,
    },
    Owned(Vec<f32>),
}

/// Almacén `safetensors` de solo lectura mapeado en memoria, de un solo archivo `model.safetensors`.
///
/// Seguridad: el único código `unsafe` de `qwen-core` reside en este módulo (`Mmap::map`
/// y vistas alineadas verificadas sobre bytes inmutables mapeados).
pub struct MmapStore {
    mmap: Arc<Mmap>,
    index: HashMap<String, TensorLoc>,
}

impl MmapStore {
    pub fn open_model(dir: &Path) -> Result<Self, CoreError> {
        let single = dir.join("model.safetensors");
        if single.exists() {
            return Self::open_file(&single);
        }
        Err(CoreError::Safetensors(format!(
            "No hay model.safetensors en {}",
            dir.display()
        )))
    }

    fn open_file(path: &Path) -> Result<Self, CoreError> {
        let mut index = HashMap::new();

        let file = File::open(path).map_err(|e| {
            CoreError::Safetensors(format!("No se puede abrir {}: {e}", path.display()))
        })?;
        let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| {
            CoreError::Safetensors(format!("Falló el mmap de {}: {e}", path.display()))
        })?);
        let st = SafeTensors::deserialize(&mmap).map_err(|e| {
            CoreError::Safetensors(format!("safetensors inválido en {}: {e}", path.display()))
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
                        CoreError::Safetensors(format!(
                            "El puntero del tensor '{name}' queda fuera del archivo"
                        ))
                    })?
            };
            if byte_offset.saturating_add(data.len()) > mmap.len() {
                return Err(CoreError::Safetensors(format!(
                    "El rango del tensor '{name}' excede la longitud del archivo"
                )));
            }
            index.insert(
                name,
                TensorLoc {
                    dtype: view.dtype(),
                    byte_offset,
                    byte_len: data.len(),
                },
            );
        }

        Ok(Self { mmap, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn locate_bf16(&self, name: &str) -> Result<(&Arc<Mmap>, usize, usize), CoreError> {
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Falta el tensor '{name}'")))?;
        if loc.dtype != Dtype::BF16 {
            return Err(CoreError::Safetensors(format!(
                "El dtype del tensor '{name}' es {:?}, se esperaba BF16",
                loc.dtype
            )));
        }
        let mmap = &self.mmap;
        let data = &mmap[loc.byte_offset..loc.byte_offset + loc.byte_len];
        if !data.len().is_multiple_of(2) || !(data.as_ptr() as usize).is_multiple_of(2) {
            return Err(CoreError::Safetensors(format!(
                "Los datos BF16 del tensor '{name}' no están alineados a 2 bytes"
            )));
        }
        Ok((mmap, loc.byte_offset, loc.byte_len / 2))
    }

    pub fn tensor_bf16_ref(&self, name: &str) -> Result<Bf16Ref, CoreError> {
        let (mmap, byte_offset, len) = self.locate_bf16(name)?;
        Ok(Bf16Ref {
            mmap: Arc::clone(mmap),
            byte_offset,
            len,
        })
    }

    fn locate_f32(&self, name: &str) -> Result<F32DataView<'_>, CoreError> {
        let loc = self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Falta el tensor '{name}'")))?;
        let mmap = &self.mmap;
        let data = &mmap[loc.byte_offset..loc.byte_offset + loc.byte_len];
        match loc.dtype {
            Dtype::F32 => {
                if !data.len().is_multiple_of(4) {
                    return Err(CoreError::Safetensors(format!(
                        "Longitud en bytes F32 no válida en el tensor '{name}'"
                    )));
                }
                if !(data.as_ptr() as usize).is_multiple_of(4) {
                    return Err(CoreError::Safetensors(format!(
                        "Los datos F32 del tensor '{name}' no están alineados a 4 bytes"
                    )));
                }
                Ok(F32DataView::AlignedMmap {
                    mmap,
                    byte_offset: loc.byte_offset,
                    len: data.len() / 4,
                })
            }
            Dtype::BF16 => {
                if !data.len().is_multiple_of(2) {
                    return Err(CoreError::Safetensors(format!(
                        "Longitud en bytes BF16 no válida en el tensor '{name}'"
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
                "El tensor '{name}' de dtype {other:?} no se puede leer como F32"
            ))),
        }
    }

    pub fn tensor_f32_ref(&self, name: &str) -> Result<F32Ref, CoreError> {
        match self.locate_f32(name)? {
            F32DataView::AlignedMmap {
                mmap,
                byte_offset,
                len,
                ..
            } => Ok(F32Ref::Mmap {
                mmap: Arc::clone(mmap),
                byte_offset,
                len,
            }),
            F32DataView::Owned(out) => Ok(F32Ref::Owned(Arc::new(out))),
        }
    }
}
