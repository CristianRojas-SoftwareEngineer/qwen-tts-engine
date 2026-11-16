use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::error::CoreError;

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

struct ShardEntry {
    mmap: Mmap,
    tensor_names: Vec<String>,
}

/// Read-only memory-mapped safetensors store supporting both single-file
/// `model.safetensors` and sharded `model.safetensors.index.json`.
///
/// Safety: the only `unsafe` in `qwen-core` lives in this module (`Mmap::map`
/// and verified-aligned zero-copy views over immutable mapped bytes).
pub struct MmapStore {
    shards: Vec<ShardEntry>,
    index: HashMap<String, usize>,
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
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                CoreError::Safetensors(format!("Mmap failed for {}: {e}", p.as_ref().display()))
            })?;
            let st = SafeTensors::deserialize(&mmap).map_err(|e| {
                CoreError::Safetensors(format!(
                    "Invalid safetensors {}: {e}",
                    p.as_ref().display()
                ))
            })?;
            let names: Vec<String> = st.names().into_iter().map(|s| (*s).to_string()).collect();
            for name in &names {
                index.insert(name.clone(), shard_id);
            }
            shards.push(ShardEntry {
                mmap,
                tensor_names: names,
            });
        }

        Ok(Self { shards, index })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.shards
            .iter()
            .flat_map(|s| s.tensor_names.iter().map(|n| n.as_str()))
    }

    pub fn mmaps_len(&self) -> usize {
        self.shards.len()
    }

    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>, CoreError> {
        let shard_id = *self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Missing tensor '{name}'")))?;
        let st = SafeTensors::deserialize(&self.shards[shard_id].mmap)
            .map_err(|e| CoreError::Safetensors(e.to_string()))?;
        let view = st
            .tensor(name)
            .map_err(|e| CoreError::Safetensors(format!("Tensor '{name}': {e}")))?;
        Ok(view.shape().to_vec())
    }

    /// Zero-copy view of a BF16 tensor as `&[u16]` (little-endian host).
    /// Requires 2-byte aligned offset inside the safetensors file (always true for standard HF exports).
    pub fn tensor_bf16(&self, name: &str) -> Result<&[u16], CoreError> {
        let shard_id = *self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Missing tensor '{name}'")))?;
        let mmap_bytes: &[u8] = &self.shards[shard_id].mmap;
        let st = SafeTensors::deserialize(mmap_bytes)
            .map_err(|e| CoreError::Safetensors(e.to_string()))?;
        let view = st
            .tensor(name)
            .map_err(|e| CoreError::Safetensors(format!("Tensor '{name}': {e}")))?;
        if view.dtype() != Dtype::BF16 {
            return Err(CoreError::Safetensors(format!(
                "Tensor '{name}' dtype is {:?}, expected BF16",
                view.dtype()
            )));
        }
        let data = view.data();
        if data.len() % 2 != 0 || (data.as_ptr() as usize) % 2 != 0 {
            return Err(CoreError::Safetensors(format!(
                "Tensor '{name}' BF16 data is not 2-byte aligned"
            )));
        }
        let slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u16, data.len() / 2)
        };
        Ok(slice)
    }

    /// Reads an F32 (or BF16->F32) tensor as `Cow<[f32]>`. Borrows zero-copy when F32 and 4-byte aligned.
    pub fn tensor_f32(&self, name: &str) -> Result<Cow<'_, [f32]>, CoreError> {
        let shard_id = *self
            .index
            .get(name)
            .ok_or_else(|| CoreError::Safetensors(format!("Missing tensor '{name}'")))?;
        let mmap_bytes: &[u8] = &self.shards[shard_id].mmap;
        let st = SafeTensors::deserialize(mmap_bytes)
            .map_err(|e| CoreError::Safetensors(e.to_string()))?;
        let view = st
            .tensor(name)
            .map_err(|e| CoreError::Safetensors(format!("Tensor '{name}': {e}")))?;
        let data = view.data();
        match view.dtype() {
            Dtype::F32 => {
                if data.len() % 4 != 0 {
                    return Err(CoreError::Safetensors(format!(
                        "Tensor '{name}' F32 byte length invalid"
                    )));
                }
                if (data.as_ptr() as usize) % 4 == 0 {
                    let slice = unsafe {
                        std::slice::from_raw_parts(data.as_ptr() as *const f32, data.len() / 4)
                    };
                    Ok(Cow::Borrowed(slice))
                } else {
                    let mut out = Vec::with_capacity(data.len() / 4);
                    for chunk in data.chunks_exact(4) {
                        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    Ok(Cow::Owned(out))
                }
            }
            Dtype::BF16 => {
                let mut out = Vec::with_capacity(data.len() / 2);
                for chunk in data.chunks_exact(2) {
                    let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                    out.push(f32::from_bits((u as u32) << 16));
                }
                Ok(Cow::Owned(out))
            }
            other => Err(CoreError::Safetensors(format!(
                "Tensor '{name}' dtype {other:?} cannot be read as F32"
            ))),
        }
    }

    pub fn tensor_bf16_opt(&self, name: &str) -> Result<Option<&[u16]>, CoreError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.tensor_bf16(name).map(Some)
    }

    pub fn tensor_f32_opt(&self, name: &str) -> Result<Option<Cow<'_, [f32]>>, CoreError> {
        if !self.contains(name) {
            return Ok(None);
        }
        self.tensor_f32(name).map(Some)
    }
}
