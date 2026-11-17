// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Núcleo de configuración `0.6B`, almacenamiento `mmap` de `safetensors`,
//! tokenizador BPE, muestreador determinista, perfiles `.qvoice`/`.bin` y planificador de marcado.

pub mod config;
pub mod error;
pub mod markup;
pub mod mmap_store;
pub mod qvoice;
pub mod sampling;
pub mod tokenizer;

pub use config::{lookup_language, lookup_speaker, ModelConfig06B, SpeakerPreset, PRESET_SPEAKERS};
pub use error::CoreError;
pub use markup::{emotion_name_to_tok, is_para_event_tag, MarkupPlanner, SpanSpec, VoiceClass};
pub use mmap_store::{Bf16Ref, F32Ref, MmapStore};
pub use qvoice::{
    decode_wdelta_bf16, encode_wdelta_bf16, read_bin_embedding, write_bin_embedding, QVoiceFile,
    QVoiceMeta, QVoiceTpad, QVoiceWeightTensor, QVoiceWeightsOverride, QVoiceWovr,
};
pub use sampling::{LcgRng, Sampler};
pub use tokenizer::BpeTokenizer;
