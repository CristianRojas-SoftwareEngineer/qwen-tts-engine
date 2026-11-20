// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Núcleo de configuración `0.6B`, almacenamiento `mmap` de `safetensors`,
//! tokenizador BPE, muestreador determinista, perfiles `.qvoice` v4 y planificador de marcado.

pub mod config;
pub mod error;
pub mod markup;
pub mod mmap_store;
pub mod qvoice;
pub mod sampling;
pub mod tokenizer;

pub use config::ModelConfig06B;
pub use error::CoreError;
pub use markup::{
    is_paralinguistic_event_tag, plan_script, resolve_sampling, MarkupMode, MarkupPlanner,
    PlannedScript, Script, SpanSpec,
};
pub use mmap_store::{Bf16Ref, F32Ref, MmapStore};
pub use qvoice::QVoiceFile;
pub use sampling::{LcgRng, Sampler, SamplingParams, SeedChoice};
pub use tokenizer::BpeTokenizer;
