// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! El crate `qwen-core` reúne la configuración `0.6B`, el almacenamiento `mmap` de `safetensors`,
//! el tokenizador BPE, el muestreador determinista, los perfiles `.qvoice` v6 y el planificador de marcado.

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
    plan_script, resolve_sampling, MarkupMode, MarkupPlanner, PlannedScript, Script, SpanSpec,
};
pub use mmap_store::{Bf16Ref, F32Ref, MmapStore};
pub use qvoice::QVoiceFile;
pub use sampling::{LcgRng, Sampler, SamplingParams, SeedChoice};
pub use tokenizer::BpeTokenizer;
