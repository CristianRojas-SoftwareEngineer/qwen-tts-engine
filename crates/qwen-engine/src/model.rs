// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use qwen_core::config::{
    CODEBOOK_DIM, CODEBOOK_SIZE, CODEC_VOCAB_SIZE, CP_HEAD_DIM, CP_HIDDEN_SIZE,
    CP_INTERMEDIATE_SIZE, CP_KV_DIM, CP_NUM_LAYERS, CP_Q_DIM, DEC_HEAD_DIM,
    DEC_NUM_LAYERS, DEC_ROPE_THETA, DEC_UPSAMPLE_CHANNELS, HEAD_DIM, HIDDEN_SIZE,
    INTERMEDIATE_SIZE, KV_DIM, NUM_CODEBOOKS, NUM_HEADS, NUM_LAYERS, Q_DIM, TEXT_HIDDEN_SIZE,
    TOKEN_TTS_BOS, TOKEN_TTS_EOS, TOKEN_TTS_PAD,
};
use qwen_core::{
    decode_wdelta_bf16, Bf16Ref, BpeTokenizer, F32Ref, LcgRng, MmapStore, ModelConfig06B,
    QVoiceWeightsOverride, Sampler,
};
use qwen_kernels::quant::{
    interleave_gate_up_bf16, quant_shape_check, quantize_bf16_to_int8,
    quantize_bf16_to_q4_0, quantize_gate_up_int8, quantize_gate_up_q4_0,
};
use qwen_kernels::{AlignedBuf, ComputeRuntime, KernelDispatch, Q2_0Block, Q4_0Block};

use crate::error::EngineError;
use crate::speech_encoder::SpeechEncoderWeights;
use crate::voice_clone::SpeakerEncoderWeights;

pub const KV_INITIAL_POSITIONS: usize = 1024;
pub const ROPE_MAX_POS: usize = 8192;
pub const CP_KV_MAX_POS: usize = 64;
pub const SD_ROPE_MAX_POS: usize = 8000;
pub const EMB_CACHE_CAPACITY: usize = 2048;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TalkerPrecision {
    Bf16,
    Int8,
    Int4,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum CpPrecision {
    Bf16,
    Int8,
    Int4,
    Mixed,
}

/// Opciones de carga y cuantización para [`Model06B::load`].
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelLoadOptions {
    pub silent: bool,
    pub debug: bool,
    pub use_int8: bool,
    pub use_int4: bool,
    pub quant_mixed: bool,
}

#[derive(Clone)]
pub enum Bf16Weight {
    Mmap(Bf16Ref),
    Owned(Arc<Vec<u16>>),
}

impl Bf16Weight {
    #[inline]
    pub fn as_slice(&self) -> &[u16] {
        match self {
            Bf16Weight::Mmap(r) => r.as_slice(),
            Bf16Weight::Owned(v) => v.as_slice(),
        }
    }
}

impl Deref for Bf16Weight {
    type Target = [u16];
    #[inline]
    fn deref(&self) -> &[u16] {
        self.as_slice()
    }
}

#[derive(Clone)]
pub enum F32Weight {
    Mmap(F32Ref),
    Owned(Arc<Vec<f32>>),
}

impl F32Weight {
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        match self {
            F32Weight::Mmap(r) => r.as_slice(),
            F32Weight::Owned(v) => v.as_slice(),
        }
    }
}

impl Deref for F32Weight {
    type Target = [f32];
    #[inline]
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

pub enum QuantWeight {
    Int8 {
        data: AlignedBuf<i8>,
        scale: AlignedBuf<f32>,
    },
    Q4_0(AlignedBuf<Q4_0Block>),
}

pub struct TalkerLayerWeights {
    pub wq_bf16: Bf16Weight,
    pub wk_bf16: Bf16Weight,
    pub wv_bf16: Bf16Weight,
    pub wo_bf16: Bf16Weight,
    pub q_norm: F32Weight,
    pub k_norm: F32Weight,
    pub input_norm: F32Weight,
    pub post_attn_norm: F32Weight,
    pub gate_bf16: Bf16Weight,
    pub up_bf16: Bf16Weight,
    pub down_bf16: Bf16Weight,
    /// Matriz intercalada `[2*inter, hidden]`; solo presente cuando el Talker corre en BF16.
    pub gate_up_fused_bf16: Option<AlignedBuf<u16>>,
    pub wq_q: Option<QuantWeight>,
    pub wk_q: Option<QuantWeight>,
    pub wv_q: Option<QuantWeight>,
    pub wo_q: Option<QuantWeight>,
    pub gate_up_q: Option<QuantWeight>,
    pub down_q: Option<QuantWeight>,
}

pub struct TalkerWeights {
    pub text_embedding_bf16: Bf16Weight,
    pub text_proj_fc1_bf16: Bf16Weight,
    pub text_proj_fc1_bias: Option<F32Weight>,
    pub text_proj_fc2_bf16: Bf16Weight,
    pub text_proj_fc2_bias: Option<F32Weight>,
    pub codec_embedding_bf16: Bf16Weight,
    pub codec_head_bf16: Bf16Weight,
    pub norm: F32Weight,
    pub layers: Vec<TalkerLayerWeights>,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
    pub precision: TalkerPrecision,
}

pub struct CpLayerWeights {
    pub wq_bf16: Bf16Weight,
    pub wk_bf16: Bf16Weight,
    pub wv_bf16: Bf16Weight,
    pub wo_bf16: Bf16Weight,
    pub q_norm: F32Weight,
    pub k_norm: F32Weight,
    pub input_norm: F32Weight,
    pub post_attn_norm: F32Weight,
    pub gate_bf16: Bf16Weight,
    pub up_bf16: Bf16Weight,
    pub down_bf16: Bf16Weight,
    /// Matriz intercalada `[2*cp_inter, cp_hidden]`; solo presente cuando el Code Predictor corre en BF16.
    pub gate_up_fused_bf16: Option<AlignedBuf<u16>>,
    pub wq_q: Option<QuantWeight>,
    pub wk_q: Option<QuantWeight>,
    pub wv_q: Option<QuantWeight>,
    pub wo_q: Option<QuantWeight>,
    pub gate_up_q: Option<QuantWeight>,
    pub down_q: Option<QuantWeight>,
}

impl CpLayerWeights {
    #[inline]
    pub fn all_quantized(&self) -> bool {
        self.wq_q.is_some()
            && self.wk_q.is_some()
            && self.wv_q.is_some()
            && self.wo_q.is_some()
            && self.gate_up_q.is_some()
            && self.down_q.is_some()
    }
}

pub struct CpLmHead {
    pub bf16: Bf16Weight,
    pub quant: Option<QuantWeight>,
}

pub struct CodePredictorWeights {
    pub norm: F32Weight,
    pub layers: Vec<CpLayerWeights>,
    pub codec_emb_bf16: Vec<Bf16Weight>,
    pub lm_heads: Vec<CpLmHead>,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
    pub precision: CpPrecision,
    pub all_quantized: bool,
}

impl CodePredictorWeights {
    pub fn prec_label(&self) -> &'static str {
        let mut all8 = true;
        let mut all4 = true;
        let mut mix = true;
        for l in &self.layers {
            let att8 = matches!(l.wq_q, Some(QuantWeight::Int8 { .. }))
                && matches!(l.wk_q, Some(QuantWeight::Int8 { .. }))
                && matches!(l.wv_q, Some(QuantWeight::Int8 { .. }))
                && matches!(l.wo_q, Some(QuantWeight::Int8 { .. }));
            let ffn8 = matches!(l.gate_up_q, Some(QuantWeight::Int8 { .. }))
                && matches!(l.down_q, Some(QuantWeight::Int8 { .. }));
            let att4 = matches!(l.wq_q, Some(QuantWeight::Q4_0(_)))
                && matches!(l.wk_q, Some(QuantWeight::Q4_0(_)))
                && matches!(l.wv_q, Some(QuantWeight::Q4_0(_)))
                && matches!(l.wo_q, Some(QuantWeight::Q4_0(_)));
            let ffn4 = matches!(l.gate_up_q, Some(QuantWeight::Q4_0(_)))
                && matches!(l.down_q, Some(QuantWeight::Q4_0(_)));
            if !(att8 && ffn8) {
                all8 = false;
            }
            if !(att4 && ffn4) {
                all4 = false;
            }
            if !(att8 && ffn4) {
                mix = false;
            }
        }
        if all8 {
            "INT8"
        } else if all4 {
            "Q4_0"
        } else if mix {
            "INT8+Q4_0-FFN"
        } else {
            "BF16"
        }
    }
}

pub struct SdPreLayerWeights {
    pub attn_norm: F32Ref,
    pub attn_q: F32Ref,
    pub attn_k: F32Ref,
    pub attn_v: F32Ref,
    pub attn_o: F32Ref,
    pub attn_layer_scale: Option<F32Ref>,
    pub ffn_norm: F32Ref,
    pub ffn_gate: F32Ref,
    pub ffn_up: F32Ref,
    pub ffn_down: F32Ref,
    pub ffn_layer_scale: Option<F32Ref>,
}

pub struct SdConvNeXtWeights {
    pub conv_weight: F32Ref,
    pub conv_bias: Option<F32Ref>,
    pub dwconv_weight: F32Ref,
    pub dwconv_bias: Option<F32Ref>,
    pub pwconv1_weight: F32Ref,
    pub pwconv1_bias: Option<F32Ref>,
    pub pwconv2_weight: F32Ref,
    pub pwconv2_bias: Option<F32Ref>,
    pub norm_weight: F32Ref,
    pub norm_bias: F32Ref,
    pub gamma: F32Ref,
}

pub struct SdResBlockWeights {
    pub snake1_alpha: Option<F32Ref>,
    pub snake1_beta: Option<F32Ref>,
    pub conv1_weight: F32Ref,
    pub conv1_bias: Option<F32Ref>,
    pub conv1_wk: AlignedBuf<f32>,
    pub snake2_alpha: Option<F32Ref>,
    pub snake2_beta: Option<F32Ref>,
    pub conv2_weight: F32Ref,
    pub conv2_bias: Option<F32Ref>,
}

pub struct SdUpsampleBlockWeights {
    pub snake_alpha: Option<F32Ref>,
    pub snake_beta: Option<F32Ref>,
    pub conv_weight: F32Ref,
    pub conv_bias: Option<F32Ref>,
    pub res_blocks: [SdResBlockWeights; 3],
}

pub struct SpeechDecoderWeights {
    pub codebooks: Arc<[AlignedBuf<f32>; NUM_CODEBOOKS]>,
    pub rvq_first_output_proj: F32Ref,
    pub rvq_rest_output_proj: F32Ref,
    pub pre_conv_weight: F32Ref,
    pub pre_conv_bias: Option<F32Ref>,
    pub pre_conv_wk: AlignedBuf<f32>,
    pub input_proj_weight: F32Ref,
    pub input_proj_bias: Option<F32Ref>,
    pub pre_layers: Vec<SdPreLayerWeights>,
    pub final_norm_weight: Option<F32Ref>,
    pub output_proj_weight: F32Ref,
    pub output_proj_bias: Option<F32Ref>,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
    pub convnext: [SdConvNeXtWeights; 2],
    pub initial_conv_weight: F32Ref,
    pub initial_conv_bias: Option<F32Ref>,
    pub initial_conv_wk: AlignedBuf<f32>,
    pub upsample_blocks: [SdUpsampleBlockWeights; 4],
    pub final_snake_alpha: F32Ref,
    pub final_snake_beta: F32Ref,
    pub final_conv_weight: F32Ref,
    pub final_conv_bias: Option<F32Ref>,
}

/// Modelo Qwen3-TTS `0.6B` (`CustomVoice` o `Base`) inmutable y compartible entre hilos (`Send + Sync`).
pub struct Model06B {
    pub config: ModelConfig06B,
    pub tokenizer: BpeTokenizer,
    pub talker: TalkerWeights,
    pub cp: CodePredictorWeights,
    pub decoder: SpeechDecoderWeights,
    pub speaker_encoder: Option<Arc<SpeakerEncoderWeights>>,
    pub speech_encoder: Option<Arc<SpeechEncoderWeights>>,
    pub cached_tts_pad_embed: AlignedBuf<f32>,
    pub cached_tts_bos_embed: AlignedBuf<f32>,
    pub cached_tts_eos_embed: AlignedBuf<f32>,
    pub ryan_embed_norm: f32,
}

#[derive(Debug, Clone)]
pub enum VoiceIdentity {
    Preset(i32),
    XVector(Vec<f32>),
    IclGraft {
        speaker_embedding: Vec<f32>,
        ref_text: String,
        ref_codes: Vec<i32>,
    },
}

#[derive(Clone)]
pub struct TextProjOverride {
    pub fc1_bf16: Vec<u16>,
    pub fc1_bias: Vec<f32>,
    pub fc2_bf16: Vec<u16>,
    pub fc2_bias: Vec<f32>,
    pub codec_emb_0_2048_bf16: Vec<u16>,
    pub tts_pad_embed: AlignedBuf<f32>,
    pub tts_bos_embed: AlignedBuf<f32>,
    pub tts_eos_embed: AlignedBuf<f32>,
}

/// Vínculos de voz inmutables por locutor (preset, x-vector o injerto ICL + `WOVR`/`emo_ref`).
#[derive(Clone)]
pub struct VoiceBindings {
    pub identity: VoiceIdentity,
    pub wovr: Option<TextProjOverride>,
    pub emo_ref_text: Option<String>,
    pub emo_ref_codes: Option<Vec<i32>>,
    pub icl_frames_cap: usize,
    pub graft_mode: bool,
}

impl VoiceBindings {
    pub fn preset(speaker_id: i32) -> Self {
        Self {
            identity: VoiceIdentity::Preset(speaker_id),
            wovr: None,
            emo_ref_text: None,
            emo_ref_codes: None,
            icl_frames_cap: 0,
            graft_mode: false,
        }
    }
}

pub struct TalkerKvCache {
    pub k: AlignedBuf<u16>,
    pub v: AlignedBuf<u16>,
    pub max_pos: usize,
    pub len: usize,
}

impl TalkerKvCache {
    pub fn new() -> Self {
        let cap = NUM_LAYERS * KV_INITIAL_POSITIONS * KV_DIM;
        Self {
            k: AlignedBuf::new_zeroed(cap),
            v: AlignedBuf::new_zeroed(cap),
            max_pos: KV_INITIAL_POSITIONS,
            len: 0,
        }
    }

    pub fn ensure_capacity(&mut self, required: usize) {
        if required <= self.max_pos {
            return;
        }
        let mut new_max = self.max_pos;
        while new_max < required {
            new_max *= 2;
        }
        let total = NUM_LAYERS * new_max * KV_DIM;
        let mut new_k = AlignedBuf::new_zeroed(total);
        let mut new_v = AlignedBuf::new_zeroed(total);
        for layer in 0..NUM_LAYERS {
            let old_off = layer * self.max_pos * KV_DIM;
            let new_off = layer * new_max * KV_DIM;
            let copy_len = self.len * KV_DIM;
            new_k[new_off..new_off + copy_len]
                .copy_from_slice(&self.k[old_off..old_off + copy_len]);
            new_v[new_off..new_off + copy_len]
                .copy_from_slice(&self.v[old_off..old_off + copy_len]);
        }
        self.k = new_k;
        self.v = new_v;
        self.max_pos = new_max;
    }
}

impl Default for TalkerKvCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CpKvCache {
    pub k: AlignedBuf<u16>,
    pub v: AlignedBuf<u16>,
    pub max_pos: usize,
    pub len: usize,
}

impl CpKvCache {
    pub fn new() -> Self {
        let cap = CP_NUM_LAYERS * CP_KV_MAX_POS * CP_KV_DIM;
        Self {
            k: AlignedBuf::new_zeroed(cap),
            v: AlignedBuf::new_zeroed(cap),
            max_pos: CP_KV_MAX_POS,
            len: 0,
        }
    }
}

impl Default for CpKvCache {
    fn default() -> Self {
        Self::new()
    }
}

pub struct TalkerBuffers {
    pub dec_x: AlignedBuf<f32>,
    pub dec_x_norm: AlignedBuf<f32>,
    pub dec_q: AlignedBuf<f32>,
    pub dec_k: AlignedBuf<f32>,
    pub dec_v: AlignedBuf<f32>,
    pub dec_attn_out: AlignedBuf<f32>,
    pub dec_proj_out: AlignedBuf<f32>,
    pub dec_gate: AlignedBuf<f32>,
    pub dec_ffn_out: AlignedBuf<f32>,
    pub swiglu_tmp: AlignedBuf<f32>,
    pub pref_residual: AlignedBuf<f32>,
    pub pref_x_norm: AlignedBuf<f32>,
    pub pref_q: AlignedBuf<f32>,
    pub pref_k: AlignedBuf<f32>,
    pub pref_v: AlignedBuf<f32>,
    pub pref_attn_out: AlignedBuf<f32>,
    pub pref_gate: AlignedBuf<f32>,
    pub pref_proj: AlignedBuf<f32>,
    pub pref_seq_cap: usize,
    pub pref_wq_f32: AlignedBuf<f32>,
    pub pref_wk_f32: AlignedBuf<f32>,
    pub pref_wv_f32: AlignedBuf<f32>,
    pub pref_wo_f32: AlignedBuf<f32>,
    pub pref_gate_up_f32: AlignedBuf<f32>,
    pub pref_down_f32: AlignedBuf<f32>,
    pub emb_tmp1: AlignedBuf<f32>,
    pub emb_tmp2: AlignedBuf<f32>,
    pub logits: AlignedBuf<f32>,
    pub step_embed: AlignedBuf<f32>,
    pub last_hidden: AlignedBuf<f32>,
}

impl TalkerBuffers {
    pub fn new() -> Self {
        Self {
            dec_x: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            dec_x_norm: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            dec_q: AlignedBuf::new_zeroed(Q_DIM),
            dec_k: AlignedBuf::new_zeroed(KV_DIM),
            dec_v: AlignedBuf::new_zeroed(KV_DIM),
            dec_attn_out: AlignedBuf::new_zeroed(Q_DIM),
            dec_proj_out: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            dec_gate: AlignedBuf::new_zeroed(2 * INTERMEDIATE_SIZE),
            dec_ffn_out: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            swiglu_tmp: AlignedBuf::new_zeroed(INTERMEDIATE_SIZE.max(CP_INTERMEDIATE_SIZE)),
            pref_residual: AlignedBuf::new_zeroed(0),
            pref_x_norm: AlignedBuf::new_zeroed(0),
            pref_q: AlignedBuf::new_zeroed(0),
            pref_k: AlignedBuf::new_zeroed(0),
            pref_v: AlignedBuf::new_zeroed(0),
            pref_attn_out: AlignedBuf::new_zeroed(0),
            pref_gate: AlignedBuf::new_zeroed(0),
            pref_proj: AlignedBuf::new_zeroed(0),
            pref_seq_cap: 0,
            pref_wq_f32: AlignedBuf::new_zeroed(Q_DIM * HIDDEN_SIZE),
            pref_wk_f32: AlignedBuf::new_zeroed(KV_DIM * HIDDEN_SIZE),
            pref_wv_f32: AlignedBuf::new_zeroed(KV_DIM * HIDDEN_SIZE),
            pref_wo_f32: AlignedBuf::new_zeroed(HIDDEN_SIZE * Q_DIM),
            pref_gate_up_f32: AlignedBuf::new_zeroed(2 * INTERMEDIATE_SIZE * HIDDEN_SIZE),
            pref_down_f32: AlignedBuf::new_zeroed(HIDDEN_SIZE * INTERMEDIATE_SIZE),
            emb_tmp1: AlignedBuf::new_zeroed(TEXT_HIDDEN_SIZE),
            emb_tmp2: AlignedBuf::new_zeroed(TEXT_HIDDEN_SIZE),
            logits: AlignedBuf::new_zeroed(CODEC_VOCAB_SIZE),
            step_embed: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            last_hidden: AlignedBuf::new_zeroed(HIDDEN_SIZE),
        }
    }

    pub fn ensure_prefill_capacity(&mut self, seq_len: usize) {
        if seq_len <= self.pref_seq_cap {
            return;
        }
        self.pref_residual = AlignedBuf::new_zeroed(seq_len * HIDDEN_SIZE);
        self.pref_x_norm = AlignedBuf::new_zeroed(seq_len * HIDDEN_SIZE);
        self.pref_q = AlignedBuf::new_zeroed(seq_len * Q_DIM);
        self.pref_k = AlignedBuf::new_zeroed(seq_len * KV_DIM);
        self.pref_v = AlignedBuf::new_zeroed(seq_len * KV_DIM);
        self.pref_attn_out = AlignedBuf::new_zeroed(seq_len * Q_DIM);
        self.pref_gate = AlignedBuf::new_zeroed(seq_len * 2 * INTERMEDIATE_SIZE);
        self.pref_proj = AlignedBuf::new_zeroed(seq_len * HIDDEN_SIZE);
        self.pref_seq_cap = seq_len;
    }
}

impl Default for TalkerBuffers {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CpBuffers {
    pub dec_x: AlignedBuf<f32>,
    pub dec_q: AlignedBuf<f32>,
    pub dec_k: AlignedBuf<f32>,
    pub dec_v: AlignedBuf<f32>,
    pub dec_attn_out: AlignedBuf<f32>,
    pub dec_gate: AlignedBuf<f32>,
    pub dec_ffn_out: AlignedBuf<f32>,
    pub x1in: AlignedBuf<f32>,
    pub emb_buf: AlignedBuf<f32>,
    // Scratch preasignado para prefill B=2
    pub p2_x2: AlignedBuf<f32>,
    pub p2_q2: AlignedBuf<f32>,
    pub p2_k2: AlignedBuf<f32>,
    pub p2_v2: AlignedBuf<f32>,
    pub p2_attn0: AlignedBuf<f32>,
    pub p2_a2: AlignedBuf<f32>,
    pub p2_p2: AlignedBuf<f32>,
    pub p2_g2: AlignedBuf<f32>,
    pub p2_g0: AlignedBuf<f32>,
    pub p2_gi2: AlignedBuf<f32>,
    pub p2_d2: AlignedBuf<f32>,
    pub p2_xn0: AlignedBuf<f32>,
    pub p2_xn1: AlignedBuf<f32>,
    pub down_q2_rough: Option<Vec<AlignedBuf<Q2_0Block>>>,
}

impl CpBuffers {
    pub fn new() -> Self {
        Self {
            dec_x: AlignedBuf::new_zeroed(CP_HIDDEN_SIZE),
            dec_q: AlignedBuf::new_zeroed(CP_Q_DIM),
            dec_k: AlignedBuf::new_zeroed(CP_KV_DIM),
            dec_v: AlignedBuf::new_zeroed(CP_KV_DIM),
            dec_attn_out: AlignedBuf::new_zeroed(CP_Q_DIM),
            dec_gate: AlignedBuf::new_zeroed(2 * CP_INTERMEDIATE_SIZE),
            dec_ffn_out: AlignedBuf::new_zeroed(CP_HIDDEN_SIZE),
            x1in: AlignedBuf::new_zeroed(CP_HIDDEN_SIZE),
            emb_buf: AlignedBuf::new_zeroed(HIDDEN_SIZE),
            p2_x2: AlignedBuf::new_zeroed(2 * CP_HIDDEN_SIZE),
            p2_q2: AlignedBuf::new_zeroed(2 * CP_Q_DIM),
            p2_k2: AlignedBuf::new_zeroed(2 * CP_KV_DIM),
            p2_v2: AlignedBuf::new_zeroed(2 * CP_KV_DIM),
            p2_attn0: AlignedBuf::new_zeroed(CP_Q_DIM),
            p2_a2: AlignedBuf::new_zeroed(2 * CP_Q_DIM),
            p2_p2: AlignedBuf::new_zeroed(2 * CP_HIDDEN_SIZE),
            p2_g2: AlignedBuf::new_zeroed(4 * CP_INTERMEDIATE_SIZE),
            p2_g0: AlignedBuf::new_zeroed(2 * CP_INTERMEDIATE_SIZE),
            p2_gi2: AlignedBuf::new_zeroed(2 * CP_INTERMEDIATE_SIZE),
            p2_d2: AlignedBuf::new_zeroed(2 * CP_HIDDEN_SIZE),
            p2_xn0: AlignedBuf::new_zeroed(CP_HIDDEN_SIZE),
            p2_xn1: AlignedBuf::new_zeroed(CP_HIDDEN_SIZE),
            down_q2_rough: None,
        }
    }
}

impl Default for CpBuffers {
    fn default() -> Self {
        Self::new()
    }
}

/// 8 ranuras monótonas reutilizables para el decodificador convolucional acústico.
pub struct DecoderWorkspace {
    pub a: AlignedBuf<f32>,
    pub b: AlignedBuf<f32>,
    pub res: AlignedBuf<f32>,
    pub ext: AlignedBuf<f32>,
    pub full: AlignedBuf<f32>,
    pub ct: AlignedBuf<f32>,
    pub xp: AlignedBuf<f32>,
    pub pw1: AlignedBuf<f32>,
}

impl DecoderWorkspace {
    pub fn new() -> Self {
        Self {
            a: AlignedBuf::new_zeroed(0),
            b: AlignedBuf::new_zeroed(0),
            res: AlignedBuf::new_zeroed(0),
            ext: AlignedBuf::new_zeroed(0),
            full: AlignedBuf::new_zeroed(0),
            ct: AlignedBuf::new_zeroed(0),
            xp: AlignedBuf::new_zeroed(0),
            pw1: AlignedBuf::new_zeroed(0),
        }
    }
}

impl Default for DecoderWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SdStreamState {
    pub k_cache: [Vec<f32>; DEC_NUM_LAYERS],
    pub v_cache: [Vec<f32>; DEC_NUM_LAYERS],
    pub kv_len: usize,
    pub kv_alloc: usize,
    pub kv_base: usize,
    pub latent_cache: Vec<f32>,
    pub latent_frames: usize,
    pub latent_alloc: usize,
    pub latent_base: usize,
    pub vq_pad: Vec<f32>,
    pub vq_pad_valid: bool,
    pub cs_cn_dw_tail: [AlignedBuf<f32>; 2],
    pub cs_init_tail: AlignedBuf<f32>,
    pub cs_up_carry: [AlignedBuf<f32>; 4],
    pub cs_res_tail: [[AlignedBuf<f32>; 3]; 4],
    pub cs_final_tail: AlignedBuf<f32>,
    pub cs_alloc: bool,
    pub cs_warm: bool,
    pub cs_tile: bool,
    pub frames_decoded: usize,
    pub samples_produced: usize,
    // Buffers scratch reutilizables por fragmento (cero asignaciones de heap durante streaming)
    pub tmp_vq_out: Vec<f32>,
    pub tmp_vq_cf: Vec<f32>,
    pub tmp_pre_conv_out: Vec<f32>,
    pub tmp_pre_conv_rm: Vec<f32>,
    pub tmp_hidden: Vec<f32>,
    pub tmp_q: Vec<f32>,
    pub tmp_new_k: Vec<f32>,
    pub tmp_new_v: Vec<f32>,
    pub tmp_x_norm: Vec<f32>,
    pub tmp_attn_out: Vec<f32>,
    pub tmp_ffn_gate: Vec<f32>,
    pub tmp_ffn_up: Vec<f32>,
    pub tmp_ffn_down: Vec<f32>,
    pub tmp_scores: Vec<f32>,
}

impl SdStreamState {
    pub fn new() -> Self {
        Self {
            k_cache: std::array::from_fn(|_| Vec::new()),
            v_cache: std::array::from_fn(|_| Vec::new()),
            kv_len: 0,
            kv_alloc: 0,
            kv_base: 0,
            latent_cache: Vec::new(),
            latent_frames: 0,
            latent_alloc: 0,
            latent_base: 0,
            vq_pad: Vec::new(),
            vq_pad_valid: false,
            cs_cn_dw_tail: [AlignedBuf::new_zeroed(0), AlignedBuf::new_zeroed(0)],
            cs_init_tail: AlignedBuf::new_zeroed(0),
            cs_up_carry: std::array::from_fn(|_| AlignedBuf::new_zeroed(0)),
            cs_res_tail: std::array::from_fn(|_| {
                std::array::from_fn(|_| AlignedBuf::new_zeroed(0))
            }),
            cs_final_tail: AlignedBuf::new_zeroed(0),
            cs_alloc: false,
            cs_warm: false,
            cs_tile: false,
            frames_decoded: 0,
            samples_produced: 0,
            tmp_vq_out: Vec::new(),
            tmp_vq_cf: Vec::new(),
            tmp_pre_conv_out: Vec::new(),
            tmp_pre_conv_rm: Vec::new(),
            tmp_hidden: Vec::new(),
            tmp_q: Vec::new(),
            tmp_new_k: Vec::new(),
            tmp_new_v: Vec::new(),
            tmp_x_norm: Vec::new(),
            tmp_attn_out: Vec::new(),
            tmp_ffn_gate: Vec::new(),
            tmp_ffn_up: Vec::new(),
            tmp_ffn_down: Vec::new(),
            tmp_scores: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
        self.kv_base = 0;
        self.latent_frames = 0;
        self.latent_base = 0;
        self.vq_pad_valid = false;
        self.cs_warm = false;
        self.cs_tile = false;
        self.frames_decoded = 0;
        self.samples_produced = 0;
        if self.cs_alloc {
            for t in &mut self.cs_cn_dw_tail {
                t.fill_zero();
            }
            self.cs_init_tail.fill_zero();
            for c in &mut self.cs_up_carry {
                c.fill_zero();
            }
            for b in &mut self.cs_res_tail {
                for r in b {
                    r.fill_zero();
                }
            }
            self.cs_final_tail.fill_zero();
        }
    }

    pub fn ensure_conv_state(&mut self) {
        if self.cs_alloc {
            return;
        }
        const UP_RATE: [usize; 4] = [8, 5, 4, 3];
        const DILS: [usize; 3] = [1, 3, 9];
        self.cs_cn_dw_tail = [
            AlignedBuf::new_zeroed(1024 * 6),
            AlignedBuf::new_zeroed(1024 * 6),
        ];
        self.cs_init_tail = AlignedBuf::new_zeroed(1024 * 6);
        self.cs_final_tail = AlignedBuf::new_zeroed(96 * 6);
        for b in 0..4 {
            self.cs_up_carry[b] = AlignedBuf::new_zeroed(DEC_UPSAMPLE_CHANNELS[b] * UP_RATE[b]);
            for r in 0..3 {
                self.cs_res_tail[b][r] =
                    AlignedBuf::new_zeroed(DEC_UPSAMPLE_CHANNELS[b] * 6 * DILS[r]);
            }
        }
        self.cs_alloc = true;
    }
}

impl Default for SdStreamState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EmbLruCache {
    keys: Vec<i32>,
    values: AlignedBuf<f32>,
    access: Vec<u32>,
    capacity: usize,
    clock: u32,
}

impl EmbLruCache {
    pub fn new() -> Self {
        Self {
            keys: vec![-1; EMB_CACHE_CAPACITY],
            values: AlignedBuf::new_zeroed(EMB_CACHE_CAPACITY * HIDDEN_SIZE),
            access: vec![0; EMB_CACHE_CAPACITY],
            capacity: EMB_CACHE_CAPACITY,
            clock: 0,
        }
    }

    pub fn clear(&mut self) {
        self.keys.fill(-1);
        self.access.fill(0);
        self.clock = 0;
    }

    pub fn get_or_insert<F>(&mut self, tid: i32, out: &mut [f32], mut compute: F)
    where
        F: FnMut(i32, &mut [f32]),
    {
        let cap = self.capacity;
        let mask = cap - 1;
        let idx = ((tid as u32).wrapping_mul(2_654_435_761)) as usize & mask;
        for probe in 0..cap {
            let slot = (idx + probe) & mask;
            if self.keys[slot] == tid {
                self.clock = self.clock.wrapping_add(1);
                self.access[slot] = self.clock;
                let base = slot * HIDDEN_SIZE;
                out[..HIDDEN_SIZE].copy_from_slice(&self.values[base..base + HIDDEN_SIZE]);
                return;
            }
            if self.keys[slot] == -1 {
                self.keys[slot] = tid;
                self.clock = self.clock.wrapping_add(1);
                self.access[slot] = self.clock;
                let base = slot * HIDDEN_SIZE;
                compute(tid, &mut self.values[base..base + HIDDEN_SIZE]);
                out[..HIDDEN_SIZE].copy_from_slice(&self.values[base..base + HIDDEN_SIZE]);
                return;
            }
        }
        let mut victim = 0usize;
        let mut min_acc = u32::MAX;
        for (i, &acc) in self.access.iter().enumerate() {
            if acc < min_acc {
                min_acc = acc;
                victim = i;
            }
        }
        self.keys[victim] = tid;
        self.clock = self.clock.wrapping_add(1);
        self.access[victim] = self.clock;
        let base = victim * HIDDEN_SIZE;
        compute(tid, &mut self.values[base..base + HIDDEN_SIZE]);
        out[..HIDDEN_SIZE].copy_from_slice(&self.values[base..base + HIDDEN_SIZE]);
    }
}

impl Default for EmbLruCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Estado mutable exclusivo de una sesión o worker (KV caches, buffers de trabajo, muestreador y RNG).
pub struct SessionScratch {
    pub talker_kv: TalkerKvCache,
    pub cp_kv: CpKvCache,
    pub talker_bufs: TalkerBuffers,
    pub cp_bufs: CpBuffers,
    pub sd_stream: SdStreamState,
    pub sd_ws: DecoderWorkspace,
    pub emb_lru: EmbLruCache,
    pub sampler: Sampler,
    pub rng: LcgRng,
    pub codec_codes: Vec<i32>,
    pub prev_tokens: Vec<i32>,
}

impl SessionScratch {
    pub fn new() -> Self {
        Self {
            talker_kv: TalkerKvCache::new(),
            cp_kv: CpKvCache::new(),
            talker_bufs: TalkerBuffers::new(),
            cp_bufs: CpBuffers::new(),
            sd_stream: SdStreamState::new(),
            sd_ws: DecoderWorkspace::new(),
            emb_lru: EmbLruCache::new(),
            sampler: Sampler::new(),
            rng: LcgRng::new(0),
            codec_codes: Vec::new(),
            prev_tokens: Vec::new(),
        }
    }
}

impl Default for SessionScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Parámetros de muestreo, audio y diagnóstico para una petición de síntesis.
#[derive(Debug, Clone)]
pub struct RequestConfig {
    pub speaker_id: i32,
    pub language_id: Option<i32>,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub rep_penalty: f32,
    pub max_tokens: usize,
    pub greedy_warmup: usize,
    pub seed: u32,
    pub stream: bool,
    pub stream_chunk_frames: usize,
    pub volume: f32,
    pub rate: f32,
    pub silent: bool,
    pub debug: bool,
    pub cp_roughness: f32,
}

impl Default for RequestConfig {
    fn default() -> Self {
        Self {
            speaker_id: 3061, // ryan
            language_id: None,
            temperature: 0.9,
            top_k: 50,
            top_p: 1.0,
            rep_penalty: 1.05,
            max_tokens: 8192,
            greedy_warmup: 0,
            seed: 42,
            stream: false,
            stream_chunk_frames: 10,
            volume: 1.0,
            rate: 1.0,
            silent: false,
            debug: false,
            cp_roughness: 0.0,
        }
    }
}

type AbortPollFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Señal cooperativa de cancelación por bandera atómica y función opcional de sondeo.
#[derive(Clone, Default)]
pub struct AbortHandle {
    cancelled: Arc<AtomicBool>,
    poll: Option<AbortPollFn>,
}

impl AbortHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_poll<F>(poll: F) -> Self
    where
        F: Fn() -> bool + Send + Sync + 'static,
    {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            poll: Some(Arc::new(poll)),
        }
    }

    #[inline]
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    #[inline]
    pub fn is_cancelled(&self) -> bool {
        if self.cancelled.load(Ordering::Acquire) {
            return true;
        }
        if let Some(poll) = &self.poll {
            if poll() {
                self.cancelled.store(true, Ordering::Release);
                return true;
            }
        }
        false
    }
}

/// Contexto efímero de ejecución de una petición (configuración + [`AbortHandle`]).
pub struct RequestExecution {
    pub config: RequestConfig,
    pub abort: AbortHandle,
}

impl RequestExecution {
    pub fn new(config: RequestConfig) -> Self {
        Self {
            config,
            abort: AbortHandle::new(),
        }
    }
}

/// Métricas de latencia y conteo de fotogramas producidas por una síntesis.
#[derive(Debug, Clone, Default)]
pub struct SynthesisTiming {
    pub prefill_ms: f64,
    pub talker_ms: f64,
    pub cp_ms: f64,
    pub decoder_ms: f64,
    pub drain_ms: f64,
    pub frames: usize,
    pub chunks: usize,
    pub max_backlog_frames: usize,
    pub ttfa_ms: Option<f64>,
    pub total_ms: f64,
    pub samples_produced: usize,
}

fn req_bf16(store: &MmapStore, name: &str) -> Result<Bf16Weight, EngineError> {
    Ok(Bf16Weight::Mmap(store.tensor_bf16_ref(name)?))
}

fn req_f32_w(store: &MmapStore, name: &str) -> Result<F32Weight, EngineError> {
    Ok(F32Weight::Mmap(store.tensor_f32_ref(name)?))
}

fn opt_f32_w(store: &MmapStore, name: &str) -> Result<Option<F32Weight>, EngineError> {
    Ok(store.tensor_f32_ref_opt(name)?.map(F32Weight::Mmap))
}

fn req_f32(store: &MmapStore, name: &str) -> Result<F32Ref, EngineError> {
    Ok(store.tensor_f32_ref(name)?)
}

fn opt_f32(store: &MmapStore, name: &str) -> Result<Option<F32Ref>, EngineError> {
    Ok(store.tensor_f32_ref_opt(name)?)
}

fn quantize_matrix_int8(src: &[u16], rows: usize, cols: usize) -> QuantWeight {
    let (data, scale) = quantize_bf16_to_int8(src, rows, cols);
    QuantWeight::Int8 { data, scale }
}

fn quantize_matrix_q4(src: &[u16], rows: usize, cols: usize) -> QuantWeight {
    QuantWeight::Q4_0(quantize_bf16_to_q4_0(src, rows, cols))
}

fn conv_reorder_weight(
    weight: &[f32],
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) -> AlignedBuf<f32> {
    let mut wr = AlignedBuf::new_zeroed(kernel * out_ch * in_ch);
    for k in 0..kernel {
        for oc in 0..out_ch {
            for ic in 0..in_ch {
                wr[(k * out_ch + oc) * in_ch + ic] =
                    weight[(oc * in_ch + ic) * kernel + k];
            }
        }
    }
    wr
}

pub fn embed_one_text_token_compute(
    dispatch: &KernelDispatch,
    pool: &qwen_kernels::ComputePool,
    model: &Model06B,
    voice: &VoiceBindings,
    tid: i32,
    tmp1: &mut [f32],
    tmp2: &mut [f32],
    out: &mut [f32],
) {
    let th = TEXT_HIDDEN_SIZE;
    let h = HIDDEN_SIZE;
    let idx = tid.max(0) as usize;
    let emb_row = &model.talker.text_embedding_bf16[idx * th..(idx + 1) * th];
    dispatch.bf16_to_f32_vec(&mut tmp1[..th], emb_row);

    let (fc1_w, fc1_b, fc2_w, fc2_b) = if let Some(wovr) = &voice.wovr {
        (
            wovr.fc1_bf16.as_slice(),
            Some(wovr.fc1_bias.as_slice()),
            wovr.fc2_bf16.as_slice(),
            Some(wovr.fc2_bias.as_slice()),
        )
    } else {
        (
            model.talker.text_proj_fc1_bf16.as_slice(),
            model.talker.text_proj_fc1_bias.as_deref(),
            model.talker.text_proj_fc2_bf16.as_slice(),
            model.talker.text_proj_fc2_bias.as_deref(),
        )
    };

    dispatch.matvec_bf16(pool, &mut tmp2[..th], fc1_w, &tmp1[..th], th, th);
    if let Some(b) = fc1_b {
        for j in 0..th {
            tmp2[j] += b[j];
        }
    }
    dispatch.silu(&mut tmp2[..th]);
    dispatch.matvec_bf16(pool, &mut out[..h], fc2_w, &tmp2[..th], h, th);
    if let Some(b) = fc2_b {
        for j in 0..h {
            out[j] += b[j];
        }
    }
}

pub fn lookup_codec_embed(
    dispatch: &KernelDispatch,
    model: &Model06B,
    voice: &VoiceBindings,
    token_id: i32,
    out: &mut [f32],
) {
    if token_id < 0 || (token_id as usize) >= CODEC_VOCAB_SIZE {
        out[..HIDDEN_SIZE].fill(0.0);
        return;
    }
    let tid = token_id as usize;
    let row = if let Some(wovr) = &voice.wovr {
        let wcv = wovr.codec_emb_0_2048_bf16.len() / HIDDEN_SIZE;
        if tid < wcv {
            &wovr.codec_emb_0_2048_bf16[tid * HIDDEN_SIZE..(tid + 1) * HIDDEN_SIZE]
        } else {
            &model.talker.codec_embedding_bf16[tid * HIDDEN_SIZE..(tid + 1) * HIDDEN_SIZE]
        }
    } else {
        &model.talker.codec_embedding_bf16[tid * HIDDEN_SIZE..(tid + 1) * HIDDEN_SIZE]
    };
    dispatch.bf16_to_f32_vec(&mut out[..HIDDEN_SIZE], row);
}

fn resolve_talker_precision(opts: ModelLoadOptions) -> Result<TalkerPrecision, EngineError> {
    if let Ok(val) = std::env::var("QWEN_TALKER_PREC") {
        let v = val.trim();
        if !v.is_empty() {
            return match v {
                "bf16" => Ok(TalkerPrecision::Bf16),
                "int8" => Ok(TalkerPrecision::Int8),
                "int4" => Ok(TalkerPrecision::Int4),
                _ => Err(EngineError::InvalidArg(format!(
                    "error: QWEN_TALKER_PREC={v} no es válido; use bf16, int8 o int4"
                ))),
            };
        }
    }
    if opts.quant_mixed || opts.use_int4 {
        Ok(TalkerPrecision::Int4)
    } else if opts.use_int8 {
        Ok(TalkerPrecision::Int8)
    } else {
        Ok(TalkerPrecision::Bf16)
    }
}

fn resolve_cp_precision(opts: ModelLoadOptions) -> Result<CpPrecision, EngineError> {
    if let Ok(val) = std::env::var("QWEN_CP_PREC") {
        let v = val.trim();
        if !v.is_empty() {
            return match v {
                "bf16" => Ok(CpPrecision::Bf16),
                "int8" => Ok(CpPrecision::Int8),
                "int4" => Ok(CpPrecision::Int4),
                "mixed" => Ok(CpPrecision::Mixed),
                _ => Err(EngineError::InvalidArg(format!(
                    "error: QWEN_CP_PREC={v} no es válido; use bf16, int8, int4 o mixed"
                ))),
            };
        }
    }
    if opts.quant_mixed {
        Ok(CpPrecision::Mixed)
    } else if opts.use_int4 {
        Ok(CpPrecision::Int4)
    } else if opts.use_int8 {
        Ok(CpPrecision::Int8)
    } else {
        Ok(CpPrecision::Bf16)
    }
}

impl Model06B {
    pub fn load(
        model_dir: &Path,
        runtime: &ComputeRuntime,
        opts: ModelLoadOptions,
    ) -> Result<Self, EngineError> {
        let t0 = Instant::now();
        let config = ModelConfig06B::load_and_validate(model_dir)?;
        quant_shape_check("talker hidden_size", HIDDEN_SIZE).map_err(EngineError::Model)?;
        quant_shape_check("talker q_dim", Q_DIM).map_err(EngineError::Model)?;
        quant_shape_check("talker intermediate_size", INTERMEDIATE_SIZE)
            .map_err(EngineError::Model)?;
        quant_shape_check("code predictor cp_hidden_size", CP_HIDDEN_SIZE)
            .map_err(EngineError::Model)?;
        quant_shape_check("code predictor cp_q_dim", CP_Q_DIM).map_err(EngineError::Model)?;
        quant_shape_check("code predictor cp_intermediate_size", CP_INTERMEDIATE_SIZE)
            .map_err(EngineError::Model)?;

        let tokenizer = BpeTokenizer::load_from_dir(model_dir)?;
        let main_store = MmapStore::open_model(model_dir)?;
        let speech_dir = model_dir.join("speech_tokenizer");
        let speech_store = MmapStore::open_model(&speech_dir)?;

        let tk_prec = resolve_talker_precision(opts)?;
        let cp_prec = resolve_cp_precision(opts)?;

        if !opts.silent {
            eprintln!(
                "Config: hidden={} text_hidden={} layers={} heads={}/{} head_dim={} inter={}",
                HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
                NUM_LAYERS,
                NUM_HEADS,
                qwen_core::config::NUM_KV_HEADS,
                HEAD_DIM,
                INTERMEDIATE_SIZE
            );
            eprintln!("Threads: {}", runtime.pool.n_threads());
        }

        let talker = Self::load_talker(&main_store, tk_prec, opts.silent)?;
        let cp = Self::load_cp(&main_store, cp_prec, opts.silent)?;
        let decoder = Self::load_decoder(&speech_store, opts.silent)?;
        let speaker_encoder =
            SpeakerEncoderWeights::load_opt(&main_store, opts.silent)?.map(Arc::new);
        let speech_encoder = SpeechEncoderWeights::load_opt(
            &speech_store,
            Arc::clone(&decoder.codebooks),
            opts.silent,
        )?
        .map(Arc::new);

        let mut cached_tts_pad_embed = AlignedBuf::new_zeroed(HIDDEN_SIZE);
        let mut cached_tts_bos_embed = AlignedBuf::new_zeroed(HIDDEN_SIZE);
        let mut cached_tts_eos_embed = AlignedBuf::new_zeroed(HIDDEN_SIZE);
        let mut tmp1 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
        let mut tmp2 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);

        let compute_special = |tid: i32,
                               out: &mut [f32],
                               tmp1: &mut [f32],
                               tmp2: &mut [f32]| {
            let idx = tid as usize;
            let row = &talker.text_embedding_bf16[idx * TEXT_HIDDEN_SIZE..(idx + 1) * TEXT_HIDDEN_SIZE];
            runtime.dispatch.bf16_to_f32_vec(tmp1, row);
            runtime.dispatch.matvec_bf16(
                &runtime.pool,
                tmp2,
                &talker.text_proj_fc1_bf16,
                tmp1,
                TEXT_HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
            );
            if let Some(b) = talker.text_proj_fc1_bias.as_deref() {
                for j in 0..TEXT_HIDDEN_SIZE {
                    tmp2[j] += b[j];
                }
            }
            runtime.dispatch.silu(tmp2);
            runtime.dispatch.matvec_bf16(
                &runtime.pool,
                out,
                &talker.text_proj_fc2_bf16,
                tmp2,
                HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
            );
            if let Some(b) = talker.text_proj_fc2_bias.as_deref() {
                for j in 0..HIDDEN_SIZE {
                    out[j] += b[j];
                }
            }
        };

        compute_special(TOKEN_TTS_PAD, &mut cached_tts_pad_embed, &mut tmp1, &mut tmp2);
        compute_special(TOKEN_TTS_BOS, &mut cached_tts_bos_embed, &mut tmp1, &mut tmp2);
        compute_special(TOKEN_TTS_EOS, &mut cached_tts_eos_embed, &mut tmp1, &mut tmp2);

        let mut ryan_tmp = [0.0f32; HIDDEN_SIZE];
        runtime.dispatch.bf16_to_f32_vec(
            &mut ryan_tmp,
            &talker.codec_embedding_bf16[3061 * HIDDEN_SIZE..3062 * HIDDEN_SIZE],
        );
        let ryan_embed_norm = ryan_tmp.iter().map(|&v| v * v).sum::<f32>().sqrt();

        if !opts.silent {
            eprintln!("Model loaded in {:.0} ms", t0.elapsed().as_secs_f64() * 1000.0);
        }

        Ok(Self {
            config,
            tokenizer,
            talker,
            cp,
            decoder,
            speaker_encoder,
            speech_encoder,
            cached_tts_pad_embed,
            cached_tts_bos_embed,
            cached_tts_eos_embed,
            ryan_embed_norm,
        })
    }

    pub fn weights_have_q4(&self) -> bool {
        for l in &self.talker.layers {
            if matches!(l.wq_q, Some(QuantWeight::Q4_0(_)))
                || matches!(l.gate_up_q, Some(QuantWeight::Q4_0(_)))
            {
                return true;
            }
        }
        for l in &self.cp.layers {
            if matches!(l.wq_q, Some(QuantWeight::Q4_0(_)))
                || matches!(l.gate_up_q, Some(QuantWeight::Q4_0(_)))
            {
                return true;
            }
        }
        false
    }

    pub fn apply_qvoice_weights_override(
        &mut self,
        runtime: &ComputeRuntime,
        weights: &QVoiceWeightsOverride,
        voice_strength: f32,
        vs_l0: i32,
        vs_l1: i32,
        silent: bool,
    ) -> Result<(), EngineError> {
        let mut loaded = 0usize;
        let mut total_bytes = 0usize;

        let parse_f32_payload = |t: &qwen_core::QVoiceWeightTensor| -> Result<F32Weight, EngineError> {
            if !t.payload.len().is_multiple_of(4) {
                return Err(EngineError::Model(format!(
                    "Invalid F32 payload length in .qvoice tensor '{}'",
                    t.name
                )));
            }
            let vals: Vec<f32> = t
                .payload
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(F32Weight::Owned(Arc::new(vals)))
        };

        for t in &weights.tensors {
            let name = t.name.as_str();
            let mut matched_bf16: Option<&mut Bf16Weight> = None;
            let mut matched_f32: Option<&mut F32Weight> = None;

            match name {
                "talker.model.text_embedding.weight" => {
                    matched_bf16 = Some(&mut self.talker.text_embedding_bf16)
                }
                "talker.text_projection.linear_fc1.weight" => {
                    matched_bf16 = Some(&mut self.talker.text_proj_fc1_bf16)
                }
                "talker.text_projection.linear_fc1.bias" => {
                    matched_f32 = self.talker.text_proj_fc1_bias.as_mut()
                }
                "talker.text_projection.linear_fc2.weight" => {
                    matched_bf16 = Some(&mut self.talker.text_proj_fc2_bf16)
                }
                "talker.text_projection.linear_fc2.bias" => {
                    matched_f32 = self.talker.text_proj_fc2_bias.as_mut()
                }
                "talker.model.codec_embedding.weight" => {
                    matched_bf16 = Some(&mut self.talker.codec_embedding_bf16)
                }
                "talker.codec_head.weight" => matched_bf16 = Some(&mut self.talker.codec_head_bf16),
                "talker.model.norm.weight" => matched_f32 = Some(&mut self.talker.norm),
                "talker.code_predictor.model.norm.weight" => matched_f32 = Some(&mut self.cp.norm),
                _ if name.starts_with("talker.code_predictor.model.codec_embedding.") => {
                    if let Some(idx_s) = name
                        .strip_prefix("talker.code_predictor.model.codec_embedding.")
                        .and_then(|s| s.split('.').next())
                    {
                        if let Ok(ci) = idx_s.parse::<usize>() {
                            if ci < self.cp.codec_emb_bf16.len() {
                                matched_bf16 = Some(&mut self.cp.codec_emb_bf16[ci]);
                            }
                        }
                    }
                }
                _ if name.starts_with("talker.code_predictor.lm_head.") => {
                    if let Some(idx_s) = name
                        .strip_prefix("talker.code_predictor.lm_head.")
                        .and_then(|s| s.split('.').next())
                    {
                        if let Ok(ci) = idx_s.parse::<usize>() {
                            if ci < self.cp.lm_heads.len() {
                                matched_bf16 = Some(&mut self.cp.lm_heads[ci].bf16);
                            }
                        }
                    }
                }
                _ if name.starts_with("talker.code_predictor.model.layers.") => {
                    if let Some(rest) = name.strip_prefix("talker.code_predictor.model.layers.") {
                        if let Some((idx_s, suffix)) = rest.split_once('.') {
                            if let Ok(li) = idx_s.parse::<usize>() {
                                if li < self.cp.layers.len() {
                                    let cl = &mut self.cp.layers[li];
                                    match suffix {
                                        "self_attn.q_proj.weight" => matched_bf16 = Some(&mut cl.wq_bf16),
                                        "self_attn.k_proj.weight" => matched_bf16 = Some(&mut cl.wk_bf16),
                                        "self_attn.v_proj.weight" => matched_bf16 = Some(&mut cl.wv_bf16),
                                        "self_attn.o_proj.weight" => matched_bf16 = Some(&mut cl.wo_bf16),
                                        "self_attn.q_norm.weight" => matched_f32 = Some(&mut cl.q_norm),
                                        "self_attn.k_norm.weight" => matched_f32 = Some(&mut cl.k_norm),
                                        "input_layernorm.weight" => matched_f32 = Some(&mut cl.input_norm),
                                        "post_attention_layernorm.weight" => {
                                            matched_f32 = Some(&mut cl.post_attn_norm)
                                        }
                                        "mlp.gate_proj.weight" => matched_bf16 = Some(&mut cl.gate_bf16),
                                        "mlp.up_proj.weight" => matched_bf16 = Some(&mut cl.up_bf16),
                                        "mlp.down_proj.weight" => matched_bf16 = Some(&mut cl.down_bf16),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                _ if name.starts_with("talker.model.layers.") => {
                    if let Some(rest) = name.strip_prefix("talker.model.layers.") {
                        if let Some((idx_s, suffix)) = rest.split_once('.') {
                            if let Ok(li) = idx_s.parse::<usize>() {
                                if li < self.talker.layers.len() {
                                    let l = &mut self.talker.layers[li];
                                    match suffix {
                                        "self_attn.q_proj.weight" => matched_bf16 = Some(&mut l.wq_bf16),
                                        "self_attn.k_proj.weight" => matched_bf16 = Some(&mut l.wk_bf16),
                                        "self_attn.v_proj.weight" => matched_bf16 = Some(&mut l.wv_bf16),
                                        "self_attn.o_proj.weight" => matched_bf16 = Some(&mut l.wo_bf16),
                                        "self_attn.q_norm.weight" => matched_f32 = Some(&mut l.q_norm),
                                        "self_attn.k_norm.weight" => matched_f32 = Some(&mut l.k_norm),
                                        "input_layernorm.weight" => matched_f32 = Some(&mut l.input_norm),
                                        "post_attention_layernorm.weight" => {
                                            matched_f32 = Some(&mut l.post_attn_norm)
                                        }
                                        "mlp.gate_proj.weight" => matched_bf16 = Some(&mut l.gate_bf16),
                                        "mlp.up_proj.weight" => matched_bf16 = Some(&mut l.up_bf16),
                                        "mlp.down_proj.weight" => matched_bf16 = Some(&mut l.down_bf16),
                                        _ => {}
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }

            if let Some(slot) = matched_bf16 {
                let updated = decode_wdelta_bf16(slot.as_slice(), t, voice_strength, vs_l0, vs_l1)?;
                *slot = Bf16Weight::Owned(Arc::new(updated));
                loaded += 1;
                total_bytes += t.payload.len();
            } else if let Some(slot) = matched_f32 {
                *slot = parse_f32_payload(t)?;
                loaded += 1;
                total_bytes += t.payload.len();
            }
        }

        if loaded > 0 && self.weights_have_q4() {
            return Err(EngineError::InvalidArg(
                "error: la voz .qvoice (WDELTA) sobrescribe pesos que están cuantizados en q4_0 y no se pueden recuantizar; use bf16 o --int8"
                    .to_string(),
            ));
        }

        if loaded > 0 {
            for l in &mut self.talker.layers {
                if let Some(ref mut fused) = l.gate_up_fused_bf16 {
                    interleave_gate_up_bf16(
                        fused,
                        &l.gate_bf16,
                        &l.up_bf16,
                        INTERMEDIATE_SIZE,
                        HIDDEN_SIZE,
                    );
                }
                if matches!(l.wq_q, Some(QuantWeight::Int8 { .. })) {
                    let (gu_d, gu_s) = quantize_gate_up_int8(
                        &l.gate_bf16,
                        &l.up_bf16,
                        INTERMEDIATE_SIZE,
                        HIDDEN_SIZE,
                    );
                    l.wq_q = Some(quantize_matrix_int8(&l.wq_bf16, Q_DIM, HIDDEN_SIZE));
                    l.wk_q = Some(quantize_matrix_int8(&l.wk_bf16, KV_DIM, HIDDEN_SIZE));
                    l.wv_q = Some(quantize_matrix_int8(&l.wv_bf16, KV_DIM, HIDDEN_SIZE));
                    l.wo_q = Some(quantize_matrix_int8(&l.wo_bf16, HIDDEN_SIZE, Q_DIM));
                    l.gate_up_q = Some(QuantWeight::Int8 { data: gu_d, scale: gu_s });
                    l.down_q = Some(quantize_matrix_int8(
                        &l.down_bf16,
                        HIDDEN_SIZE,
                        INTERMEDIATE_SIZE,
                    ));
                }
            }
            for cl in &mut self.cp.layers {
                if let Some(ref mut fused) = cl.gate_up_fused_bf16 {
                    interleave_gate_up_bf16(
                        fused,
                        &cl.gate_bf16,
                        &cl.up_bf16,
                        CP_INTERMEDIATE_SIZE,
                        CP_HIDDEN_SIZE,
                    );
                }
                if matches!(cl.wq_q, Some(QuantWeight::Int8 { .. })) {
                    let (gu_d, gu_s) = quantize_gate_up_int8(
                        &cl.gate_bf16,
                        &cl.up_bf16,
                        CP_INTERMEDIATE_SIZE,
                        CP_HIDDEN_SIZE,
                    );
                    cl.wq_q = Some(quantize_matrix_int8(&cl.wq_bf16, CP_Q_DIM, CP_HIDDEN_SIZE));
                    cl.wk_q = Some(quantize_matrix_int8(&cl.wk_bf16, CP_KV_DIM, CP_HIDDEN_SIZE));
                    cl.wv_q = Some(quantize_matrix_int8(&cl.wv_bf16, CP_KV_DIM, CP_HIDDEN_SIZE));
                    cl.wo_q = Some(quantize_matrix_int8(&cl.wo_bf16, CP_HIDDEN_SIZE, CP_Q_DIM));
                    cl.gate_up_q = Some(QuantWeight::Int8 { data: gu_d, scale: gu_s });
                    cl.down_q = Some(quantize_matrix_int8(
                        &cl.down_bf16,
                        CP_HIDDEN_SIZE,
                        CP_INTERMEDIATE_SIZE,
                    ));
                }
            }
            for head in &mut self.cp.lm_heads {
                if matches!(head.quant, Some(QuantWeight::Int8 { .. })) {
                    head.quant = Some(quantize_matrix_int8(&head.bf16, CODEBOOK_SIZE, CP_HIDDEN_SIZE));
                }
            }

            let mut tmp1 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
            let mut tmp2 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
            for (tid, out) in [
                (TOKEN_TTS_PAD, &mut self.cached_tts_pad_embed),
                (TOKEN_TTS_BOS, &mut self.cached_tts_bos_embed),
                (TOKEN_TTS_EOS, &mut self.cached_tts_eos_embed),
            ] {
                let idx = tid as usize;
                let row = &self.talker.text_embedding_bf16
                    [idx * TEXT_HIDDEN_SIZE..(idx + 1) * TEXT_HIDDEN_SIZE];
                runtime.dispatch.bf16_to_f32_vec(&mut tmp1, row);
                runtime.dispatch.matvec_bf16(
                    &runtime.pool,
                    &mut tmp2,
                    &self.talker.text_proj_fc1_bf16,
                    &tmp1,
                    TEXT_HIDDEN_SIZE,
                    TEXT_HIDDEN_SIZE,
                );
                if let Some(b) = self.talker.text_proj_fc1_bias.as_deref() {
                    for j in 0..TEXT_HIDDEN_SIZE {
                        tmp2[j] += b[j];
                    }
                }
                runtime.dispatch.silu(&mut tmp2);
                runtime.dispatch.matvec_bf16(
                    &runtime.pool,
                    out,
                    &self.talker.text_proj_fc2_bf16,
                    &tmp2,
                    HIDDEN_SIZE,
                    TEXT_HIDDEN_SIZE,
                );
                if let Some(b) = self.talker.text_proj_fc2_bias.as_deref() {
                    for j in 0..HIDDEN_SIZE {
                        out[j] += b[j];
                    }
                }
            }

            if !silent {
                eprintln!(
                    "  Loaded {loaded}/{} source talker tensors ({:.1} MB) — full weight override",
                    weights.tensors.len(),
                    total_bytes as f32 / (1024.0 * 1024.0)
                );
            }
        }

        Ok(())
    }

    fn load_talker(
        store: &MmapStore,
        prec: TalkerPrecision,
        silent: bool,
    ) -> Result<TalkerWeights, EngineError> {
        if !silent {
            eprintln!(
                "Loading Talker weights (hidden={}, head_dim={}, layers={})...",
                HIDDEN_SIZE, HEAD_DIM, NUM_LAYERS
            );
        }

        let text_embedding_bf16 = req_bf16(store, "talker.model.text_embedding.weight")?;
        let text_proj_fc1_bf16 = req_bf16(store, "talker.text_projection.linear_fc1.weight")?;
        let text_proj_fc1_bias = opt_f32_w(store, "talker.text_projection.linear_fc1.bias")?;
        let text_proj_fc2_bf16 = req_bf16(store, "talker.text_projection.linear_fc2.weight")?;
        let text_proj_fc2_bias = opt_f32_w(store, "talker.text_projection.linear_fc2.bias")?;
        let codec_embedding_bf16 = req_bf16(store, "talker.model.codec_embedding.weight")?;
        let codec_head_bf16 = req_bf16(store, "talker.codec_head.weight")?;
        let norm = req_f32_w(store, "talker.model.norm.weight")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let wq_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.self_attn.q_proj.weight"))?;
            let wk_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.self_attn.k_proj.weight"))?;
            let wv_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.self_attn.v_proj.weight"))?;
            let wo_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.self_attn.o_proj.weight"))?;
            let q_norm = req_f32_w(store, &format!("talker.model.layers.{i}.self_attn.q_norm.weight"))?;
            let k_norm = req_f32_w(store, &format!("talker.model.layers.{i}.self_attn.k_norm.weight"))?;
            let input_norm = req_f32_w(store, &format!("talker.model.layers.{i}.input_layernorm.weight"))?;
            let post_attn_norm = req_f32_w(
                store,
                &format!("talker.model.layers.{i}.post_attention_layernorm.weight"),
            )?;
            let gate_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.mlp.gate_proj.weight"))?;
            let up_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.mlp.up_proj.weight"))?;
            let down_bf16 = req_bf16(store, &format!("talker.model.layers.{i}.mlp.down_proj.weight"))?;

            let gate_up_fused_bf16 = if prec == TalkerPrecision::Bf16 {
                let mut fused = AlignedBuf::new_zeroed(2 * INTERMEDIATE_SIZE * HIDDEN_SIZE);
                interleave_gate_up_bf16(
                    &mut fused,
                    &gate_bf16,
                    &up_bf16,
                    INTERMEDIATE_SIZE,
                    HIDDEN_SIZE,
                );
                Some(fused)
            } else {
                None
            };

            let (wq_q, wk_q, wv_q, wo_q, gate_up_q, down_q) = match prec {
                TalkerPrecision::Bf16 => (None, None, None, None, None, None),
                TalkerPrecision::Int8 => {
                    let (gu_d, gu_s) =
                        quantize_gate_up_int8(&gate_bf16, &up_bf16, INTERMEDIATE_SIZE, HIDDEN_SIZE);
                    (
                        Some(quantize_matrix_int8(&wq_bf16, Q_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wk_bf16, KV_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wv_bf16, KV_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wo_bf16, HIDDEN_SIZE, Q_DIM)),
                        Some(QuantWeight::Int8 { data: gu_d, scale: gu_s }),
                        Some(quantize_matrix_int8(&down_bf16, HIDDEN_SIZE, INTERMEDIATE_SIZE)),
                    )
                }
                TalkerPrecision::Int4 => {
                    let gu_q4 =
                        quantize_gate_up_q4_0(&gate_bf16, &up_bf16, INTERMEDIATE_SIZE, HIDDEN_SIZE);
                    (
                        Some(quantize_matrix_q4(&wq_bf16, Q_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wk_bf16, KV_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wv_bf16, KV_DIM, HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wo_bf16, HIDDEN_SIZE, Q_DIM)),
                        Some(QuantWeight::Q4_0(gu_q4)),
                        Some(quantize_matrix_q4(&down_bf16, HIDDEN_SIZE, INTERMEDIATE_SIZE)),
                    )
                }
            };

            layers.push(TalkerLayerWeights {
                wq_bf16,
                wk_bf16,
                wv_bf16,
                wo_bf16,
                q_norm,
                k_norm,
                input_norm,
                post_attn_norm,
                gate_bf16,
                up_bf16,
                down_bf16,
                gate_up_fused_bf16,
                wq_q,
                wk_q,
                wv_q,
                wo_q,
                gate_up_q,
                down_q,
            });
        }

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            ROPE_MAX_POS,
            HEAD_DIM,
            qwen_core::config::ROPE_THETA,
        );
        let rope_cos = AlignedBuf::from_slice(&cos_v);
        let rope_sin = AlignedBuf::from_slice(&sin_v);

        if !silent {
            eprintln!(
                "  Talker: {} layers loaded, KV cache {} slots",
                NUM_LAYERS, KV_INITIAL_POSITIONS
            );
        }

        Ok(TalkerWeights {
            text_embedding_bf16,
            text_proj_fc1_bf16,
            text_proj_fc1_bias,
            text_proj_fc2_bf16,
            text_proj_fc2_bias,
            codec_embedding_bf16,
            codec_head_bf16,
            norm,
            layers,
            rope_cos,
            rope_sin,
            precision: prec,
        })
    }

    fn load_cp(
        store: &MmapStore,
        prec: CpPrecision,
        silent: bool,
    ) -> Result<CodePredictorWeights, EngineError> {
        if !silent {
            eprintln!(
                "Loading Code Predictor weights (hidden={}, layers={})...",
                CP_HIDDEN_SIZE, CP_NUM_LAYERS
            );
        }
        let norm = req_f32_w(store, "talker.code_predictor.model.norm.weight")?;
        let mut layers = Vec::with_capacity(CP_NUM_LAYERS);

        for i in 0..CP_NUM_LAYERS {
            let wq_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.q_proj.weight"),
            )?;
            let wk_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.k_proj.weight"),
            )?;
            let wv_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.v_proj.weight"),
            )?;
            let wo_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.o_proj.weight"),
            )?;
            let q_norm = req_f32_w(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.q_norm.weight"),
            )?;
            let k_norm = req_f32_w(
                store,
                &format!("talker.code_predictor.model.layers.{i}.self_attn.k_norm.weight"),
            )?;
            let input_norm = req_f32_w(
                store,
                &format!("talker.code_predictor.model.layers.{i}.input_layernorm.weight"),
            )?;
            let post_attn_norm = req_f32_w(
                store,
                &format!("talker.code_predictor.model.layers.{i}.post_attention_layernorm.weight"),
            )?;
            let gate_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.mlp.gate_proj.weight"),
            )?;
            let up_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.mlp.up_proj.weight"),
            )?;
            let down_bf16 = req_bf16(
                store,
                &format!("talker.code_predictor.model.layers.{i}.mlp.down_proj.weight"),
            )?;

            let gate_up_fused_bf16 = if prec == CpPrecision::Bf16 {
                let mut fused = AlignedBuf::new_zeroed(2 * CP_INTERMEDIATE_SIZE * CP_HIDDEN_SIZE);
                interleave_gate_up_bf16(
                    &mut fused,
                    &gate_bf16,
                    &up_bf16,
                    CP_INTERMEDIATE_SIZE,
                    CP_HIDDEN_SIZE,
                );
                Some(fused)
            } else {
                None
            };

            let (wq_q, wk_q, wv_q, wo_q, gate_up_q, down_q) = match prec {
                CpPrecision::Bf16 => (None, None, None, None, None, None),
                CpPrecision::Int8 => {
                    let (gu_d, gu_s) = quantize_gate_up_int8(
                        &gate_bf16,
                        &up_bf16,
                        CP_INTERMEDIATE_SIZE,
                        CP_HIDDEN_SIZE,
                    );
                    (
                        Some(quantize_matrix_int8(&wq_bf16, CP_Q_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wk_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wv_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wo_bf16, CP_HIDDEN_SIZE, CP_Q_DIM)),
                        Some(QuantWeight::Int8 { data: gu_d, scale: gu_s }),
                        Some(quantize_matrix_int8(
                            &down_bf16,
                            CP_HIDDEN_SIZE,
                            CP_INTERMEDIATE_SIZE,
                        )),
                    )
                }
                CpPrecision::Int4 => {
                    let gu_q4 = quantize_gate_up_q4_0(
                        &gate_bf16,
                        &up_bf16,
                        CP_INTERMEDIATE_SIZE,
                        CP_HIDDEN_SIZE,
                    );
                    (
                        Some(quantize_matrix_q4(&wq_bf16, CP_Q_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wk_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wv_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_q4(&wo_bf16, CP_HIDDEN_SIZE, CP_Q_DIM)),
                        Some(QuantWeight::Q4_0(gu_q4)),
                        Some(quantize_matrix_q4(
                            &down_bf16,
                            CP_HIDDEN_SIZE,
                            CP_INTERMEDIATE_SIZE,
                        )),
                    )
                }
                CpPrecision::Mixed => {
                    let gu_q4 = quantize_gate_up_q4_0(
                        &gate_bf16,
                        &up_bf16,
                        CP_INTERMEDIATE_SIZE,
                        CP_HIDDEN_SIZE,
                    );
                    (
                        Some(quantize_matrix_int8(&wq_bf16, CP_Q_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wk_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wv_bf16, CP_KV_DIM, CP_HIDDEN_SIZE)),
                        Some(quantize_matrix_int8(&wo_bf16, CP_HIDDEN_SIZE, CP_Q_DIM)),
                        Some(QuantWeight::Q4_0(gu_q4)),
                        Some(quantize_matrix_q4(
                            &down_bf16,
                            CP_HIDDEN_SIZE,
                            CP_INTERMEDIATE_SIZE,
                        )),
                    )
                }
            };

            layers.push(CpLayerWeights {
                wq_bf16,
                wk_bf16,
                wv_bf16,
                wo_bf16,
                q_norm,
                k_norm,
                input_norm,
                post_attn_norm,
                gate_bf16,
                up_bf16,
                down_bf16,
                gate_up_fused_bf16,
                wq_q,
                wk_q,
                wv_q,
                wo_q,
                gate_up_q,
                down_q,
            });
        }

        let mut codec_emb_bf16 = Vec::with_capacity(15);
        let mut lm_heads = Vec::with_capacity(15);
        for g in 0..15 {
            let emb = req_bf16(
                store,
                &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
            )?;
            let head_bf16 =
                req_bf16(store, &format!("talker.code_predictor.lm_head.{g}.weight"))?;
            let quant = match prec {
                CpPrecision::Bf16 => None,
                CpPrecision::Int8 | CpPrecision::Mixed => {
                    Some(quantize_matrix_int8(&head_bf16, CODEBOOK_SIZE, CP_HIDDEN_SIZE))
                }
                CpPrecision::Int4 => {
                    Some(quantize_matrix_q4(&head_bf16, CODEBOOK_SIZE, CP_HIDDEN_SIZE))
                }
            };
            codec_emb_bf16.push(emb);
            lm_heads.push(CpLmHead {
                bf16: head_bf16,
                quant,
            });
        }

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            CP_KV_MAX_POS,
            CP_HEAD_DIM,
            qwen_core::config::ROPE_THETA,
        );
        let all_quantized = layers.iter().all(|l| l.all_quantized());

        let cp = CodePredictorWeights {
            norm,
            layers,
            codec_emb_bf16,
            lm_heads,
            rope_cos: AlignedBuf::from_slice(&cos_v),
            rope_sin: AlignedBuf::from_slice(&sin_v),
            precision: prec,
            all_quantized,
        };

        if !silent {
            eprintln!(
                "  Code Predictor: {} layers loaded, q_dim={} kv_dim={} [{}]",
                CP_NUM_LAYERS,
                CP_Q_DIM,
                CP_KV_DIM,
                cp.prec_label()
            );
        }

        Ok(cp)
    }

    fn load_decoder(store: &MmapStore, silent: bool) -> Result<SpeechDecoderWeights, EngineError> {
        if !silent {
            eprintln!("Loading Speech Decoder weights...");
        }
        let mut codebooks: [AlignedBuf<f32>; NUM_CODEBOOKS] =
            std::array::from_fn(|_| AlignedBuf::new_zeroed(CODEBOOK_SIZE * CODEBOOK_DIM));

        let dequant_cb = |dst: &mut [f32], sum: &[f32], usage: &[f32]| {
            for i in 0..CODEBOOK_SIZE {
                let inv = 1.0f32 / usage[i].max(1e-5f32);
                for d in 0..CODEBOOK_DIM {
                    dst[i * CODEBOOK_DIM + d] = sum[i * CODEBOOK_DIM + d] * inv;
                }
            }
        };

        let s0 = req_f32(
            store,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum",
        )?;
        let u0 = req_f32(
            store,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook.cluster_usage",
        )?;
        dequant_cb(&mut codebooks[0], &s0, &u0);

        for k in 0..15 {
            let sk = req_f32(
                store,
                &format!("decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.embedding_sum"),
            )?;
            let uk = req_f32(
                store,
                &format!("decoder.quantizer.rvq_rest.vq.layers.{k}._codebook.cluster_usage"),
            )?;
            dequant_cb(&mut codebooks[k + 1], &sk, &uk);
        }

        let rvq_first_output_proj =
            req_f32(store, "decoder.quantizer.rvq_first.output_proj.weight")?;
        let rvq_rest_output_proj =
            req_f32(store, "decoder.quantizer.rvq_rest.output_proj.weight")?;

        let pre_conv_weight = req_f32(store, "decoder.pre_conv.conv.weight")?;
        let pre_conv_bias = opt_f32(store, "decoder.pre_conv.conv.bias")?;
        let pre_conv_wk = conv_reorder_weight(&pre_conv_weight, 1024, 512, 3);

        let input_proj_weight = req_f32(store, "decoder.pre_transformer.input_proj.weight")?;
        let input_proj_bias = opt_f32(store, "decoder.pre_transformer.input_proj.bias")?;
        let final_norm_weight = opt_f32(store, "decoder.pre_transformer.norm.weight")?;
        let output_proj_weight = req_f32(store, "decoder.pre_transformer.output_proj.weight")?;
        let output_proj_bias = opt_f32(store, "decoder.pre_transformer.output_proj.bias")?;

        let mut pre_layers = Vec::with_capacity(DEC_NUM_LAYERS);
        for i in 0..DEC_NUM_LAYERS {
            pre_layers.push(SdPreLayerWeights {
                attn_norm: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.input_layernorm.weight"),
                )?,
                attn_q: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.self_attn.q_proj.weight"),
                )?,
                attn_k: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.self_attn.k_proj.weight"),
                )?,
                attn_v: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.self_attn.v_proj.weight"),
                )?,
                attn_o: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.self_attn.o_proj.weight"),
                )?,
                attn_layer_scale: opt_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.self_attn_layer_scale.scale"),
                )?,
                ffn_norm: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.post_attention_layernorm.weight"),
                )?,
                ffn_gate: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.mlp.gate_proj.weight"),
                )?,
                ffn_up: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.mlp.up_proj.weight"),
                )?,
                ffn_down: req_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.mlp.down_proj.weight"),
                )?,
                ffn_layer_scale: opt_f32(
                    store,
                    &format!("decoder.pre_transformer.layers.{i}.mlp_layer_scale.scale"),
                )?,
            });
        }

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            SD_ROPE_MAX_POS,
            DEC_HEAD_DIM,
            DEC_ROPE_THETA,
        );

        let load_cn = |b: usize| -> Result<SdConvNeXtWeights, EngineError> {
            Ok(SdConvNeXtWeights {
                conv_weight: req_f32(store, &format!("decoder.upsample.{b}.0.conv.weight"))?,
                conv_bias: opt_f32(store, &format!("decoder.upsample.{b}.0.conv.bias"))?,
                dwconv_weight: req_f32(
                    store,
                    &format!("decoder.upsample.{b}.1.dwconv.conv.weight"),
                )?,
                dwconv_bias: opt_f32(
                    store,
                    &format!("decoder.upsample.{b}.1.dwconv.conv.bias"),
                )?,
                pwconv1_weight: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv1.weight"))?,
                pwconv1_bias: opt_f32(store, &format!("decoder.upsample.{b}.1.pwconv1.bias"))?,
                pwconv2_weight: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv2.weight"))?,
                pwconv2_bias: opt_f32(store, &format!("decoder.upsample.{b}.1.pwconv2.bias"))?,
                norm_weight: req_f32(store, &format!("decoder.upsample.{b}.1.norm.weight"))?,
                norm_bias: req_f32(store, &format!("decoder.upsample.{b}.1.norm.bias"))?,
                gamma: req_f32(store, &format!("decoder.upsample.{b}.1.gamma"))?,
            })
        };
        let convnext = [load_cn(0)?, load_cn(1)?];

        let initial_conv_weight = req_f32(store, "decoder.decoder.0.conv.weight")?;
        let initial_conv_bias = opt_f32(store, "decoder.decoder.0.conv.bias")?;
        let initial_conv_wk = conv_reorder_weight(&initial_conv_weight, 1536, 1024, 7);

        let load_ub = |b: usize| -> Result<SdUpsampleBlockWeights, EngineError> {
            let bi = b + 1;
            let ch = DEC_UPSAMPLE_CHANNELS[b];
            let load_res = |r: usize| -> Result<SdResBlockWeights, EngineError> {
                let ri = r + 2;
                let conv1_weight = req_f32(
                    store,
                    &format!("decoder.decoder.{bi}.block.{ri}.conv1.conv.weight"),
                )?;
                let conv1_wk = conv_reorder_weight(&conv1_weight, ch, ch, 7);
                Ok(SdResBlockWeights {
                    snake1_alpha: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act1.alpha"),
                    )?,
                    snake1_beta: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act1.beta"),
                    )?,
                    conv1_weight,
                    conv1_bias: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv1.conv.bias"),
                    )?,
                    conv1_wk,
                    snake2_alpha: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act2.alpha"),
                    )?,
                    snake2_beta: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act2.beta"),
                    )?,
                    conv2_weight: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv2.conv.weight"),
                    )?,
                    conv2_bias: opt_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv2.conv.bias"),
                    )?,
                })
            };
            Ok(SdUpsampleBlockWeights {
                snake_alpha: opt_f32(store, &format!("decoder.decoder.{bi}.block.0.alpha"))?,
                snake_beta: opt_f32(store, &format!("decoder.decoder.{bi}.block.0.beta"))?,
                conv_weight: req_f32(
                    store,
                    &format!("decoder.decoder.{bi}.block.1.conv.weight"),
                )?,
                conv_bias: opt_f32(store, &format!("decoder.decoder.{bi}.block.1.conv.bias"))?,
                res_blocks: [load_res(0)?, load_res(1)?, load_res(2)?],
            })
        };
        let upsample_blocks = [load_ub(0)?, load_ub(1)?, load_ub(2)?, load_ub(3)?];

        let final_snake_alpha = req_f32(store, "decoder.decoder.5.alpha")?;
        let final_snake_beta = req_f32(store, "decoder.decoder.5.beta")?;
        let final_conv_weight = req_f32(store, "decoder.decoder.6.conv.weight")?;
        let final_conv_bias = opt_f32(store, "decoder.decoder.6.conv.bias")?;

        if !silent {
            eprintln!("  Codebooks: 16/16 (dequantized from EMA)");
        }

        Ok(SpeechDecoderWeights {
            codebooks: Arc::new(codebooks),
            rvq_first_output_proj,
            rvq_rest_output_proj,
            pre_conv_weight,
            pre_conv_bias,
            pre_conv_wk,
            input_proj_weight,
            input_proj_bias,
            pre_layers,
            final_norm_weight,
            output_proj_weight,
            output_proj_bias,
            rope_cos: AlignedBuf::from_slice(&cos_v),
            rope_sin: AlignedBuf::from_slice(&sin_v),
            convnext,
            initial_conv_weight,
            initial_conv_bias,
            initial_conv_wk,
            upsample_blocks,
            final_snake_alpha,
            final_snake_beta,
            final_conv_weight,
            final_conv_bias,
        })
    }
}
