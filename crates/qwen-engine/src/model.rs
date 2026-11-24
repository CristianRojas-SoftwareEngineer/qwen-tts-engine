// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::speech_decoder::SD_TILE_FROM_BLOCK;

use qwen_core::config::{
    CODEBOOK_DIM, CODEBOOK_SIZE, CODEC_VOCAB_SIZE, CP_HEAD_DIM, CP_HIDDEN_SIZE,
    CP_INTERMEDIATE_SIZE, CP_KV_DIM, CP_NUM_LAYERS, CP_Q_DIM, DEC_HEAD_DIM,
    DEC_NUM_LAYERS, DEC_ROPE_THETA, DEC_UPSAMPLE_CHANNELS,
    HEAD_DIM, HIDDEN_SIZE, INTERMEDIATE_SIZE, KV_DIM, NUM_CODEBOOKS, NUM_HEADS, NUM_LAYERS,
    Q_DIM, TEXT_HIDDEN_SIZE, TOKEN_TTS_BOS, TOKEN_TTS_EOS, TOKEN_TTS_PAD,
};
use qwen_core::{
    validate_model_config, Bf16Ref, BpeTokenizer, F32Ref, LcgRng, MmapStore, Sampler,
};
use qwen_kernels::quant::{
    quant_shape_check, quantize_bf16_to_int8, quantize_bf16_to_q4_0, quantize_gate_up_int8,
    quantize_gate_up_q4_0,
};
use qwen_kernels::{AlignedBuf, ComputePool, ComputeRuntime, KernelDispatch, Q4_0Block};

use crate::error::EngineError;
use crate::voice_clone::SpeakerEncoderWeights;

pub const KV_INITIAL_POSITIONS: usize = 256;
pub const ROPE_MAX_POS: usize = 8192;
pub const CP_KV_MAX_POS: usize = 64;
pub const SD_ROPE_MAX_POS: usize = 8000;

/// Matriz INT8 con una escala por fila.
pub struct Int8Weight {
    pub data: AlignedBuf<i8>,
    pub scale: AlignedBuf<f32>,
}

/// Pesos de una capa del Talker. Los BF16 los lee el prefill y los INT8 el decodificado paso a paso;
/// `gate_up` entrelaza las filas de `gate_proj` (pares) y `up_proj` (impares).
pub struct TalkerLayerWeights {
    pub wq_bf16: Bf16Ref,
    pub wk_bf16: Bf16Ref,
    pub wv_bf16: Bf16Ref,
    pub wo_bf16: Bf16Ref,
    pub q_norm: F32Ref,
    pub k_norm: F32Ref,
    pub input_norm: F32Ref,
    pub post_attn_norm: F32Ref,
    pub gate_bf16: Bf16Ref,
    pub up_bf16: Bf16Ref,
    pub down_bf16: Bf16Ref,
    pub wq: Int8Weight,
    pub wk: Int8Weight,
    pub wv: Int8Weight,
    pub wo: Int8Weight,
    pub gate_up: Int8Weight,
    pub down: Int8Weight,
}

/// QKV fusionado en INT8 sobre las tres matrices de una capa.
#[allow(clippy::too_many_arguments)]
pub(crate) fn int8_qkv(
    runtime: &ComputeRuntime,
    wq: &Int8Weight,
    wk: &Int8Weight,
    wv: &Int8Weight,
    x_norm: &[f32],
    q: &mut [f32],
    k: &mut [f32],
    v: &mut [f32],
    hidden: usize,
    q_dim: usize,
    kv_dim: usize,
) {
    runtime.dispatch.matvec_int8_qkv(
        &runtime.pool,
        q,
        k,
        v,
        &wq.data,
        &wq.scale,
        &wk.data,
        &wk.scale,
        &wv.data,
        &wv.scale,
        x_norm,
        hidden,
        q_dim,
        kv_dim,
    );
}

impl TalkerLayerWeights {
    fn load(
        store: &MmapStore,
        prefix: &str,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        inter: usize,
    ) -> Result<Self, EngineError> {
        let wq_bf16 = req_bf16(store, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let wk_bf16 = req_bf16(store, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let wv_bf16 = req_bf16(store, &format!("{prefix}.self_attn.v_proj.weight"))?;
        let wo_bf16 = req_bf16(store, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let q_norm = req_f32(store, &format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm = req_f32(store, &format!("{prefix}.self_attn.k_norm.weight"))?;
        let input_norm = req_f32(store, &format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_norm = req_f32(store, &format!("{prefix}.post_attention_layernorm.weight"))?;
        let gate_bf16 = req_bf16(store, &format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_bf16 = req_bf16(store, &format!("{prefix}.mlp.up_proj.weight"))?;
        let down_bf16 = req_bf16(store, &format!("{prefix}.mlp.down_proj.weight"))?;

        // Filas de `gate` y `up` entrelazadas (gate en las pares, up en las impares). Con una escala
        // por fila, equivale a cuantizar cada matriz por separado.
        let (data, scale) = quantize_gate_up_int8(&gate_bf16, &up_bf16, inter, hidden);
        let gate_up = Int8Weight { data, scale };
        let wq = quantize_matrix_int8(&wq_bf16, q_dim, hidden);
        let wk = quantize_matrix_int8(&wk_bf16, kv_dim, hidden);
        let wv = quantize_matrix_int8(&wv_bf16, kv_dim, hidden);
        let wo = quantize_matrix_int8(&wo_bf16, hidden, q_dim);
        let down = quantize_matrix_int8(&down_bf16, hidden, inter);

        Ok(Self {
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
            wq,
            wk,
            wv,
            wo,
            gate_up,
            down,
        })
    }
}

/// Pesos de una capa del Code Predictor. Los pesos BF16 solo se leen al cuantizar durante la carga.
pub struct CpLayerWeights {
    pub q_norm: F32Ref,
    pub k_norm: F32Ref,
    pub input_norm: F32Ref,
    pub post_attn_norm: F32Ref,
    pub wq: Int8Weight,
    pub wk: Int8Weight,
    pub wv: Int8Weight,
    pub wo: Int8Weight,
    pub gate_up: AlignedBuf<Q4_0Block>,
    pub down: AlignedBuf<Q4_0Block>,
}

impl CpLayerWeights {
    fn load(
        store: &MmapStore,
        prefix: &str,
        hidden: usize,
        q_dim: usize,
        kv_dim: usize,
        inter: usize,
    ) -> Result<Self, EngineError> {
        let wq_bf16 = req_bf16(store, &format!("{prefix}.self_attn.q_proj.weight"))?;
        let wk_bf16 = req_bf16(store, &format!("{prefix}.self_attn.k_proj.weight"))?;
        let wv_bf16 = req_bf16(store, &format!("{prefix}.self_attn.v_proj.weight"))?;
        let wo_bf16 = req_bf16(store, &format!("{prefix}.self_attn.o_proj.weight"))?;
        let q_norm = req_f32(store, &format!("{prefix}.self_attn.q_norm.weight"))?;
        let k_norm = req_f32(store, &format!("{prefix}.self_attn.k_norm.weight"))?;
        let input_norm = req_f32(store, &format!("{prefix}.input_layernorm.weight"))?;
        let post_attn_norm = req_f32(store, &format!("{prefix}.post_attention_layernorm.weight"))?;
        let gate_bf16 = req_bf16(store, &format!("{prefix}.mlp.gate_proj.weight"))?;
        let up_bf16 = req_bf16(store, &format!("{prefix}.mlp.up_proj.weight"))?;
        let down_bf16 = req_bf16(store, &format!("{prefix}.mlp.down_proj.weight"))?;

        let gate_up = quantize_gate_up_q4_0(&gate_bf16, &up_bf16, inter, hidden);

        Ok(Self {
            q_norm,
            k_norm,
            input_norm,
            post_attn_norm,
            wq: quantize_matrix_int8(&wq_bf16, q_dim, hidden),
            wk: quantize_matrix_int8(&wk_bf16, kv_dim, hidden),
            wv: quantize_matrix_int8(&wv_bf16, kv_dim, hidden),
            wo: quantize_matrix_int8(&wo_bf16, hidden, q_dim),
            gate_up,
            down: quantize_matrix_q4(&down_bf16, hidden, inter),
        })
    }
}

pub struct TalkerWeights {
    pub text_embedding_bf16: Bf16Ref,
    pub text_proj_fc1_bf16: Bf16Ref,
    pub text_proj_fc1_bias: F32Ref,
    pub text_proj_fc2_bf16: Bf16Ref,
    pub text_proj_fc2_bias: F32Ref,
    pub codec_embedding_bf16: Bf16Ref,
    pub codec_head_bf16: Bf16Ref,
    pub norm: F32Ref,
    pub layers: Vec<TalkerLayerWeights>,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
}

/// Cabeza del lm_head de un codebook del Code Predictor, en INT8.
pub struct CpLmHead {
    pub weight: Int8Weight,
}

impl CpLmHead {
    #[inline]
    pub fn argmax(&self, runtime: &ComputeRuntime, normed: &[f32]) -> usize {
        runtime.dispatch.argmax_matvec_int8(
            &runtime.pool,
            normed,
            &self.weight.data,
            &self.weight.scale,
            CP_HIDDEN_SIZE,
            CODEBOOK_SIZE,
        )
    }
}

pub struct CodePredictorWeights {
    pub norm: F32Ref,
    pub layers: Vec<CpLayerWeights>,
    pub codec_emb_bf16: Vec<Bf16Ref>,
    pub lm_heads: Vec<CpLmHead>,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
}

pub struct SdPreLayerWeights {
    pub attn_norm: F32Ref,
    pub attn_q: F32Ref,
    pub attn_k: F32Ref,
    pub attn_v: F32Ref,
    pub attn_o: F32Ref,
    pub attn_layer_scale: F32Ref,
    pub ffn_norm: F32Ref,
    pub ffn_gate: F32Ref,
    pub ffn_up: F32Ref,
    pub ffn_down: F32Ref,
    pub ffn_layer_scale: F32Ref,
}

pub struct SdConvNeXtWeights {
    pub conv_weight: F32Ref,
    pub conv_bias: F32Ref,
    pub dwconv_weight: F32Ref,
    pub dwconv_bias: F32Ref,
    pub pwconv1_weight: F32Ref,
    pub pwconv1_bias: F32Ref,
    pub pwconv2_weight: F32Ref,
    pub pwconv2_bias: F32Ref,
    pub norm_weight: F32Ref,
    pub norm_bias: F32Ref,
    pub gamma: F32Ref,
}

pub struct SdResBlockWeights {
    pub snake1_alpha: F32Ref,
    pub snake1_beta: F32Ref,
    pub conv1_bias: F32Ref,
    /// Pesos de `conv1` preparados para el camino de su bloque: reordenados (`conv_reorder_weight`)
    /// si `b < SD_TILE_FROM_BLOCK`, y en layout im2col `[out][k · in]` (`conv_im2col_weight`, una sola
    /// `sgemm` con `K = 7 · in`) en los bloques que el *streaming* procesa por tiles.
    pub conv1_w: AlignedBuf<f32>,
    pub snake2_alpha: F32Ref,
    pub snake2_beta: F32Ref,
    pub conv2_weight: F32Ref,
    pub conv2_bias: F32Ref,
}

pub struct SdUpsampleBlockWeights {
    pub snake_alpha: F32Ref,
    pub snake_beta: F32Ref,
    pub conv_weight: F32Ref,
    pub conv_bias: F32Ref,
    pub res_blocks: [SdResBlockWeights; 3],
}

pub struct SpeechDecoderWeights {
    pub codebooks: [AlignedBuf<f32>; NUM_CODEBOOKS],
    pub rvq_first_output_proj: F32Ref,
    pub rvq_rest_output_proj: F32Ref,
    pub pre_conv_bias: F32Ref,
    pub pre_conv_wk: AlignedBuf<f32>,
    pub input_proj_weight: F32Ref,
    pub input_proj_bias: F32Ref,
    pub pre_layers: Vec<SdPreLayerWeights>,
    pub final_norm_weight: F32Ref,
    pub output_proj_weight: F32Ref,
    pub output_proj_bias: F32Ref,
    pub rope_cos: AlignedBuf<f32>,
    pub rope_sin: AlignedBuf<f32>,
    pub convnext: [SdConvNeXtWeights; 2],
    pub initial_conv_bias: F32Ref,
    pub initial_conv_wk: AlignedBuf<f32>,
    pub upsample_blocks: [SdUpsampleBlockWeights; 4],
    pub final_snake_alpha: F32Ref,
    pub final_snake_beta: F32Ref,
    pub final_conv_weight: F32Ref,
    pub final_conv_bias: F32Ref,
}

/// Modelo Qwen3-TTS `0.6B` (`Base`) inmutable y compartible entre hilos (`Send + Sync`).
pub struct Model06B {
    pub tokenizer: BpeTokenizer,
    pub talker: TalkerWeights,
    pub cp: CodePredictorWeights,
    pub decoder: SpeechDecoderWeights,
    pub speaker_encoder: SpeakerEncoderWeights,
    pub cached_tts_pad_embed: AlignedBuf<f32>,
    pub cached_tts_bos_embed: AlignedBuf<f32>,
    pub cached_tts_eos_embed: AlignedBuf<f32>,
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
        let mut new_max = self.max_pos.max(1);
        while new_max < required {
            new_max = new_max
                .checked_mul(2)
                .unwrap_or(required)
                .min(ROPE_MAX_POS.max(required));
            if new_max >= required {
                break;
            }
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
    // Scratch de las pasadas de 2 posiciones del Code Predictor (`cp_step2`)
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
        }
    }
}

impl Default for CpBuffers {
    fn default() -> Self {
        Self::new()
    }
}

/// 8 slots monótonos reutilizables para el decoder convolucional acústico.
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
    // Buffers scratch reutilizables por chunk (cero asignaciones de heap durante streaming)
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
        }
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
        self.kv_base = 0;
        self.latent_frames = 0;
        self.latent_base = 0;
        self.vq_pad_valid = false;
        self.cs_warm = false;
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
            for (r, &dil) in DILS.iter().enumerate() {
                self.cs_res_tail[b][r] =
                    AlignedBuf::new_zeroed(DEC_UPSAMPLE_CHANNELS[b] * 6 * dil);
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

/// Estado mutable exclusivo de una sesión o trabajador (cachés KV, buffers de trabajo, muestreador y RNG).
pub struct SessionScratch {
    pub talker_kv: TalkerKvCache,
    pub cp_kv: CpKvCache,
    pub talker_bufs: TalkerBuffers,
    pub cp_bufs: CpBuffers,
    pub sd_stream: SdStreamState,
    pub sd_ws: DecoderWorkspace,
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

type AbortPollFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Señal cooperativa de cancelación por flag atómico y función opcional de sondeo.
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

/// Métricas de latencia y conteo de frames producidas por una síntesis.
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
    /// Frame del primer `CODEC_EOS`, o `None` si la síntesis llegó al tope sin EOS.
    pub first_eos_frame: Option<usize>,
}

fn req_bf16(store: &MmapStore, name: &str) -> Result<Bf16Ref, EngineError> {
    Ok(store.tensor_bf16_ref(name)?)
}

fn req_f32(store: &MmapStore, name: &str) -> Result<F32Ref, EngineError> {
    Ok(store.tensor_f32_ref(name)?)
}

/// Ejecuta `load(i)` para `i` en `0..n` repartido en el pool y devuelve los resultados en orden.
/// Cada tarea es independiente; el primer error (en orden de índice) se devuelve. `load` no debe
/// usar el pool: `ComputePool::parallel` no es reentrante.
fn load_parallel<T: Send>(
    pool: &ComputePool,
    n: usize,
    load: impl Fn(usize) -> Result<T, EngineError> + Sync,
) -> Result<Vec<T>, EngineError> {
    let slots: Vec<Mutex<Option<Result<T, EngineError>>>> =
        (0..n).map(|_| Mutex::new(None)).collect();
    pool.parallel(n, |i, _| {
        let result = load(i);
        *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
    });
    slots
        .into_iter()
        .map(|slot| {
            slot.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .expect("cada tarea de carga debe haber guardado su resultado")
        })
        .collect()
}

fn quantize_matrix_int8(src: &[u16], rows: usize, cols: usize) -> Int8Weight {
    let (data, scale) = quantize_bf16_to_int8(src, rows, cols);
    Int8Weight { data, scale }
}

fn quantize_matrix_q4(src: &[u16], rows: usize, cols: usize) -> AlignedBuf<Q4_0Block> {
    quantize_bf16_to_q4_0(src, rows, cols)
}

/// Reordena `weight[out][in][k]` al layout im2col `[out][k · in + ic]`.
fn conv_im2col_weight(
    weight: &[f32],
    out_ch: usize,
    in_ch: usize,
    kernel: usize,
) -> AlignedBuf<f32> {
    let mut wr = AlignedBuf::new_zeroed(out_ch * kernel * in_ch);
    for oc in 0..out_ch {
        for k in 0..kernel {
            for ic in 0..in_ch {
                wr[oc * kernel * in_ch + k * in_ch + ic] = weight[(oc * in_ch + ic) * kernel + k];
            }
        }
    }
    wr
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_text_proj_token(
    dispatch: &KernelDispatch,
    pool: &qwen_kernels::ComputePool,
    text_embedding_bf16: &[u16],
    fc1_w: &[u16],
    fc1_b: &[f32],
    fc2_w: &[u16],
    fc2_b: &[f32],
    tid: i32,
    tmp1: &mut [f32],
    tmp2: &mut [f32],
    out: &mut [f32],
) {
    let th = TEXT_HIDDEN_SIZE;
    let h = HIDDEN_SIZE;
    let idx = tid as usize;
    let emb_row = &text_embedding_bf16[idx * th..(idx + 1) * th];
    dispatch.bf16_to_f32_vec(&mut tmp1[..th], emb_row);

    dispatch.matvec_bf16(pool, &mut tmp2[..th], fc1_w, &tmp1[..th], th, th);
    for (slot, &bv) in tmp2[..th].iter_mut().zip(fc1_b[..th].iter()) {
        *slot += bv;
    }
    dispatch.silu(&mut tmp2[..th]);
    dispatch.matvec_bf16(pool, &mut out[..h], fc2_w, &tmp2[..th], h, th);
    for (slot, &bv) in out[..h].iter_mut().zip(fc2_b[..h].iter()) {
        *slot += bv;
    }
}

pub fn embed_one_text_token_compute(
    dispatch: &KernelDispatch,
    pool: &qwen_kernels::ComputePool,
    model: &Model06B,
    tid: i32,
    tmp1: &mut [f32],
    tmp2: &mut [f32],
    out: &mut [f32],
) {
    let (fc1_w, fc1_b, fc2_w, fc2_b) = (
        model.talker.text_proj_fc1_bf16.as_slice(),
        &model.talker.text_proj_fc1_bias,
        model.talker.text_proj_fc2_bf16.as_slice(),
        &model.talker.text_proj_fc2_bias,
    );
    compute_text_proj_token(
        dispatch,
        pool,
        &model.talker.text_embedding_bf16,
        fc1_w,
        fc1_b,
        fc2_w,
        fc2_b,
        tid,
        tmp1,
        tmp2,
        out,
    );
}

pub fn lookup_codec_embed(
    dispatch: &KernelDispatch,
    model: &Model06B,
    token_id: i32,
    out: &mut [f32],
) {
    let tid = token_id as usize;
    let row = &model.talker.codec_embedding_bf16[tid * HIDDEN_SIZE..(tid + 1) * HIDDEN_SIZE];
    dispatch.bf16_to_f32_vec(&mut out[..HIDDEN_SIZE], row);
}

fn compute_special_text_embeds(
    runtime: &ComputeRuntime,
    talker: &TalkerWeights,
) -> (AlignedBuf<f32>, AlignedBuf<f32>, AlignedBuf<f32>) {
    let mut pad = AlignedBuf::new_zeroed(HIDDEN_SIZE);
    let mut bos = AlignedBuf::new_zeroed(HIDDEN_SIZE);
    let mut eos = AlignedBuf::new_zeroed(HIDDEN_SIZE);
    let mut tmp1 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);
    let mut tmp2 = AlignedBuf::<f32>::new_zeroed(TEXT_HIDDEN_SIZE);

    for (tid, out) in [
        (TOKEN_TTS_PAD, &mut pad),
        (TOKEN_TTS_BOS, &mut bos),
        (TOKEN_TTS_EOS, &mut eos),
    ] {
        compute_text_proj_token(
            runtime.dispatch,
            &runtime.pool,
            &talker.text_embedding_bf16,
            &talker.text_proj_fc1_bf16,
            &talker.text_proj_fc1_bias,
            &talker.text_proj_fc2_bf16,
            &talker.text_proj_fc2_bias,
            tid,
            &mut tmp1,
            &mut tmp2,
            out,
        );
    }
    (pad, bos, eos)
}

impl Model06B {
    pub fn load(
        model_dir: &Path,
        runtime: &ComputeRuntime,
        silent: bool,
    ) -> Result<Self, EngineError> {
        let t0 = Instant::now();
        validate_model_config(model_dir)?;
        quant_shape_check("`hidden_size` del Talker", HIDDEN_SIZE).map_err(EngineError::Model)?;
        quant_shape_check("`q_dim` del Talker", Q_DIM).map_err(EngineError::Model)?;
        quant_shape_check("`intermediate_size` del Talker", INTERMEDIATE_SIZE)
            .map_err(EngineError::Model)?;
        quant_shape_check("`cp_hidden_size` del Code Predictor", CP_HIDDEN_SIZE)
            .map_err(EngineError::Model)?;
        quant_shape_check("`cp_q_dim` del Code Predictor", CP_Q_DIM).map_err(EngineError::Model)?;
        quant_shape_check(
            "`cp_intermediate_size` del Code Predictor",
            CP_INTERMEDIATE_SIZE,
        )
        .map_err(EngineError::Model)?;

        let tokenizer = BpeTokenizer::load_from_dir(model_dir)?;
        let main_store = MmapStore::open_model(model_dir)?;
        let speech_dir = model_dir.join("speech_tokenizer");
        let speech_store = MmapStore::open_model(&speech_dir)?;

        if !silent {
            eprintln!(
                "Configuración: hidden={} text_hidden={} layers={} heads={}/{} head_dim={} inter={}",
                HIDDEN_SIZE,
                TEXT_HIDDEN_SIZE,
                NUM_LAYERS,
                NUM_HEADS,
                qwen_core::config::NUM_KV_HEADS,
                HEAD_DIM,
                INTERMEDIATE_SIZE
            );
            eprintln!("Hilos: {}", runtime.pool.n_threads());
        }

        let talker = Self::load_talker(&main_store, &runtime.pool, silent)?;
        let cp = Self::load_cp(&main_store, &runtime.pool, silent)?;
        let decoder = Self::load_decoder(&speech_store, silent)?;
        let speaker_encoder = SpeakerEncoderWeights::load(&main_store, silent)?;

        let (cached_tts_pad_embed, cached_tts_bos_embed, cached_tts_eos_embed) =
            compute_special_text_embeds(runtime, &talker);

        if !silent {
            eprintln!(
                "Modelo cargado en {:.0} ms",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(Self {
            tokenizer,
            talker,
            cp,
            decoder,
            speaker_encoder,
            cached_tts_pad_embed,
            cached_tts_bos_embed,
            cached_tts_eos_embed,
        })
    }

    fn load_talker(
        store: &MmapStore,
        pool: &ComputePool,
        silent: bool,
    ) -> Result<TalkerWeights, EngineError> {
        if !silent {
            eprintln!(
                "Cargando pesos del Talker (hidden={}, head_dim={}, layers={})...",
                HIDDEN_SIZE, HEAD_DIM, NUM_LAYERS
            );
        }

        let text_embedding_bf16 = req_bf16(store, "talker.model.text_embedding.weight")?;
        let text_proj_fc1_bf16 = req_bf16(store, "talker.text_projection.linear_fc1.weight")?;
        let text_proj_fc1_bias = req_f32(store, "talker.text_projection.linear_fc1.bias")?;
        let text_proj_fc2_bf16 = req_bf16(store, "talker.text_projection.linear_fc2.weight")?;
        let text_proj_fc2_bias = req_f32(store, "talker.text_projection.linear_fc2.bias")?;
        let codec_embedding_bf16 = req_bf16(store, "talker.model.codec_embedding.weight")?;
        let codec_head_bf16 = req_bf16(store, "talker.codec_head.weight")?;
        let norm = req_f32(store, "talker.model.norm.weight")?;

        let layers = load_parallel(pool, NUM_LAYERS, |i| {
            TalkerLayerWeights::load(
                store,
                &format!("talker.model.layers.{i}"),
                HIDDEN_SIZE,
                Q_DIM,
                KV_DIM,
                INTERMEDIATE_SIZE,
            )
        })?;

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            ROPE_MAX_POS,
            HEAD_DIM,
            qwen_core::config::ROPE_THETA,
        );
        let rope_cos = AlignedBuf::from_slice(&cos_v);
        let rope_sin = AlignedBuf::from_slice(&sin_v);

        if !silent {
            eprintln!(
                "  Talker: {} capas cargadas, caché KV de {} slots [INT8]",
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
        })
    }

    fn load_cp(
        store: &MmapStore,
        pool: &ComputePool,
        silent: bool,
    ) -> Result<CodePredictorWeights, EngineError> {
        if !silent {
            eprintln!(
                "Cargando pesos del Code Predictor (hidden={}, layers={})...",
                CP_HIDDEN_SIZE, CP_NUM_LAYERS
            );
        }
        let norm = req_f32(store, "talker.code_predictor.model.norm.weight")?;
        let layers = load_parallel(pool, CP_NUM_LAYERS, |i| {
            CpLayerWeights::load(
                store,
                &format!("talker.code_predictor.model.layers.{i}"),
                CP_HIDDEN_SIZE,
                CP_Q_DIM,
                CP_KV_DIM,
                CP_INTERMEDIATE_SIZE,
            )
        })?;

        let mut codec_emb_bf16 = Vec::with_capacity(15);
        for g in 0..15 {
            codec_emb_bf16.push(req_bf16(
                store,
                &format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
            )?);
        }
        let lm_heads = load_parallel(pool, 15, |g| {
            Ok(CpLmHead {
                weight: quantize_matrix_int8(
                    &req_bf16(store, &format!("talker.code_predictor.lm_head.{g}.weight"))?,
                    CODEBOOK_SIZE,
                    CP_HIDDEN_SIZE,
                ),
            })
        })?;

        let (cos_v, sin_v) = KernelDispatch::compute_rope_neox_table(
            CP_KV_MAX_POS,
            CP_HEAD_DIM,
            qwen_core::config::ROPE_THETA,
        );

        let cp = CodePredictorWeights {
            norm,
            layers,
            codec_emb_bf16,
            lm_heads,
            rope_cos: AlignedBuf::from_slice(&cos_v),
            rope_sin: AlignedBuf::from_slice(&sin_v),
        };

        if !silent {
            eprintln!(
                "  Code Predictor: {} capas cargadas, q_dim={} kv_dim={} [INT8+Q4_0-FFN]",
                CP_NUM_LAYERS,
                CP_Q_DIM,
                CP_KV_DIM
            );
        }

        Ok(cp)
    }

    fn load_decoder(store: &MmapStore, silent: bool) -> Result<SpeechDecoderWeights, EngineError> {
        if !silent {
            eprintln!("Cargando pesos del Speech Decoder...");
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
        let pre_conv_bias = req_f32(store, "decoder.pre_conv.conv.bias")?;
        let pre_conv_wk = conv_reorder_weight(&pre_conv_weight, 1024, 512, 3);

        let input_proj_weight = req_f32(store, "decoder.pre_transformer.input_proj.weight")?;
        let input_proj_bias = req_f32(store, "decoder.pre_transformer.input_proj.bias")?;
        let final_norm_weight = req_f32(store, "decoder.pre_transformer.norm.weight")?;
        let output_proj_weight = req_f32(store, "decoder.pre_transformer.output_proj.weight")?;
        let output_proj_bias = req_f32(store, "decoder.pre_transformer.output_proj.bias")?;

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
                attn_layer_scale: req_f32(
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
                ffn_layer_scale: req_f32(
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
                conv_bias: req_f32(store, &format!("decoder.upsample.{b}.0.conv.bias"))?,
                dwconv_weight: req_f32(
                    store,
                    &format!("decoder.upsample.{b}.1.dwconv.conv.weight"),
                )?,
                dwconv_bias: req_f32(
                    store,
                    &format!("decoder.upsample.{b}.1.dwconv.conv.bias"),
                )?,
                pwconv1_weight: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv1.weight"))?,
                pwconv1_bias: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv1.bias"))?,
                pwconv2_weight: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv2.weight"))?,
                pwconv2_bias: req_f32(store, &format!("decoder.upsample.{b}.1.pwconv2.bias"))?,
                norm_weight: req_f32(store, &format!("decoder.upsample.{b}.1.norm.weight"))?,
                norm_bias: req_f32(store, &format!("decoder.upsample.{b}.1.norm.bias"))?,
                gamma: req_f32(store, &format!("decoder.upsample.{b}.1.gamma"))?,
            })
        };
        let convnext = [load_cn(0)?, load_cn(1)?];

        let initial_conv_weight = req_f32(store, "decoder.decoder.0.conv.weight")?;
        let initial_conv_bias = req_f32(store, "decoder.decoder.0.conv.bias")?;
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
                let conv1_w = if b >= SD_TILE_FROM_BLOCK {
                    conv_im2col_weight(&conv1_weight, ch, ch, 7)
                } else {
                    conv_reorder_weight(&conv1_weight, ch, ch, 7)
                };
                Ok(SdResBlockWeights {
                    snake1_alpha: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act1.alpha"),
                    )?,
                    snake1_beta: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act1.beta"),
                    )?,
                    conv1_bias: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv1.conv.bias"),
                    )?,
                    conv1_w,
                    snake2_alpha: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act2.alpha"),
                    )?,
                    snake2_beta: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.act2.beta"),
                    )?,
                    conv2_weight: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv2.conv.weight"),
                    )?,
                    conv2_bias: req_f32(
                        store,
                        &format!("decoder.decoder.{bi}.block.{ri}.conv2.conv.bias"),
                    )?,
                })
            };
            Ok(SdUpsampleBlockWeights {
                snake_alpha: req_f32(store, &format!("decoder.decoder.{bi}.block.0.alpha"))?,
                snake_beta: req_f32(store, &format!("decoder.decoder.{bi}.block.0.beta"))?,
                conv_weight: req_f32(
                    store,
                    &format!("decoder.decoder.{bi}.block.1.conv.weight"),
                )?,
                conv_bias: req_f32(store, &format!("decoder.decoder.{bi}.block.1.conv.bias"))?,
                res_blocks: [load_res(0)?, load_res(1)?, load_res(2)?],
            })
        };
        let upsample_blocks = [load_ub(0)?, load_ub(1)?, load_ub(2)?, load_ub(3)?];

        let final_snake_alpha = req_f32(store, "decoder.decoder.5.alpha")?;
        let final_snake_beta = req_f32(store, "decoder.decoder.5.beta")?;
        let final_conv_weight = req_f32(store, "decoder.decoder.6.conv.weight")?;
        let final_conv_bias = req_f32(store, "decoder.decoder.6.conv.bias")?;

        if !silent {
            eprintln!("  Codebooks: 16/16 (descuantizados desde EMA)");
        }

        Ok(SpeechDecoderWeights {
            codebooks,
            rvq_first_output_proj,
            rvq_rest_output_proj,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talker_kv_grows_on_demand_and_preserves_contents() {
        let mut kv = TalkerKvCache::new();
        assert_eq!(kv.max_pos, KV_INITIAL_POSITIONS);
        let used = 7usize;
        for layer in 0..NUM_LAYERS {
            let off = layer * kv.max_pos * KV_DIM;
            for i in 0..used * KV_DIM {
                kv.k[off + i] = (layer * 31 + i) as u16;
                kv.v[off + i] = (layer * 17 + i) as u16;
            }
        }
        kv.len = used;

        kv.ensure_capacity(KV_INITIAL_POSITIONS + 1);
        assert_eq!(kv.max_pos, 2 * KV_INITIAL_POSITIONS);
        for layer in 0..NUM_LAYERS {
            let off = layer * kv.max_pos * KV_DIM;
            for i in 0..used * KV_DIM {
                assert_eq!(kv.k[off + i], (layer * 31 + i) as u16);
                assert_eq!(kv.v[off + i], (layer * 17 + i) as u16);
            }
            assert!(kv.k[off + used * KV_DIM..off + kv.max_pos * KV_DIM]
                .iter()
                .all(|&x| x == 0));
        }
    }
}
