#[cfg(target_arch = "aarch64")]
pub use crate::simd::scalar::{
    apply_rope_neox_inplace, argmax_matvec_bf16, bf16_accum_f32, bf16_matmat_slice,
    bf16_matvec_slice, bf16_to_f32_vec, causal_attention_bf16kv_heads, causal_attention_f32_heads,
    f32_matvec_accum, f32_to_bf16_vec, int8_matmat_slice, int8_matvec_slice, q4_0_matmat_slice,
    q4_0_matvec_slice, quantize_act_int8, rms_norm, rms_norm_per_head, rms_norm_residual,
    snake_row,
};
