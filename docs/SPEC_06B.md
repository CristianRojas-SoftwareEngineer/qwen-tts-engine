# Especificación Técnica de Qwen3-TTS `0.6B` (`SPEC_06B.md`)

Consolida la geometría exacta, tensores, formatos binarios propios y construcción de secuencias del modelo **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`) implementados en `crates/`.

---

## 1. Geometría Fija del Modelo `0.6B`

Al restringir el motor al `0.6B`, todas estas dimensiones son constantes conocidas en compilación (sin `cp_mtp_proj`):

| Componente | Parámetro | Valor fijo `0.6B` |
|---|---|---:|
| **Global / Audio** | `sample_rate` / `frame_rate` / `hop_samples` | `24000 Hz` / `12.5 Hz` / `1920` |
| | `text_vocab_size` / `codec_vocab_size` | `151936` / `3072` |
| | `num_codebooks` / `codebook_size` / `codebook_dim` | `16` / `2048` / `256` |
| **Talker** | `text_hidden_size` / `hidden_size` | `2048` / **`1024`** |
| | `num_layers` / `num_heads` / `num_kv_heads` / `head_dim` | `28` / `16` / `8` / `128` |
| | `q_dim` / `kv_dim` / `intermediate_size` | `2048` / `1024` / **`3072`** |
| | `rms_norm_eps` / `rope_theta` / estilo RoPE | `1e-6` / `1e6` / **NeoX split-half** |

> **Nota crítica sobre RoPE**: Aunque algunas descripciones externas mencionan M-RoPE *interleaved*, el modelo `0.6B` usa **RoPE estilo NeoX split-half** (`qwen-kernels`): `inv_freq[i] = 1.0 / theta^(2*i / head_dim)` para `i = 0..head_dim/2 - 1`, rotando pares `(x[i], x[i + head_dim/2])`.
| **Code Predictor** | `cp_hidden_size` / `cp_num_layers` | **`1024`** / `5` |
| | `cp_num_heads` / `cp_num_kv_heads` / `cp_head_dim` / `cp_intermediate` | `16` / `8` / `128` / `3072` |
| | `small_to_mtp_projection` | **No existe en 0.6B** (`talker_h == cp_h`) |
| **Speech Decoder** | `dec_num_layers` / `dec_hidden_size` / `dec_latent_dim` | `8` / `512` / `1024` |
| | `dec_codebook_dim` / `dec_decoder_dim` / `dec_intermediate_size` | `512` / `1536` / `1024` |
| | `dec_num_heads` / `dec_head_dim` / `dec_sliding_window` | `16` / `64` / `72` |
| | `dec_rms_norm_eps` / `dec_rope_theta` | `1e-5` / `10000.0` |
| | `dec_upsample_rates` / `dec_convnext_ratios` | `[8, 5, 4, 3]` (canales `[768, 384, 192, 96]`) / `[2, 2]` |
| **Speaker Encoder** (`Base`) | `enc_dim` / `mel_dim` | `1024` / `128` |

> La carga debe validar `hidden_size == 1024`, `intermediate_size == 3072`, `cp_hidden_size == 1024` y rechazar explícitamente checkpoints `1.7B` (`2048`) o `VoiceDesign` (`spk_id: {}`).

---

## 2. Mapa de Tensores (`safetensors`)

### 2.1 Modelo principal (`model.safetensors` o shards vía `model.safetensors.index.json`)
- **Talker global** (BF16 salvo biases/norm en F32):
  - `talker.model.text_embedding.weight` `[151936, 2048]` (BF16)
  - `talker.text_projection.linear_fc1.weight` `[2048, 2048]` (BF16), `.bias` `[2048]` (F32)
  - `talker.text_projection.linear_fc2.weight` `[1024, 2048]` (BF16), `.bias` `[1024]` (F32)
  - `talker.model.codec_embedding.weight` `[3072, 1024]` (BF16)
  - `talker.codec_head.weight` `[3072, 1024]` (BF16)
  - `talker.model.norm.weight` `[1024]` (F32)
- **Talker por capa** `i = 0..27` (`talker.model.layers.{i}.*`):
  - BF16: `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`
  - F32: `self_attn.{q,k}_norm.weight` `[128]`, `input_layernorm.weight` `[1024]`, `post_attention_layernorm.weight` `[1024]`
- **Code Predictor** (`talker.code_predictor.*`):
  - `talker.code_predictor.model.norm.weight` `[1024]` (F32)
  - `talker.code_predictor.lm_head.{g}.weight` (`g=0..14`, `[2048, 1024]`, BF16)
  - `talker.code_predictor.model.codec_embedding.{g}.weight` (`g=0..14`, `[2048, 1024]`, BF16)
  - Por capa `i = 0..4` (`talker.code_predictor.model.layers.{i}.*`): mismos 11 tensores que Talker.
- **Speaker Encoder** (solo `0.6B-Base`, F32):
  - `speaker_encoder.blocks.0.conv.{weight,bias}` (`128→512, k=5`)
  - `speaker_encoder.blocks.{1..3}.*` (`tdnn1`, `res2net_block.blocks.{0..6}`, `tdnn2`, `se_block.conv{1,2}`; dilataciones `{2, 3, 4}`)
  - `speaker_encoder.mfa.conv.{weight,bias}`, `speaker_encoder.asp.{tdnn,conv}.{weight,bias}`, `speaker_encoder.fc.{weight,bias}` (`3072→1024`).

### 2.2 Speech Tokenizer (`speech_tokenizer/model.safetensors`, todo F32)
- **Decoder**:
  - Codebook 0: `decoder.quantizer.rvq_first.vq.layers.0._codebook.{embedding_sum,cluster_usage}` (EMA: `emb = embedding_sum / max(usage, 1e-5)`).
  - Codebooks 1–15: `decoder.quantizer.rvq_rest.vq.layers.{0..14}._codebook.{embedding_sum,cluster_usage}`.
  - `decoder.quantizer.{rvq_first,rvq_rest}.output_proj.weight`
  - `decoder.pre_conv.conv.{weight,bias}` `[1024, 512, 3]`
  - `decoder.pre_transformer.{input_proj,output_proj}.{weight,bias}`, `norm.weight`, y 8 capas `layers.{0..7}.{input_layernorm.weight,self_attn.{q,k,v,o}_proj.weight,self_attn_layer_scale.scale,post_attention_layernorm.weight,mlp.{gate,up,down}_proj.weight,mlp_layer_scale.scale}`.
  - 2 bloques ConvNeXt: `decoder.upsample.{0,1}.0.conv.{weight,bias}` y `.1.{dwconv.conv,pwconv1,pwconv2,norm}.{weight,bias}` + `.1.gamma`.
  - Red causal: `decoder.decoder.0.conv.{weight,bias}`, 4 bloques `decoder.decoder.{1..4}` (`block.0.{alpha,beta}`, `block.1.conv.{weight,bias}`, 3 residuales `block.{2..4}.{act1.{alpha,beta},conv1.conv.{weight,bias},act2.{alpha,beta},conv2.conv.{weight,bias}}`), `decoder.decoder.5.{alpha,beta}`, `decoder.decoder.6.conv.{weight,bias}`.
- **Encoder** (solo cuando se usa `--ref-audio` o `--emo-ref` con ICL):
  - `encoder.encoder.layers.{0,3,6,9,12,14}.conv.{weight,bias}` y bloques residuales `layers.{1,4,7,10}.block.{1,3}.conv.{weight,bias}`.
  - `encoder.encoder_transformer.layers.{0..7}.*` (`input_layernorm`, `self_attn.{q,k,v,o}_proj`, `self_attn_layer_scale`, `post_attention_layernorm`, `mlp.{fc1,fc2}`, `mlp_layer_scale`).
  - `encoder.downsample.conv.weight`, `encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer.input_proj.weight`.

---

## 3. Formatos Binarios Propios (Little-Endian)

### 3.1 Perfil de voz `.qvoice` (versión 3)
Secuencia secuencial en disco (`qwen-core/src/qvoice.rs`):
1. **Cabecera base**:
   - `magic`: `4B` = `"QVCE"`
   - `version`: `u32` (`3`; versiones anteriores son rechazadas)
   - `enc_dim`: `u32` (en `0.6B` debe ser `1024`)
   - `speaker_embedding`: `enc_dim × f32` (`4096 B`)
   - `ref_text_len`: `u32` + `ref_text_len` bytes UTF-8 (sin `\0`)
   - `n_ref_frames`: `u32` (`<= 1_000_000` por seguridad) + `n_ref_frames * 16 * i32` (`cached_ref_codes`)
2. **Sección `META`** (opcional):
   - `"META"` (`4B`), `lang_id` (`u32`), `lang_name` (`[u8; 16]`), `model_size` (`u32`, `1024` para `0.6B`), `enc_dim` (`u32`), `ref_dur` (`f32`), `voice_name` (`[u8; 64]`), `flags` (`u32`: bit0=`xvector_only`, bit1=`has_icl`, bit2=`is_base_model`).
3. **Sección `TPAD`** (opcional):
   - `"TPAD"` (`4B`), `hidden` (`u32` = `1024`), `tts_pad_embed` (`1024×f32`), `tts_bos_embed` (`1024×f32`), `tts_eos_embed` (`1024×f32`).
4. **Sección `WOVR`** (opcional — **se carga incluso con `--icl-only`**, salvo `QWEN_GRAFT_NO_WOVR`):
   - `"WOVR"` (`4B`), `wh` (`u32=1024`), `wth` (`u32=2048`), `wcv` (`u32=2048`).
   - Payload: `fc1` (`2048×2048×u16`), `fc1_bias` (`2048×f32`), `fc2` (`1024×2048×u16`), `fc2_bias` (`1024×f32`), `codec_emb[0..2048]` (`2048×1024×u16`). Si no hubo `TPAD`, recalcula `tts_pad/bos/eos`.
5. **Sección `WDLT`** (opcional; **omitida si `--icl-only`**):
   - `"WDLT"` (`4B`) + `target_hidden` (`u32`, debe ser `1024` o falla) + `n_tensors` (`u32`).
   - Por cada tensor: `name_len` (`u16`), `tname` (`name_len B`), `data_bytes` (`u32`), `dtype_flag` (`u8`: `4` = delta `i16` comprimido con LZ4 respecto a los bits BF16 originales del `CustomVoice`; `0` = payload BF16/F32 sin comprimir), `compressed_size` (`u32`), payload (`compressed_size B`).
   - Reconstrucción de `WDELTA` (`dtype_flag == 4`): `bf16_result[i] = (uint16_t)(cv_orig[i] + delta16[i])`. Si `--voice-strength < 1.0`, se interpola linealmente en F32 (`vc + alpha * (vcl - vc)`) solo en las proyecciones `self_attn.*_proj.weight` y `mlp.*_proj.weight` de `talker.model.layers.{vs_l0..vs_l1}` (default `14..27`).

### 3.2 Archivos `.bin` y `emovoice` (`0.6B`)
- **`.bin` x-vector**: exactamente `1024 × f32` (`4096 B`) raw little-endian.
- **Emociones `0.6B`** (`qwen-engine/src/emotion.rs`):
  1. Busca primero junto a `--load-voice` y luego en `presets/emovoice/` un activo `<voice_key>_<tok>.qvoice` (cargado como `--icl-only`) o `<voice_key>_<tok>.bin` (`--xvector-only`).
  2. Si no existe y hay un clon cargado, busca `presets/emovoice/dir_<tok>.bin` (`1024 × f32`) y lo suma preservando la norma L2 del embedding original `n0`:
     `emb_new = normalize_to(emb + emotion_strength * n0 * dir, n0)`.
  3. **Tabla de nombres y alias (`qwen-core/src/markup.rs`)**:
     `sad`/`sadness`→`sad`, `joy`/`happy`/`joyful`→`joy`, `anger`/`angry`/`rage`→`ang`, `fear`/`afraid`→`fear`, `disgust`/`disgusted`→`disgust`, `surprise`/`surprised`→`surprise`, `contempt`/`scorn`→`contempt`, `awe`/`wonder`→`awe`, `nostalgia`/`wistful`→`nostalgia`, `disapproval`→`disapproval`, `remorse`/`regret`→`remorse`, `outrage`→`outrage`, `despair`→`despair`.

---

## 4. Vocabulario, Prompting y Paralingüística (`0.6B`)

### 4.1 IDs de tokens especiales y codec
- **Texto** (`0..151935`): `<|endoftext|>=151643`, `<|im_start|>=151644`, `<|im_end|>=151645`, `assistant=77091`, `\n=198`, `TTS_PAD=151671`, `TTS_BOS=151672`, `TTS_EOS=151673`.
- **Codec control** (`0..3071`): `CODEC_PAD=2148`, `CODEC_BOS=2149`, `CODEC_EOS=2150`, `CODEC_THINK=2154`, `CODEC_NO_THINK=2155`, `CODEC_THINK_BOS=2156`, `CODEC_THINK_EOS=2157`.
- **Idiomas**: `English=2050`, `German=2053`, `Spanish=2054`, `Chinese=2055`, `Japanese=2058`, `French=2061`, `Korean=2064`, `Russian=2069`, `Italian=2070`, `Portuguese=2071` (`config.json` también incluye `sichuan_dialect=2062`, `beijing_dialect=2074`).
- **Speakers CustomVoice**: `aiden=2861`, `sohee=2864`, `ono_anna=2873`, `eric=2875`, `dylan=2878`, `uncle_fu=3010`, `ryan=3061`, `vivian=3065`, `serena=3066`.

### 4.2 Estructura del Prefill (`qwen-engine/src/pipeline.rs`)
Secuencia de texto: `[151644, 77091, 198] + BPE(text) + [151645, 198, 151644, 77091, 198]`.
Prefijo codec (`codec_tokens`): con idioma `[2154, 2156, lang_id, 2157, speaker_o_-1, 2148, 2149]`; sin idioma `[2155, 2156, 2157, speaker_o_-1, 2148, 2149]`.

1. **Sección 1 (3 pos)**: `text_proj(role_prefix)` sin par codec.
2. **Sección 2 (`codec_len - 1` pos)**: `tts_pad * (codec_len - 2) + tts_bos` **+** `codec_embed(codec[0..codec_len-2])` *(en posición `speaker == -1` se inyecta `speaker_embedding` reescalado a la norma L2 del preset `ryan=3061` en `talker.model.codec_embedding`, multiplicable por `QWEN_SPK_SCALE`)*.
3. **Modo normal (sin ICL)**:
   - **Sección 3 (`N + 1` pos)**: `text_proj(BPE(text)) + tts_eos` **+** `codec_pad`.
   - **Sección 4 (1 pos)**: `tts_pad + codec_bos`.
4. **Modo ICL (`--ref-text` + `ref_codes`, o `--emo-ref`)**:
   - **Sección 3' (`N_ref + N + 1` pos)**: `text_proj(BPE(ref_text) + BPE(text)) + tts_eos` **+** `codec_pad`.
   - **Sección 4' (`n_ref_frames + 1` pos)**: `tts_pad` **+** `codec_bos` seguido de `talker.codec_embedding(ref_codes[f][0]) + sum_{g=1..15} cp_codec_emb[g-1](ref_codes[f][g])`. *(En modo ICL se recortan los primeros `QWEN_ICL_TRIM_FRAMES` = `2` fotogramas = 3840 muestras de audio generado al inicio para evitar transitorio "tud".)*

### 4.3 Reglas de generación, muestreo, recorte y paralingüística `0.6B`
- **Recorte de silencio final en audio de referencia (`trim_trailing_silence`)**: antes de ECAPA/RVQ (si `n_samples >= 3 * 24000` y no existe `QWEN_NO_REF_TRIM`), escanea ventanas de `20 ms` desde el final con umbral `max(0.18 * global_rms, 0.004)`, añade `+50 ms` de margen y limita el recorte a como máximo `40 %` de la duración conservando al menos `2 s`.
- **EOS boosting** (`pipeline.rs`): `expected = text_content_len * 3`. Si `text_content_len <= 11`: `start = expected * 3 / 2`, `+1.0/frame` (tope `+15.0`). Si `> 11`: `start = expected * 2`, `+0.5/frame` (tope `+10.0`). Se suprimen tokens `>= 2048` excepto `CODEC_EOS`, y `CODEC_EOS` se suprime en `frame < 2`.
- **Pipeline de muestreo y LCG (`qwen-core/src/sampling.rs`)**:
  1. Penalización de repetición sobre `prev_tokens`: si `logit > 0`, `logit /= rep_penalty`; si `<= 0`, `logit *= rep_penalty`.
  2. Si `temperature < 1e-6`, argmax greedy; si no: `softmax(logits / temperature)` → `top_k` (quickselect) → `top_p` (ordenación parcial por selección sobre índices `idx[]` hasta `cumsum > p`) → renormalización → muestreo con LCG `seed = seed.wrapping_mul(1103515245).wrapping_add(12345)`, `r = ((seed >> 16) & 0x7FFF) as f32 / 32768.0`.
- **Paralingüística inline y `--compose` en `0.6B` (`qwen-core/src/markup.rs`, `qwen-engine/src/compose.rs`)**:
  - Paso 1 (`para_substitute`, antes del compositor salvo `--no-compose`): sustituye `[tag]` por onomatopeya delimitada por comas en una sola generación y fija la semilla/temperatura de la primera etiqueta:
    - `[laugh]`/`[laughs]` → `"哈哈哈"` (`seed=2024`, `T=1.1`)
    - `[sigh]`/`[sighs]` → `"唉"` (`seed=42`, `T=1.1`) o `"ahh"` (`seed=7` si `vivian`)
    - `[yawn]`/`[yawns]` → `"哈啊"` (`seed=42`, `T=1.1`)
    - `[wow]` → `"哇"` (`seed=2024`, `T=1.1`)
    - `[giggle]`/`[giggles]` → `"嘿嘿"` (`seed=42`, `T=1.1`)
    - `[scoff]` → `"切"` (`seed=42`, `T=1.0`)
  - Paso 2 (`MarkupPlanner::parse_spans`, si quedan `[pause:...]`, `[break:...]`, `[0.5]`, `|` o `--compose`): divide en tramos independientes (prefill frío por tramo) y concatena silencios/audios. En `0.6B`, los tags `[emotion]` dentro de `--compose` no inyectan steering (`hidden_size < 2048`, se sintetizan neutros).
