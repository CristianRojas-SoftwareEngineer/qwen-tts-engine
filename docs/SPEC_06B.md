# Especificación Técnica de Qwen3-TTS `0.6B` (`SPEC_06B.md`)

> Consolida la geometría fija, el mapa de tensores `safetensors`, los formatos binarios propios (`.qvoice`/`.bin`) y las reglas de *prefill*, muestreo y paralingüística de **Qwen3-TTS 0.6B** (`CustomVoice` y `Base`). Pensado para ingenieros del motor y mantenedores.

---

## 1. Geometría fija del modelo `0.6B`

Al especializar el motor en `0.6B`, todas estas dimensiones son constantes en compilación (`qwen-core/src/config.rs`):

| Componente | Parámetro | Valor fijo `0.6B` |
|---|---|---:|
| **Global / Audio** | `sample_rate` / `frame_rate` / `hop_samples` | `24000 Hz` / `12.5 Hz` / `1920` |
| | `text_vocab_size` / `codec_vocab_size` | `151936` / `3072` |
| | `num_codebooks` / `codebook_size` / `codebook_dim` | `16` / `2048` / `256` |
| **Talker** | `text_hidden_size` / `hidden_size` | `2048` / **`1024`** |
| | `num_layers` / `num_heads` / `num_kv_heads` / `head_dim` | `28` / `16` / `8` / `128` |
| | `q_dim` / `kv_dim` / `intermediate_size` | `2048` / `1024` / **`3072`** |
| | `rms_norm_eps` / `rope_theta` / estilo RoPE | `1e-6` / `1e6` / **NeoX split-half** |
| **Code Predictor** | `cp_hidden_size` / `cp_num_layers` | **`1024`** / `5` |
| | `cp_num_heads` / `cp_num_kv_heads` / `cp_head_dim` / `cp_intermediate` | `16` / `8` / `128` / `3072` |
| | `Acoplamiento Talker ↔ CP` | **Proyección directa** (`talker_hidden_size == cp_hidden_size == 1024`) |
| **Speech Decoder** | `dec_num_layers` / `dec_hidden_size` / `dec_latent_dim` | `8` / `512` / `1024` |
| | `dec_codebook_dim` / `dec_decoder_dim` / `dec_intermediate_size` | `512` / `1536` / `1024` |
| | `dec_num_heads` / `dec_head_dim` / `dec_sliding_window` | `16` / `64` / `72` |
| | `dec_rms_norm_eps` / `dec_rope_theta` | `1e-5` / `10000.0` |
| | `dec_upsample_rates` / `dec_convnext_ratios` | `[8, 5, 4, 3]` (canales `[768, 384, 192, 96]`) / `[2, 2]` |
| **Speaker Encoder** (`Base`) | `enc_dim` / `mel_dim` | `1024` / `128` |

> **RoPE NeoX split-half**: `inv_freq[i] = 1.0 / theta^(2*i / head_dim)` para `i = 0..head_dim/2 - 1`, rotando pares `(x[i], x[i + head_dim/2])`.
> **Validación de carga**: `ModelConfig06B::load_and_validate` exige `hidden_size == 1024`, `intermediate_size == 3072`, `cp_hidden_size == 1024` y rechaza `VoiceDesign` o checkpoints `1.7B` (`tts_model_size != "0b6"`).

---

## 2. Mapa de tensores (`safetensors`)

### 2.1 Modelo principal (`model.safetensors` o shards vía `model.safetensors.index.json`)

- **Talker global** (`BF16`, salvo biases/norm en `F32`):
  - `talker.model.text_embedding.weight` `[151936, 2048]` (`BF16`)
  - `talker.text_projection.linear_fc1.weight` `[2048, 2048]` (`BF16`), `.bias` `[2048]` (`F32`)
  - `talker.text_projection.linear_fc2.weight` `[1024, 2048]` (`BF16`), `.bias` `[1024]` (`F32`)
  - `talker.model.codec_embedding.weight` `[3072, 1024]` (`BF16`)
  - `talker.codec_head.weight` `[3072, 1024]` (`BF16`)
  - `talker.model.norm.weight` `[1024]` (`F32`)
- **Talker por capa** `i = 0..27` (`talker.model.layers.{i}.*`):
  - `BF16`: `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`.
  - `F32`: `self_attn.{q,k}_norm.weight` `[128]`, `input_layernorm.weight` `[1024]`, `post_attention_layernorm.weight` `[1024]`.
- **Code Predictor** (`talker.code_predictor.*`):
  - `talker.code_predictor.model.norm.weight` `[1024]` (`F32`)
  - `talker.code_predictor.lm_head.{g}.weight` (`g=0..14`, `[2048, 1024]`, `BF16`)
  - `talker.code_predictor.model.codec_embedding.{g}.weight` (`g=0..14`, `[2048, 1024]`, `BF16`)
  - Por capa `i = 0..4` (`talker.code_predictor.model.layers.{i}.*`): mismos 11 tensores que Talker.
- **Speaker Encoder** (solo `0.6B-Base`, `F32`):
  - `speaker_encoder.blocks.0.conv.{weight,bias}` (`128→512, k=5`)
  - `speaker_encoder.blocks.{1..3}.*` (`tdnn1`, `res2net_block.blocks.{0..6}`, `tdnn2`, `se_block.conv{1,2}`; dilataciones `{2, 3, 4}`)
  - `speaker_encoder.mfa.conv.{weight,bias}`, `speaker_encoder.asp.{tdnn,conv}.{weight,bias}`, `speaker_encoder.fc.{weight,bias}` (`3072→1024`).

### 2.2 Speech Tokenizer (`speech_tokenizer/model.safetensors`, todo `F32`)

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

## 3. Formatos binarios propios (Little-Endian)

### 3.1 Perfil de voz `.qvoice` (versión 3)

Secuencia binaria en `qwen-core/src/qvoice.rs`:

1. **Cabecera base**:
   - `magic`: `4 B` = `"QVCE"`
   - `version`: `u32` (`3`; cualquier otra versión es rechazada)
   - `enc_dim`: `u32` (`1024` en `0.6B`)
   - `speaker_embedding`: `enc_dim × f32` (`4096 B`)
   - `ref_text_len`: `u32` + `ref_text_len` bytes UTF-8 (sin `\0`)
   - `n_ref_frames`: `u32` (`<= 1_000_000`) + `n_ref_frames × 16 × i32` (`cached_ref_codes`)
2. **Sección `META`** (opcional): `"META"` (`4 B`), `lang_id` (`u32`), `lang_name` (`[u8; 16]`), `model_size` (`u32`, `1024` en `0.6B`), `enc_dim` (`u32`), `ref_dur` (`f32`), `voice_name` (`[u8; 64]`), `flags` (`u32`: bit0=`xvector_only`, bit1=`has_icl`, bit2=`is_base_model`).
3. **Sección `TPAD`** (opcional): `"TPAD"` (`4 B`), `hidden` (`u32=1024`), `tts_pad_embed` (`1024×f32`), `tts_bos_embed` (`1024×f32`), `tts_eos_embed` (`1024×f32`).
4. **Sección `WOVR`** (opcional; se carga también con `--icl-only`, salvo `QWEN_GRAFT_NO_WOVR`): `"WOVR"` (`4 B`), `wh` (`u32=1024`), `wth` (`u32=2048`), `wcv` (`u32=2048`; rango válido `1..=6144`), seguido de `fc1` (`2048×2048×u16`), `fc1_bias` (`2048×f32`), `fc2` (`1024×2048×u16`), `fc2_bias` (`1024×f32`), `codec_emb[0..2048]` (`2048×1024×u16`). Si falta `TPAD`, recalcula `tts_pad/bos/eos`.
5. **Sección `WDLT`** (opcional; omitida con `--icl-only` o `--graft`):
   - `"WDLT"` (`4 B`), `target_hidden` (`u32=1024`), `n_tensors` (`u32`).
   - Por tensor: `name_len` (`u16`), `tname`, `data_bytes` (`u32`), `dtype_flag` (`u8`: `4` = delta `i16` LZ4 sobre bits BF16; `0` = payload crudo BF16/F32), `compressed_size` (`u32`), `payload`.
   - Reconstrucción `dtype_flag == 4`: `bf16_result[i] = (cv_orig[i] as i32 + delta16[i] as i32) as u16`. Con `--voice-strength < 1.0` (finito), interpola linealmente en F32 (`vc + alpha * (vcl - vc)`) solo en proyecciones `self_attn.*_proj.weight` y `mlp.*_proj.weight` de `talker.model.layers.{vs_l0..vs_l1}` (default `14..27`).

### 3.2 Archivos `.bin` y activos `emovoice` (`0.6B`)

- **`.bin` *x-vector***: exactamente `1024 × f32` (`4096 B`) raw little-endian.
- **Resolución `emovoice` (`qwen-engine/src/emotion.rs`)**:
  1. Busca junto a `--load-voice` y luego en `presets/emovoice/` un archivo `<voice_key>_<tok>.qvoice` (`--icl-only`) o `<voice_key>_<tok>.bin` (`--xvector-only`).
  2. Si no existe y hay un clon activo, busca `presets/emovoice/dir_<tok>.bin` (`1024 × f32`) y aplica desplazamiento preservando la norma L2 original `n0`:
     `emb_new = normalize_to(emb + emotion_strength * n0 * dir, n0)`.
- **Alias de emociones (`qwen-core/src/markup.rs`)**:
  `sad`/`sadness` → `sad`, `joy`/`happy`/`joyful` → `joy`, `anger`/`angry`/`rage` → `ang`, `fear`/`afraid` → `fear`, `disgust`/`disgusted` → `disgust`, `surprise`/`surprised` → `surprise`, `contempt`/`scorn` → `contempt`, `awe`/`wonder` → `awe`, `nostalgia`/`wistful` → `nostalgia`, `disapproval` → `disapproval`, `remorse`/`regret` → `remorse`, `outrage` → `outrage`, `despair` → `despair`.

---

## 4. Vocabulario, *prefill* y paralingüística

### 4.1 IDs de tokens especiales, idiomas y speakers

- **Texto** (`0..151935`): `<|endoftext|>=151643`, `<|im_start|>=151644`, `<|im_end|>=151645`, `assistant=77091`, `\n=198`, `TTS_PAD=151671`, `TTS_BOS=151672`, `TTS_EOS=151673`.
- **Control códec** (`0..3071`): `CODEC_PAD=2148`, `CODEC_BOS=2149`, `CODEC_EOS=2150`, `CODEC_THINK=2154`, `CODEC_NO_THINK=2155`, `CODEC_THINK_BOS=2156`, `CODEC_THINK_EOS=2157`.

| Idioma (`lookup_language`) | ID | Speaker (`PRESET_SPEAKERS`) | ID | Idioma / Género |
|---|---:|---|---:|---|
| `english` | `2050` | `ryan` | `3061` | `English` / `male` |
| `german` | `2053` | `aiden` | `2861` | `English` / `male` |
| `spanish` | `2054` | `vivian` | `3065` | `Chinese` / `female` |
| `chinese` | `2055` | `serena` | `3066` | `Chinese` / `female` |
| `japanese` | `2058` | `uncle_fu` | `3010` | `Chinese` / `male` |
| `french` | `2061` | `dylan` | `2878` | `Chinese` / `male` |
| `sichuan_dialect` | `2062` | `eric` | `2875` | `Chinese` / `male` |
| `korean` | `2064` | `ono_anna` | `2873` | `Japanese` / `female` |
| `russian` | `2069` | `sohee` | `2864` | `Korean` / `female` |
| `italian` | `2070` | | | |
| `portuguese` | `2071` | | | |
| `beijing_dialect` | `2074` | | | |

### 4.2 Estructura del *prefill* (`qwen-engine/src/pipeline.rs`)

- **Role prefix**: `[151644, 77091, 198]` (`<|im_start|>assistant\n`).
- **Prefijo códec**: con idioma `[2154, 2156, lang_id, 2157, speaker_o_-1, 2148, 2149]`; sin idioma `[2155, 2156, 2157, speaker_o_-1, 2148, 2149]`.

1. **Sección 1 (3 pos)**: `text_proj(role_prefix)` sin par códec.
2. **Sección 2 (`codec_len - 1` pos)**: `tts_pad * (codec_len - 2) + tts_bos` + `codec_embed(codec[0..codec_len-1])` *(cuando `speaker == -1`, inyecta `speaker_embedding` reescalado a la norma L2 de `ryan=3061` multiplicable por `QWEN_SPK_SCALE`)*.
3. **Modo normal (sin ICL)**:
   - **Sección 3 (`N + 1` pos)**: `text_proj(BPE(text)) + tts_eos` + `codec_pad`.
   - **Sección 4 (1 pos)**: `tts_pad + codec_bos`.
4. **Modo ICL (`--ref-text` + `ref_codes`, o `--emo-ref`)**:
   - **Sección 3' (`N_ref + N + 1` pos)**: `text_proj(BPE(ref_text) + BPE(text)) + tts_eos` + `codec_pad`.
   - **Sección 4' (`n_ref_frames + 1` pos)**: cada posición lleva `tts_pad` sumado con `codec_bos` en `pos 0` y, en `pos 1..n_ref_frames`, con `talker.codec_embedding(ref_codes[f][0]) + sum_{g=1..15} cp_codec_emb[g-1](ref_codes[f][g])`. *(En modo ICL se recortan los primeros `QWEN_ICL_TRIM_FRAMES = 2` fotogramas = `3840` muestras al inicio del audio generado.)*

### 4.3 Recorte, EOS boosting, muestreo y marcado

- **Recorte de silencio final (`trim_trailing_silence`)**: activo si `n_samples >= 3 * 24000` y no existe `QWEN_NO_REF_TRIM`. Escanea ventanas de `20 ms` desde el final con umbral `max(0.18 * global_rms, 0.004)`, deja `+50 ms` de margen y limita el recorte al `40 %` de la duración conservando al menos `2 s`.
- **EOS boosting (`pipeline.rs`)**: `expected = text_content_len * 3`. Si `0 < text_content_len <= 11`: `start = expected * 3 / 2`, aplica `+1.0/frame` (tope `+15.0`) cuando `frame > start`. Si `> 11`: `start = expected * 2`, aplica `+0.5/frame` (tope `+10.0`) cuando `frame > start`. Suprime tokens `>= 2048` excepto `CODEC_EOS`, y suprime `CODEC_EOS` para `frame < 2`.
- **Muestreo (`qwen-core/src/sampling.rs`)**: temperatura por defecto `DEFAULT_TEMPERATURE = 0.325` (`qwen-core/src/config.rs`); penalización de repetición sobre `prev_tokens` → si `T < 1e-6`, argmax greedy; si no, `softmax(logits / T)` → *top-k* → *top-p* → renormalización → LCG `seed = seed.wrapping_mul(1103515245).wrapping_add(12345)`, `r = ((seed >> 16) & 0x7FFF) / 32768.0`.
- **Paralingüística inline y `--compose` (`markup.rs`, `compose.rs`)**:
  - Paso 1 (`para_substitute`): sustituye `[laugh]`/`[laughs]` (`"哈哈哈"`, `seed=2024, T=1.1`), `[sigh]`/`[sighs]` (`"唉"`, `seed=42, T=1.1` o `"ahh"`, `seed=7` para `vivian`), `[yawn]`/`[yawns]` (`"哈啊"`, `seed=42, T=1.1`), `[wow]` (`"哇"`, `seed=2024, T=1.1`), `[giggle]`/`[giggles]` (`"嘿嘿"`, `seed=42, T=1.1`) y `[scoff]` (`"切"`, `seed=42, T=1.0`).
  - Paso 2 (`MarkupPlanner::parse_spans`): si persisten `[pause:...]`, `[break:...]`, `[0.5]`, `|` o `--compose`, divide en tramos con *prefill* frío por tramo. En `0.6B`, los tags `[emotion]` dividen el tramo sin aplicar steering.
