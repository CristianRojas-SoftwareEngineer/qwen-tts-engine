# Especificación Técnica de Qwen3-TTS `0.6B` (`SPEC_06B.md`)

> Consolida la geometría fija, el mapa de tensores `safetensors`, el formato binario propio (`.qvoice` v6) y las reglas de *prefill*, muestreo y marcado de **Qwen3-TTS 0.6B Base**. Pensado para ingenieros del motor y mantenedores.

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
> **Validación de carga**: `validate_model_config` exige `hidden_size == 1024`, `intermediate_size == 3072`, `cp_hidden_size == 1024` y rechaza todo checkpoint que no sea Base (`tts_model_type == "base"` con `speaker_encoder_config` presente) con el mensaje `Este motor solo carga Qwen3-TTS 0.6B Base: <detalle>`; también rechaza un `config.json` al que le falte cualquier clave que comprueba o con `rms_norm_eps` distinto de `1e-6`.

---

## 2. Mapa de tensores (`safetensors`)

### 2.1 Modelo principal (`model.safetensors`)

- **Talker global** (`BF16`, también los biases y las norms, que el motor lee como `F32`):
  - `talker.model.text_embedding.weight` `[151936, 2048]` (`BF16`)
  - `talker.text_projection.linear_fc1.weight` `[2048, 2048]` (`BF16`), `.bias` `[2048]` (`BF16` en el checkpoint; el motor lo lee como `F32`)
  - `talker.text_projection.linear_fc2.weight` `[1024, 2048]` (`BF16`), `.bias` `[1024]` (`BF16` en el checkpoint; el motor lo lee como `F32`)
  - `talker.model.codec_embedding.weight` `[3072, 1024]` (`BF16`)
  - `talker.codec_head.weight` `[3072, 1024]` (`BF16`)
  - `talker.model.norm.weight` `[1024]` (`BF16` en el checkpoint; el motor lo lee como `F32`)
- **Talker por capa** `i = 0..27` (`talker.model.layers.{i}.*`):
  - `BF16`: `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`.
  - en el checkpoint son `BF16`; el motor los lee como `F32`: `self_attn.{q,k}_norm.weight` `[128]`, `input_layernorm.weight` `[1024]`, `post_attention_layernorm.weight` `[1024]`.
- **Code Predictor** (`talker.code_predictor.*`):
  - `talker.code_predictor.model.norm.weight` `[1024]` (`BF16` en el checkpoint; el motor lo lee como `F32`)
  - `talker.code_predictor.lm_head.{g}.weight` (`g=0..14`, `[2048, 1024]`, `BF16`)
  - `talker.code_predictor.model.codec_embedding.{g}.weight` (`g=0..14`, `[2048, 1024]`, `BF16`)
  - Por capa `i = 0..4` (`talker.code_predictor.model.layers.{i}.*`): mismos 11 tensores que Talker.
- **Speaker Encoder** (solo `0.6B-Base`; en el checkpoint son `BF16`; el motor los lee como `F32`):
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

---

## 3. Formatos binarios propios (little-endian)

### 3.1 Perfil de voz `.qvoice` (versión 6)

Único formato de voz, en `qwen-core/src/qvoice.rs`. No tiene secciones con etiqueta: todos los campos son obligatorios y van en este orden, little-endian.

| Campo | Tipo | Valor |
|---|---|---|
| `magic` | 4 B | `"QVCE"` |
| `version` | `u32` | `6` |
| `enc_dim` | `u32` | `1024` |
| `speaker_embedding` | `enc_dim × f32` | x-vector |
| `lang_id` | `u32` | `0` = sin idioma; si no, un id válido de `lookup_language` |

- **Tamaño**: `4112` bytes justos. El nombre de la voz es el del archivo (sin `.qvoice`).
- **La lectura falla** (con un mensaje que nombra la ruta) si el `magic` no coincide, la versión no es `6` (las voces anteriores se vuelven a clonar), `enc_dim != 1024`, `lang_id` es inválido, el archivo está truncado o sobran bytes al final.
- La voz `default` no es un archivo: su x-vector (`4096 B`, SHA-256 `d2ceafbb…3b94`) va embebido en el binario (`qwen-engine/assets/default_voice.xvec`).

---

## 4. Vocabulario, *prefill* y marcado

### 4.1 IDs de tokens especiales e idiomas

- **Texto** (`0..151935`): `<|endoftext|>=151643`, `<|im_start|>=151644`, `<|im_end|>=151645`, `assistant=77091`, `\n=198`, `TTS_PAD=151671`, `TTS_BOS=151672`, `TTS_EOS=151673`.
- **Control códec** (`0..3071`): `CODEC_PAD=2148`, `CODEC_BOS=2149`, `CODEC_EOS=2150`, `CODEC_THINK=2154`, `CODEC_NO_THINK=2155` (del vocabulario del checkpoint; el motor no lo usa porque el prompt siempre lleva idioma), `CODEC_THINK_BOS=2156`, `CODEC_THINK_EOS=2157`.

| Idioma (`lookup_language`) | ID |
|---|---:|
| `english` | `2050` |
| `german` | `2053` |
| `spanish` | `2054` |
| `chinese` | `2055` |
| `japanese` | `2058` |
| `french` | `2061` |
| `korean` | `2064` |
| `russian` | `2069` |
| `italian` | `2070` |
| `portuguese` | `2071` |

### 4.2 Estructura del *prefill* (`qwen-engine/src/prompt.rs`)

- **Prefijo de rol**: `[151644, 77091, 198]` (`<|im_start|>assistant\n`).
- **Idioma**: siempre hay un `lang_id`, el de la petición, el de la voz o, si no hay ninguno, español (`2054`, `DEFAULT_LANGUAGE_ID`); la voz `default` declara español. No existe el prompt sin idioma.
- **Prefijo códec**: `[2154, 2156, lang_id, 2157, <locutor>, 2148, 2149]`. La posición del locutor es explícita en el código (`Sec2Token::Speaker`, entre `CODEC_THINK_EOS` y `CODEC_PAD`); no hay token centinela.

1. **Sección 1 (3 pos)**: `text_proj(role_prefix)` sin par códec.
2. **Sección 2 (`codec_len - 1` pos)**: `tts_pad * (codec_len - 2) + tts_bos` + `codec_embed(codec[0..codec_len-1])` *(en la posición del locutor, `dst = tts_pad` y `dst[j] += spk[j]`)*.
3. **Sección 3 (1 pos)**: `text_proj(BPE(text)[0]) + codec_bos`. El *prefill* tiene así 10 posiciones.

**Texto por pasos.** El texto entra un token por paso, como en el modo por defecto de la implementación oficial (`non_streaming_mode=False`). Los `N−1` tokens restantes del texto y `tts_eos` (`N` entradas en total, con `N` el número de tokens del texto) se suman a la entrada de los pasos 0, 1, 2… del bucle, uno por paso, y a partir de ahí la entrada solo lleva `tts_pad`. Cada paso parte de su entrada habitual (los 16 códigos del frame anterior más `tts_pad`) y le suma `text_proj(token) - tts_pad` (o `tts_eos - tts_pad`), restando en `f32` elemento a elemento. Eso fija el orden de redondeo con el que se validó el audio. El límite de longitud cuenta las posiciones que necesita el texto: `10 + N <= ROPE_MAX_POS`.

### 4.3 Recorte, refuerzo y retención de EOS, muestreo y marcado

- **Recorte de silencio final (`trim_trailing_silence`)**: activo si `n_samples >= 3 * 24000`. Escanea ventanas de `20 ms` desde el final con umbral `max(0.18 * global_rms, 0.004)`, deja `+50 ms` de margen y limita el recorte al `40 %` de la duración conservando al menos `2 s`.
- **Refuerzo de EOS (`pipeline.rs`)**: `expected = text_content_len * 3`. Si `0 < text_content_len <= 11`: `start = expected * 3 / 2`, aplica `+1.0/frame` (tope `+15.0`) cuando `frame > start`. Si `> 11`: `start = expected * 2`, aplica `+0.5/frame` (tope `+10.0`) cuando `frame > start`. Suprime tokens `>= 2048` excepto `CODEC_EOS`, y suprime `CODEC_EOS` para `frame < 2`.
- **Retención tras el primer EOS (`pipeline.rs`)**: `EOS_HOLD_FRAMES = 2`. En el primer frame en que el muestreo del `code0` devuelve `CODEC_EOS`, se descarta (no entra en `prev_tokens`), se recalculan los logits de ese mismo frame con las mismas reglas, se fija el logit de `CODEC_EOS` a `-1e30` y se vuelve a muestrear con el mismo generador, sin reiniciarlo; ese código es el primer frame retenido. El EOS sigue bloqueado mientras dura la retención, y tras entregar el segundo frame retenido la síntesis termina sin dar otro paso del Talker. Nunca pasa de `max_frames`: si el primer EOS cae en el último frame permitido, se retiene un frame. `SynthesisTiming::first_eos_frame` guarda el frame del primer EOS (`None` si llegó al tope sin EOS). Decisión y evidencia en [ADR-008](adr/008-end-of-synthesis.md).
- **Muestreo (`qwen-core/src/sampling.rs`)**: temperatura por defecto `DEFAULT_TEMPERATURE = 0.325` y semilla por defecto `DEFAULT_SEED = 4` (`qwen-core/src/config.rs`); penalización de repetición sobre `prev_tokens` → si `T < 1e-6`, argmax (decodificación voraz); si no, `softmax(logits / T)` → *top-k* → *top-p* → renormalización → LCG `seed = seed.wrapping_mul(1103515245).wrapping_add(12345)`, `r = ((seed >> 16) & 0x7FFF) / 32768.0`.
- **Marcado en línea (`markup.rs`, `compose.rs`)**: `plan_script` divide el texto en tramos, también con `--stdout`, y fija el punto final (`close_final_sentence`): al texto plano o al último tramo hablado, ignorando las pausas finales. Si el último carácter que no es espacio ni signo de cierre (`"`, `'`, `»`, `”`, `’`, `)`) es `.`, `!`, `?`, `…`, `。`, `！` o `？`, el texto no cambia; si es `,`, `;` o `:`, se sustituye por `.`; en otro caso se inserta `.` tras él, conservando los espacios finales. Si hay `[pause:...]`, `[break:...]`, `[0.5]`, `|` o una muletilla (`[ahh]`, `[relief]`, `[phew]`, `[hmm]`, `[hmpf]`, `[mah]`, `[uhm]`, `[haha]`, `[heh]`, `[ouch]`, `[ahi]`, `[huff]`, `[ugh]`), `MarkupPlanner::parse_spans` divide en tramos con *prefill* frío por tramo. Un tag desconocido (por ejemplo `[xyz]`) no es marcado y queda como texto.
