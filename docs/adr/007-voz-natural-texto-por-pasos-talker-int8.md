# ADR-007: Voz más natural: texto por pasos, Talker INT8 y clonación solo por timbre

Reemplaza parcialmente a [ADR-001](001-only-06b.md) y [ADR-006](006-single-synthesis-path.md) en lo que toca al formato `.qvoice` y a la clonación con transcripción.

> Registro de decisión sobre los cambios que acercan la voz del motor a la de la implementación oficial: el texto entra paso a paso, el Talker calcula en `INT8` y la clonación pasa a depender solo del x-vector.

- **Estado**: Aceptado
- **Fecha**: 2026-11-22

---

## Contexto

Frente a la implementación oficial (paquete `qwen_tts` de `QwenLM/Qwen3-TTS`), el motor tenía tres defectos audibles:

- **Pausas largas a mitad de frase.** El texto entero iba en el prompt, mientras que el modelo está entrenado con el texto entregado paso a paso.
- **Pronunciación degradada y entonación plana.** El decodificado paso a paso del Talker usaba pesos `Q4_0`, que se alejan demasiado de los `BF16`.
- **Ruido grave al inicio** y voz más plana y robótica en las voces clonadas con transcripción (ICL).

El caso de uso es local: una síntesis a la vez en un equipo modesto, con la CPU como único recurso, y sin usuarios externos, de modo que no se conserva ningún camino heredado.

---

## Decisiones

1. **Texto por pasos, solo para el x-vector.** El prompt tiene el prefijo de rol, la sección del locutor y una sola posición: el primer token del texto más `codec_bos` (10 posiciones). Los demás tokens y `tts_eos` se suman a la entrada de los pasos 0, 1, 2…, uno por paso, sobre el `tts_pad` que ya lleva la entrada. Es el modo por defecto de la implementación oficial (`non_streaming_mode=False`).
   - **Evidencia:** con 16 semillas, la mediana del silencio interno más largo baja de 0,92 s a 0,14 s (Mann-Whitney bilateral, p = 0,0001), y el TTFA de unos 698 ms a unos 406 ms. El usuario lo confirmó de oído.
2. **Talker en `INT8` en el decodificado paso a paso.** `wq`, `wk`, `wv`, `wo`, `gate_up` y `down` en `INT8`, con una escala por fila. El prefill sigue en `BF16`.
   - **Precisión:** se forzaron en el motor los códigos de la implementación oficial y se comparó lo que cada versión del Talker predice frente a ella (mediana de la diferencia absoluta de logits): 2,75 en `Q4_0`, 1,04 en `INT8` y 0,06 en `BF16`.
   - **Velocidad**, frase larga con texto por pasos: RTF 0,62 en `Q4_0`, 0,65 en `INT8` (el mismo que antes) y 0,80 en `BF16`.
   - El usuario lo eligió de oído.
   - **Memoria:** se aceptan unos 185 MiB más de RAM. El Talker tiene 440,4 M pesos, que ocupan unos 236 MiB en `Q4_0` y unos 422 MiB en `INT8` (cálculo con las dimensiones de `qwen-core/src/config.rs`).
3. **Se conserva el refuerzo del EOS**, con un comentario que explica qué evita.
   - **En frases normales no actúa:** con texto por pasos y Talker `INT8`, el PCM es idéntico con y sin él en 48 de 48 síntesis y en la frase larga española de referencia.
   - **En textos muy cortos sí actúa:** son los tramos sueltos del marcado, como «Hola.» o el «Hmmm...» de `[hmm]`. Ahí el propio modelo apenas propone terminar, también en la implementación oficial: con su configuración por defecto (8 semillas), el «Hmmm...» a veces no se detiene y repite.
   - **El motor alineado con el oficial reproduce lo mismo** (sin diferencias, p ≥ 0,07). Ni la cuantización ni el prompt lo causan.
   - **Sin el refuerzo**, a T 0,325, la probabilidad del EOS en el «Hmmm...» se queda entre 1e-6 y 1e-18 durante 63 frames. El refuerzo es lo único que la hace muestreable: acorta el «hmm» de 29 a 22 frames (p = 0,004) y la pausa de 40 a 34 (p = 0,016).
   - **De oído:** el usuario oye un solo «hmm» con el refuerzo y dos sin él.
   - El tope fijo sigue siendo `min(DEFAULT_MAX_FRAMES, ROPE_MAX_POS − prefill_len)`. No se añade un tope proporcional.
4. **Fuera el reescalado del x-vector.** La norma de la fila 3061 del embedding de códec es 0,0157, menor que 0,1, así que la condición daba siempre `scale = 1.0`: era código muerto.
5. **Fuera la clonación ICL entera.** En la escucha final, el x-vector clona mejor y suena más natural. El ICL suena más plano y robótico y añade un ruido grave al inicio.
6. **`.qvoice` versión 6:** `magic` `QVCE`, `version` = 6 (`u32`), `enc_dim` = 1024 (`u32`), x-vector (`1024 × f32`) y `lang_id` (`u32`); 4112 bytes justos, little-endian. Cualquier otra versión se rechaza pidiendo clonar de nuevo.
7. **Se conserva el recorte del silencio final de la referencia** (`trim_trailing_silence` en `load_ref_audio`). La implementación oficial no recorta; de oído, el clon de una grabación con silencio final suena más consistentemente bien con el recorte.
8. **Se conservan sin cambios** el argmax del Code Predictor, la penalización de repetición por aparición y la temperatura por defecto, 0,325: el usuario prefirió de oído la configuración actual frente a la de la implementación oficial, en 3 de 3 pares.

Decisiones de implementación derivadas: `SpeechDecoderWeights.codebooks` deja de ser `Arc`, porque solo se compartía con el codificador ICL; el autotest sustituye la comprobación del núcleo QKV `Q4_0` por la del QKV `INT8`, que pasa a usar el Talker; `--delete-voice` y `--list-voices` siguen aceptando archivos `QVCE` de cualquier versión, para ver y borrar las voces v5 que dejan de cargarse.

---

## Qué se conserva y por qué

- **El refuerzo del EOS**, por la investigación de los textos cortos (decisión 3): es la defensa ante una debilidad del propio modelo.
- **El recorte del silencio final** de la referencia (decisión 7).
- **El argmax del Code Predictor**, **la penalización por aparición** y **la temperatura 0,325** (decisión 8).

---

## Consecuencias

- Unos 185 MiB más de RAM según el cálculo; medido, unos 174 MiB (sección «Medición»).
- `.qvoice` v6: las voces anteriores se clonan de nuevo.
- No hay clonación con transcripción: desaparecen `--ref-text` y `--xvector-only`, y el codificador de voz.
- El tope de una síntesis que no termina es `min(DEFAULT_MAX_FRAMES, ROPE_MAX_POS − prefill_len)`, y el prompt solo se rechaza si `prefill_len + n_tokens_de_texto > ROPE_MAX_POS`.
- El audio de referencia de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) se regenera.

---

## Limitación conocida

En algunas frases la última sílaba sale con voz crepitante, de tono más bajo y sonido áspero.

- También ocurre en la implementación oficial (2 de 2 tomas) y en el motor anterior (4 de 6); en otras frases, 1 de 20 tomas.
- No lo corrigen ni un símbolo de relleno al final del texto ni subir la temperatura a 0,7 o 0,9: con 12 semillas por grupo, los rasgos de voz medidos con Praat (*jitter*, HNR y fracción sin voz) no difieren (Mann-Whitney bilateral).

---

## Rendimiento

Cifras de las decisiones:

| Cambio | Medida |
|---|---|
| Texto por pasos | Silencio interno más largo (mediana, 16 semillas): 0,92 s a 0,14 s; TTFA: ~698 ms a ~406 ms |
| Talker `INT8` frente a `Q4_0` y `BF16` | Diferencia de logits con el oficial: 2,75 (`Q4_0`), 1,04 (`INT8`), 0,06 (`BF16`); RTF de la frase larga: 0,62, 0,65 y 0,80 |
| Memoria del Talker | ~236 MiB (`Q4_0`) a ~422 MiB (`INT8`): ~185 MiB más |

### Medición

Ryzen 7 5825U, Windows, `-j 4 --seed 4 -l Spanish`, sin otras cargas, una síntesis a la vez y en orden intercalado. Se dan la mediana y el rango [mín, máx] de 3 síntesis por celda.

| Frase | Motor | RTF | TTFA (ms) | Carga del modelo (ms) | Pico de memoria (MiB) |
|---|---|---|---|---|---|
| Corta | anterior | 0,65 [0,62, 0,66] | 495 [475, 531] | 4581 [4428, 4619] | 2147 |
| Corta | este ADR | 0,71 [0,68, 0,74] | 385 [381, 395] | 2297 [2293, 2343] | 2323 |
| Larga | anterior | 0,65 [0,64, 0,65] | 657 [652, 663] | 4388 [4365, 4395] | 2199 |
| Larga | este ADR | 0,67 [0,67, 0,72] | 430 [420, 442] | 2302 [2273, 2350] | 2371 |

- **TTFA:** baja entre un 22 % y un 35 %, porque el prefill pasa a 10 posiciones.
- **Carga:** baja a la mitad. Ya no se carga el codificador de voz ni se cuantiza a `Q4_0` el Talker.
- **Memoria:** sube unos 174 MiB. Coincide con los ~185 MiB calculados para el Talker en `INT8`.
- **RTF:** el paso del Talker pasa de unos 13 a unos 16 ms por frame (`INT8` frente a `Q4_0`), y el RTF sube un poco en la frase larga.
  - En la frase corta el audio es otro (3,28 s frente a 4,64 s), así que su RTF no se compara directamente.
  - Con 6 repeticiones intercaladas por binario, el motor de este ADR y la configuración experimental equivalente dan el mismo tiempo en la frase larga: 10,1 s frente a 10,05 s de mediana, con rangos solapados.
