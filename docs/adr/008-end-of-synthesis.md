# ADR-008: Final de la síntesis: retención tras el primer EOS y punto final del texto

Relacionado: [ADR-007](007-natural-voice-stepwise-text-int8-talker.md), que conserva el refuerzo del EOS; este ADR decide cómo termina la síntesis cuando el Talker elige el EOS.

> Registro de decisión sobre el final del audio: el Talker genera 2 frames más tras el primer `CODEC_EOS`, con el EOS bloqueado, y el texto que se sintetiza termina siempre en puntuación final.

- **Estado**: Aceptado
- **Fecha**: 2026-11-24

---

## Contexto

La última sílaba de una síntesis puede oírse cortada: «Hola mundo» se oye «Hola mund», con la voz `default` y con una voz clonada. El Talker elige `CODEC_EOS` en un frame en que la voz todavía suena, y el audio termina sin la caída final del sonido; la rampa de salida de 40 ms apaga lo que quedaba.

- **Es un defecto del modelo.** Está reportado en el repositorio oficial de Qwen3-TTS para varios idiomas, sin solución oficial. La implementación oficial no tiene refuerzo del EOS: corta en el primer EOS, con `min_new_tokens=2`.
- **El refuerzo lo agrava.** En «Hola mundo» (voz `default`, temperatura 0,35, semilla 4), la probabilidad de `CODEC_EOS` en el frame 12 es 7·10⁻⁵ sin refuerzo y 0,28 con él, y esa semilla elige terminar ahí. Sin refuerzo, el modelo genera el frame 12 y termina en el 13, con la voz aún sonando.
- **Pasa también con punto.** Tres de los cuatro textos del audio de referencia de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md), que terminan en punto, acaban con energía de voz en su último frame: de −42 a −33 dB antes de la rampa de salida. El cuarto termina en silencio.

---

## Decisiones

1. **Retención de 2 frames tras el primer EOS.** Cada frame es un paso del Talker y dura 80 ms (1 920 muestras a 24 kHz). La constante `EOS_HOLD_FRAMES = 2` (`pipeline.rs`) fija la regla:
   - en el primer frame en que el muestreo devuelve `CODEC_EOS`, ese EOS se descarta y no entra en `prev_tokens`;
   - se recalculan los logits de ese mismo frame, con las mismas reglas, y se bloquea `CODEC_EOS` (logit `-1e30`);
   - se vuelve a muestrear con el mismo generador de números aleatorios, sin reiniciarlo: el código resultante es el primer frame retenido;
   - el EOS sigue bloqueado en el segundo frame retenido, y tras entregarlo la síntesis termina sin dar otro paso del Talker ni comprobar otra vez la cancelación;
   - la retención nunca supera `max_frames`: si el primer EOS cae en el último frame permitido, se retiene un solo frame. El número máximo de frames de una síntesis no cambia.

   Para decidir cuándo termina la síntesis no se añade ninguna lista de códigos de silencio ni ninguna puerta que dependa de ella. `SynthesisTiming::first_eos_frame` guarda el frame del primer EOS (`None` si la síntesis llegó al tope sin EOS).
   - **La retención completa el sonido.** El modelo necesita de 0 a 2 frames tras el primer EOS para llegar a silencio. Con 1 frame, «Hola mundo» con la voz clonada termina aún a −41 dB; con 2, a −65 dB. En la escucha del usuario, con 2 frames los dos «Hola mundo» (`default` y clonada) se oyen completos y con 1 frame el clonado sigue cortado. Con el punto final, el clonado «Hola mundo.» termina a −45 dB tras los 2 frames y el usuario lo oye completo.
   - **No hace hablar al modelo.** Donde ya había silencio («Error del servidor», «Hmmm...»), los frames retenidos siguen en silencio (−99 a −180 dB). «Hmmm...» conserva el frame de su primer EOS.
   - **Solo cambia el final.** Con la retención, el PCM de los cuatro textos de referencia es idéntico bit a bit salvo en los últimos 40 ms del audio original, donde caía la rampa de salida.
   - **Sin lista de silencio fiable para la parada.** Los códigos `code0` que el codificador del propio checkpoint produce para silencio y ruido de hasta −50 dBFS son 22, y ninguno aparece en frames con voz, pero cubren solo el 44,8 % de los frames silenciosos que genera el Talker. Una lista aprendida de síntesis queda incompleta por construcción: 13 de 49 códigos aparecieron una sola vez. Por eso la parada no depende de clasificar el silencio.
2. **Punto final en el texto completo.** `plan_script` cierra el texto que se sintetiza, con y sin `--no-compose`: al texto plano o, si hay marcado, al texto del último tramo hablado, ignorando las pausas que vengan detrás. Los demás tramos no cambian. Quien construye un `Script` directamente (la API Rust, el calentamiento del servidor, las pruebas) no recibe el punto. La regla (`close_final_sentence`, `markup.rs`):
   - espacio es todo carácter con `char::is_whitespace`; los signos de cierre son `"`, `'`, `»`, `”`, `’` y `)`; se busca el último carácter que no es espacio ni signo de cierre;
   - si no existe (texto vacío, solo espacios o solo signos de cierre), el texto no cambia;
   - si es `.`, `!`, `?`, `…`, `。`, `！` o `？`, el texto no cambia, con sus espacios finales incluidos;
   - si es `,`, `;` o `:`, ese carácter se sustituye por `.` y el resto queda igual;
   - en cualquier otro caso se inserta `.` justo después del último carácter que no es espacio (detrás de los signos de cierre) y los espacios finales se conservan detrás.

   - **Evidencia:** en la escucha del usuario, 3 de 3 anuncios sin punto («Tarea completada», «Subagente terminado», «Error del servidor») cierran con una entonación más natural cuando terminan en punto.
3. **Se conservan el refuerzo del EOS y la rampa de salida.** El refuerzo sigue siendo la cota de los textos muy cortos («Hmmm...»), y con la retención adelantar el EOS ya no corta la voz. La rampa de 40 ms actúa sobre el final retenido.

Los cuatro modos de entrega (CLI con `-o`, CLI con `--stdout`, `/v1/tts` y `/v1/tts/stream`) dan el mismo PCM.

---

## Alternativas descartadas

- **Lista de códigos de silencio para decidir la parada**, para terminar solo cuando el frame es silencio: queda incompleta (44,8 % de cobertura con los códigos del códec; 13 de 49 códigos aprendidos aparecieron una sola vez).
- **Puerta que solo acepta el EOS sobre silencio:** depende de esa lista.
- **Añadir silencio tras el EOS:** inútil, porque el decodificador es causal y el sonido que falta lo tiene que generar el Talker; el final no se puede añadir después.
- **1 frame de retención:** deja cortado el «Hola mundo» clonado (−41 dB al terminar).
- **Quitar o recalibrar el refuerzo del EOS:** deja sin cota los textos muy cortos, que se alargan y repiten ([ADR-007](007-natural-voice-stepwise-text-int8-talker.md), decisión 3).

---

## Qué se conserva y por qué

- **El refuerzo del EOS**, por los textos muy cortos (ADR-007).
- **La rampa de salida de 40 ms**, que ahora actúa sobre el final retenido.

---

## Consecuencias

- El audio cambia solo al final: los últimos 40 ms del audio original, más hasta 160 ms nuevos.
- El peor caso son 2 frames más por síntesis.
- Cada tramo del marcado crece hasta 160 ms. En «Tarea completada [pause:0.5] Revisa los resultados [hmm] y avísame si falta algo.», los silencios entre tramos pasan de 1 570, 1 050 y 950 ms a 1 730, 1 210 y 1 120 ms. El usuario oyó bien el ritmo. El principio de cada tramo lo acorta el descarte de la cadena de silencio inicial ([ADR-009](009-leading-silence.md)).
- El audio de referencia de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) se regenera: cada texto dura 160 ms más.
- Un texto sin puntuación final se sintetiza con un punto añadido; «Tarea completada» y «Tarea completada.» dan el mismo audio.

---

## Limitaciones

- **N = 2 sale de cuatro casos con una semilla**, no de un barrido.
- **El punto final solo se escuchó en español.** La regla se aplica igual a los demás idiomas.
- **El punto añade un token al texto.** Un texto que cabe justo en `10 + N <= ROPE_MAX_POS` puede pasar a rechazarse.
- **La voz crepitante de la última sílaba no depende del punto final.** Es la limitación conocida del [ADR-007](007-natural-voice-stepwise-text-int8-talker.md), un comportamiento del modelo sin corrección en inferencia. Medido con Praat en los últimos 300 ms con voz (*jitter* local, HNR y fracción de tramos sin voz, el mismo método del ADR-007, que separa los finales que el usuario oyó crepitantes de los que oyó bien): 3 frases cortas en español («Tarea completada», «Hola mundo», «Error del servidor») × 12 semillas, con y sin punto, voz `default`, T 0,35 y la retención de 2 frames.
  - Con punto la última sílaba no sale peor: Wilcoxon emparejado p = 0,12 (*jitter*), 0,25 (HNR) y 0,64 (fracción sin voz) en los 36 pares, y p = 0,08, 0,24 y 1,0 en los 29 pares sin audios repetidos (con T 0,35 algunas semillas dan el mismo audio). La mediana del *jitter* es algo menor con punto (−0,002). Hay finales crepitantes en los dos lados.
  - La síntesis es determinista para una frase y una semilla, así que una frase concreta puede salir siempre con la última sílaba crepitante. Es el caso de «Tarea completada.» con la semilla 4: *jitter* 0,027, HNR 4,2 dB y 27 % de tramos sin voz, frente a 0,011, 9,4 dB y 0 % sin punto. Cambiar la puntuación solo cambia qué frases caen en ese caso.
