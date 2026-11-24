# ADR-009: Inicio de la voz: descarte de la cadena de silencio inicial

Relacionado: [ADR-008](008-end-of-synthesis.md), que decide cómo termina la síntesis; este ADR decide cómo empieza el audio que se entrega.

> Registro de decisión sobre el principio del audio: el Talker y el decodificador procesan todos los frames, y el motor no entrega el audio de los frames iniciales cuyo `code0` pertenece a la cadena de silencio inicial.

- **Estado**: Aceptado
- **Fecha**: 2026-11-24

---

## Contexto

Al principio de cada síntesis, y de cada tramo del marcado, la voz tarda en empezar: el audio abre con 400 ms o más de silencio digital.

- **Es un comportamiento del modelo.** Todas las voces del motor son de tipo x-vector, también la `default`: el prompt lleva el idioma, el x-vector del locutor y `CODEC_PAD`, sin códigos de voz de la referencia (`build_prompt`, `prompt.rs`). En ese modo el Talker arranca en frío y empieza emitiendo códigos de silencio. Está documentado en otros motores de Qwen3-TTS (sglang-omni, PR #2357; vllm-omni, issue #4966 y PR #5048) y ocurre también en la implementación oficial.
- **La cadena es fija.** En 72 de 72 síntesis en español (voces `default` y dos clonadas, las frases «Tarea completada», «Hola mundo», «Error del servidor» y «Subagente terminado», semillas 4, 5 y 6, temperatura 0,35), los cinco primeros `code0` son `1995 215 212 1181 462`, a veces con `462` repetido: de 5 a 7 frames. Esos ids son los que vllm-omni obtiene al codificar silencio digital con el mismo códec. El decodificador los convierte en silencio (como mucho −96 dB en lo medido).
- **No siempre aparece.** En inglés (voz `default`, 6 frases, semillas 4, 5 y 6) la cadena aparece en 14 de 18 síntesis; las 4 restantes, frases que empiezan por «Hello», empiezan con voz desde el frame 0 (`code0` 9). El texto corto en inglés del audio de referencia de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) es uno de esos casos.
- **Tras la cadena puede seguir más silencio** con otros códigos, según la frase y la semilla. Con la cadena incluida, el silencio inicial tiene una mediana de 480 ms y llega a 2,9 s en lo medido.
- **El marcado lo multiplica.** Cada tramo es una síntesis con su propio prefill y su propia cadena, así que las pausas suenan más largas de lo pedido: en «Tarea completada [pause:0.5] Revisa los resultados [hmm] y avísame si falta algo.», con la voz clonada del usuario, el silencio tras «Tarea completada» dura 1 940 ms.

---

## Decisiones

1. **Se descarta el audio de la cadena; no se prohíbe al Talker.** El Talker y el decodificador generan y decodifican todos los frames, y el calendario de chunks no cambia. El motor no entrega las muestras de los frames iniciales cuyo `code0` pertenece a `{1995, 215, 212, 1181, 462}`, desde el primero y mientras sigan perteneciendo. El primer frame que no pertenece termina el descarte, aunque después vuelva a salir uno de esos códigos. Cada tramo del marcado, al ser una síntesis, descarta su propia cadena. El descarte ocurre en el hilo del decodificador, antes de la ganancia. `SynthesisTiming::leading_silence_frames` guarda cuántos frames se descartaron.
   - **Descartar no cambia nada más.** En texto plano con `rate = 1`, el resto del audio es idéntico bit a bit al que se obtiene entregando la cadena, después de la rampa de entrada (comprobado en 4 de 4 frases con la voz del usuario), sin chasquidos. Lo descartado no pasa de −96 dB.
   - **Efecto en el inicio.** En las 36 síntesis en español de la línea base, el silencio inicial baja de 480 a 40 ms de mediana, y 27 de 36 empiezan con voz en 80 ms o menos. En inglés, 18 de 18 empiezan con voz en 80 ms o menos.
   - **Escucha del usuario.** Con su voz clonada: los cuatro anuncios y el texto con marcado se oyen bien, con la primera sílaba completa y sin chasquidos.
2. **La cadena es una constante del checkpoint.** `LEADING_SILENCE_CODES: [i32; 5] = [1995, 215, 212, 1181, 462]` (`pipeline.rs`), junto a `EOS_HOLD_FRAMES`: son los ids que el códec de este checkpoint asigna al silencio digital. El motor no tiene el codificador del códec para derivarlos al arrancar. La prueba `test_leading_silence_discard` (`engine_tests.rs`), con el modelo real, falla si con la voz `default` en español la cadena deja de aparecer al principio, y comprueba que lo descartado es exactamente la cadena inicial.
   - **Por qué aquí sí sirve una lista de códigos.** El [ADR-008](008-end-of-synthesis.md) descartó una lista de silencio para decidir cuándo termina la síntesis, porque una lista incompleta cambia el audio. Aquí la lista solo reconoce un prefijo fijo que el modelo emite al arrancar, y si un código de silencio no está en ella, el motor entrega audio de más (silencio), nunca de menos.
3. **Si la síntesis termina sin salir de la cadena, su audio se entrega entero.** Un tramo vacío sería un error de la composición (`Composición: falló la síntesis del tramo <i>`), así que esa síntesis entrega su audio como si no hubiera cadena. En ese caso `leading_silence_frames` vale 0. Si la síntesis se cancela o falla, no se entrega el audio apartado.
4. **El silencio que sigue a la cadena con otros códigos queda fuera.** Recortarlo exigiría una lista de códigos que queda incompleta o un umbral de energía que puede comerse consonantes suaves.
5. **El TTFA mantiene su definición:** el primer chunk no vacío entregado. No incluye la cadena, así que su valor sube; no se añade ninguna métrica nueva.
6. **El log.** Sin `-S`, `  Frames de silencio inicial descartados: <K>` sale antes de `  Generados <N> frames (<S>s de audio)`. En esta última, `N` son los frames generados y `S` los segundos de audio que entrega el decodificador, antes de `--rate`.

Los cuatro modos de entrega (CLI con `-o`, CLI con `--stdout`, `/v1/tts` y `/v1/tts/stream`) dan el mismo PCM.

---

## Alternativas descartadas

- **Prohibir la cadena al Talker:** prohibir esos cinco ids en los frames 0 y 1 adelanta la voz (mediana de 80 ms), pero obliga al Talker a un arranque que no elegiría: el usuario oyó un chasquido, como una tecla de teclado mecánico, en «Error del servidor» y «Subagente terminado». En la forma de onda es una ráfaga aislada al empezar (de 0 a 15 ms, o a los 80 ms).
- **Dar al decodificador frames de silencio antes de empezar:** atenúa el chasquido en un caso (de −29 a −44 dB) y no cambia el otro, porque lo genera el Talker.
- **Recortar por energía:** un umbral puede comerse consonantes suaves.
- **Una lista de silencio más larga:** prohibir al Talker 22 ids de silencio (los de la cadena más otros observados en silencios iniciales y los que vllm-omni obtiene del códec) da el mismo audio, byte a byte, que prohibir los cinco de la cadena: 36 de 36 síntesis.
- **Recortar el silencio inicial de la referencia al clonar:** la cadena aparece también con la voz `default` y con una voz clonada de una grabación cuya voz empieza en el primer frame, sin silencio inicial.

---

## Consecuencias

- En texto plano con `rate = 1`, el audio cambia solo al principio: es el que genera el modelo menos los frames de la cadena.
- Con `--rate` distinto de 1 y con marcado, el audio cambia también después del prefijo: el estiramiento WSOLA ancla su rejilla al inicio del audio, y el fundido de 45 ms entre tramos mezcla la cola del tramo anterior con la voz del siguiente. Los frames que genera el Talker son los mismos.
- Cada tramo del marcado descarta su cadena y las pausas se acercan a lo pedido. En el texto con marcado del Contexto, con la voz clonada del usuario, los silencios entre tramos pasan de 1 940, 1 120 y 760 ms a 1 380, 480 y 360 ms. Las cifras de silencios entre tramos del ADR-008 (1 730, 1 210 y 1 120 ms) son de la voz `default`.
- En streaming la voz se oye antes (TTFA más el silencio que queda delante de la voz), en 3 de 3 repeticiones con la voz `default`, `-j 4`, semilla 4, temperatura 0,35 y `-l Spanish`: de 806 a 569 ms en «Hola mundo, esta es una prueba rápida de referencia.» y de 794 a 613 ms en «Tarea completada.». El TTFA sube (de unos 400 a unos 570–610 ms), porque el primer chunk, que era silencio, no se entrega.
- El RTF sube: el tiempo de proceso es el mismo y el audio entregado tiene `K × 80 ms` menos.
- El audio de referencia de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md) se regenera: la frase corta en inglés no tiene cadena y no cambia; las otras tres duran 400 ms menos.

---

## Limitaciones

- **El silencio que sigue a la cadena con otros códigos se conserva** (decisión 4).
- **La cadena no aparece en todas las síntesis.** En inglés, 14 de 18; las frases que empiezan por «Hello» empiezan con voz.
- **La constante es de este checkpoint.** Con otro checkpoint o con otra ruta de cálculo los ids pueden ser otros; la prueba con el modelo avisa si la cadena deja de aparecer.
- **La evidencia es de una CPU x86-64 con la ruta `avx2`** (Ryzen 7 5825U, Windows).
