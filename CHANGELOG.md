# Registro de cambios

Cambios de cada versión de **Qwen TTS Engine** (`qwen-tts-engine`) que afectan a quien lo usa. Las versiones siguen el versionado semántico sobre el contrato de uso ([`docs/INTERFACE_CONTRACT.md`](docs/INTERFACE_CONTRACT.md), sección «Versionado»): un cambio del audio para la misma petición sube la versión menor.

## [No publicado]

### Añadido

- **Prueba del audio de referencia:** con `QWEN_TTS_CHECK_REFERENCE_AUDIO=1`, `cargo test` sintetiza las cuatro frases de la tabla «Audio de referencia (x86-64)» de [`docs/NUMERIC_CONTRACT.md`](docs/NUMERIC_CONTRACT.md) y falla si algún hash cambia. Los hashes son del equipo de referencia: otra CPU x86-64 da otros, así que en la CI la prueba se omite.
- **Pruebas del cambio de velocidad (`--rate`):** la duración, el tono, los límites del factor y los casos sin estiramiento de `time_stretch`.
- **Este registro de cambios.**

### Corregido

- **README:** la fila de Windows de «Dependencias por plataforma» decía que no hacía falta MSYS2 ni el OpenBLAS propio; solo Visual Studio sobra.
- **Cifras de rendimiento** (README, [ADR-005](docs/adr/005-cpu-inference-optimizations.md) y `ARCHITECTURE.md`), medidas de nuevo con la v1.2.1:
  - **Equipo de referencia:** sin carga se mantiene (RTF 0,67 en la frase larga en español). Con una carga de fondo moderada da 0,99–1,02, no 0,93–0,95, así que con carga no hay margen de tiempo real.
  - **Máquina Windows de 4 núcleos de la CI:** da 1,16–1,28 y ya no llega a tiempo real. La v1.0.0 da hoy 1,47–1,63 en ella, así que cambió la máquina, no el motor.
  - **macOS:** sin cambios (0,23–0,25).
  - **Linux ARM64:** 1,43–1,48.

## [1.2.1] — 2026-11-25

Sin cambios en el binario ni en el audio.

- **Pruebas:** los temporales de las pruebas van a `target/` y cada prueba borra los suyos al terminar, también si falla.
- **Documentación:** la voz crepitante ocasional de la última sílaba queda descrita como limitación del modelo ([ADR-007](docs/adr/007-natural-voice-stepwise-text-int8-talker.md)).

## [1.2.0] — 2026-11-24

### Cambiado

- **El audio empieza con voz:** el motor ya no entrega el silencio con que el modelo arranca cada síntesis y cada tramo del marcado (los frames iniciales seguidos con los códigos de silencio de este checkpoint). En 36 síntesis en español, el silencio inicial baja de 480 a 40 ms de mediana. Detalle en [ADR-009](docs/adr/009-leading-silence.md).
  - El primer audio entregado (TTFA) llega más tarde, de unos 400 a unos 570–610 ms, porque el primer chunk ya no es silencio; la voz se oye antes.
  - El audio entregado dura menos para el mismo trabajo, así que el RTF calculado sobre él sube.
  - Las pausas del marcado se acercan a lo pedido, porque cada tramo ya no suma su silencio inicial.
- **Cambia el audio de referencia:** tres de las cuatro frases duran 400 ms menos; la corta en inglés no tiene ese silencio y no cambia.

## [1.1.0] — 2026-11-24

### Cambiado

- **La última sílaba ya no se corta** («Hola mundo» se oía «Hola mund»): tras el primer fin de síntesis que elige el modelo, el motor genera 2 frames más. Detalle en [ADR-008](docs/adr/008-end-of-synthesis.md).
- **El texto termina con punto final:** si el texto, o el último tramo hablado del marcado, no termina en `.`, `!`, `?`, `…` o `...`, el motor añade un punto; una `,`, `;` o `:` final se sustituye por punto. El punto añade un token, así que un texto que llegaba justo al límite de posiciones puede pasar a rechazarse.
- **Cambia el audio de referencia** de las cuatro frases.

## [1.0.0] — 2026-11-23

Primera versión publicada: CLI `qwen_tts`, servidor HTTP (`--serve`) y clonación de voz a `.qvoice` v6 para Qwen3-TTS 0.6B Base, con binarios para Windows x86_64, Linux x86_64, Linux ARM64 y macOS ARM64.

[No publicado]: https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/compare/v1.2.1...HEAD
[1.2.1]: https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/CristianRojas-SoftwareEngineer/qwen-tts-engine/releases/tag/v1.0.0
