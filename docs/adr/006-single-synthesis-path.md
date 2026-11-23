# ADR-006: Un único camino de síntesis

Relacionado: ADR 007 (`.qvoice` v6 y clonación solo por x-vector).

> Registro de decisión arquitectónica sobre cómo se sintetiza el audio: una sola entrada, dos entregas (completa y *streaming*), una regla de idioma y una composición multi-tramo idéntica en ambas.

- **Estado**: Aceptado
- **Fecha**: 2026-11-20

---

## Contexto

Una síntesis varía en dos ejes: qué se dice (texto plano o varios tramos con pausas y muletillas) y cómo se entrega (audio completo o *streaming*). Con un modelo y una sola forma de voz ([ADR-001](001-only-06b.md)), un camino propio por cada combinación tendría consecuencias concretas:

- Cada optimización del camino principal habría que replicarla y probarla en cada camino, con el paso de generación copiado varias veces.
- La misma petición podría dar PCM distinto según el modo: el orden de la ganancia, el fundido cruzado o la salida por `stdout` con marcado.
- El idioma se resolvería en varios sitios con reglas distintas, y viajaría fuera de la voz.
- La semilla, la temperatura y el marcado se prepararían varias veces.
- Habría varios destinos de audio con rampa de borde.

El caso de uso del motor es local: una síntesis a la vez en un equipo modesto, con la CPU como único recurso. Ni un procesamiento por lotes ni un servidor concurrente lo servirían.

---

## Decisión

1. **Una sola entrada de síntesis**: `Synthesizer { runtime, model, voice }.synthesize(scratch, &req, &abort, &script, delivery)`.
   - `Script` es `Plain(String)` o `Spans(Vec<SpanSpec>)`; `Delivery` es `Buffer` o `Stream(&mut dyn AudioSink)`.
   - La petición es un solo tipo, `SynthesisRequest`, con los valores por defecto en un solo sitio; la voz es un solo tipo, `Voice`, con el idioma resuelto al construirla.
2. **Una regla de idioma**: el del pedido (`-l` o `language`); si no hay, el de la voz; si tampoco, español (`2054`). La voz `default` declara español. El prompt siempre lleva idioma.
3. **Una composición multi-tramo idéntica en ambas entregas**: ganancia `x·(v·o)` por tramo, pausas, fundido cruzado de 45 ms junto a las muletillas. Va por el mismo camino en el WAV completo, en `--stdout` y en `/v1/tts/stream`. Una composición solo de pausas es un error (`Composición: nada que sintetizar`) también en la entrega completa.
4. **El servidor tiene un único trabajador** que atiende cada conexión en línea, con un solo enrutado y un solo calentamiento; no hay procesamiento por lotes ni servidor concurrente.
5. **Un solo destino de audio con rampas de borde** (`EdgeRampSink<W: PcmWrite>`), que no aplica ganancia; la ganancia la aplica `synthesize` justo antes de escribir. Los destinos `AudioSink` son `MemorySink` y `EdgeRampSink`; los escritores `PcmWrite` (`Vec<u8>`, `StdoutWriter` y `HttpChunkWriter`) solo escriben bytes.
6. **Preparación de la petición en un solo sitio** de `qwen-app` (`from_cli` y `from_http`), y una resolución única del modo de la CLI (`CliMode`).

---

## Alternativas descartadas

- **Mantener un camino de procesamiento por lotes**: dejaría un segundo camino de generación, con sus propios topes y proyecciones, que hay que optimizar y probar aparte, para un caso de uso (varios clientes simultáneos) que el motor no tiene. Lo mismo vale para un servidor concurrente.
- **Mantener los caminos paralelos y solo unificar la preparación**: elimina la copia de la configuración pero no la divergencia del PCM ni la duplicación del generador y de la composición.

---

## Consecuencias

### Positivas

- Toda optimización del camino principal se propaga a todos los casos de uso (CLI, *streaming*, servidor, composición) sin replicarla ni probarla por separado.
- El mismo guion da el mismo PCM en la entrega completa y en *streaming*; lo comprueba `qwen-engine/tests/synthesis_paths_tests.rs` en texto plano, con voz clonada por x-vector y con marcado.
- Menos código y menos superficie de API: sin tipos sustituidos ni alias, y sin `#[allow(clippy::too_many_arguments)]` en el camino de síntesis.

### Riesgos y mitigación

- Quien necesite atender varias peticiones a la vez debe ejecutar varios procesos.
- El audio de la entrega completa y el del *streaming* solo coinciden mientras compartan el generador; la prueba de equivalencia entre entregas lo vigila.
