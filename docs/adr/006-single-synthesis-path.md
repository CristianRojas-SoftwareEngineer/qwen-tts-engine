# ADR-006: Un único camino de síntesis

> Registro de decisión arquitectónica sobre cómo se sintetiza el audio: una sola entrada, dos entregas (completa y *streaming*), una regla de idioma y una composición multi-tramo idéntica en ambas.

- **Estado**: Aceptado
- **Fecha**: 2026-11-20

---

## Contexto

Tras pasar a un solo modelo y a una sola forma de voz ([ADR-001](001-only-06b.md)), el motor conservaba varios caminos de síntesis que repetían la preparación de la petición y divergían entre sí:

- la entrega completa, el *streaming* y la composición multi-tramo (con una versión en buffer y otra en *streaming*);
- el procesamiento por lotes, con su propio bucle de generación, sus propios topes y sus propios *kernels* de proyección;
- el servidor con varios trabajadores y con lotes continuos, con su propio enrutado HTTP;
- el calentamiento, que construía una petición como si llegara por HTTP.

Las consecuencias eran concretas:

- Cada optimización del camino principal había que replicarla y probarla en cada camino; el paso de generación estaba copiado tres veces.
- La misma petición daba PCM distinto según el modo: la ganancia se aplicaba en otro orden en *streaming*, no había fundido cruzado, la CLI con `--stream` y marcado iba por la versión en buffer y `--stdout` con marcado no sacaba nada por `stdout`.
- El idioma se resolvía en cuatro sitios con reglas distintas (la CLI usaba inglés por defecto y el servidor, ninguno), y viajaba fuera de la voz.
- La semilla, la temperatura y el marcado se preparaban por triplicado, con pares `Option` más un indicador «fue dado».
- Había cuatro destinos de audio con rampa de borde y tipos heredados que ya no significaban nada (`VoiceBindings`, `VoiceIdentity`, `RequestConfig`).

El caso de uso del motor es local: una síntesis a la vez en un equipo modesto, con la CPU como único recurso. Ni el lote ni el servidor concurrente lo servían.

---

## Decisión

1. **Una sola entrada de síntesis**: `Synthesizer { runtime, model, voice }.synthesize(scratch, &req, &abort, &script, delivery)`.
   - `Script` es `Plain(String)` o `Spans(Vec<SpanSpec>)`; `Delivery` es `Buffer` o `Stream(&mut dyn AudioSink)`.
   - La petición es un solo tipo, `SynthesisRequest`, con los valores por defecto en un solo sitio; la voz es un solo tipo, `Voice`, con el idioma y el tope de ICL resueltos al construirla.
2. **Una regla de idioma**: el del pedido (`-l` o `language`); si no hay, el de la voz; si tampoco, español (`2054`). La voz `default` declara español. El prompt siempre lleva idioma.
3. **Una composición multi-tramo idéntica en ambas entregas**: ganancia `x·(v·o)` por tramo, pausas, fundido cruzado de 45 ms junto a las muletillas y acabado (`tail_trim`, `onset_fade`) solo en la entrega completa. El marcado, con `--compose` o sin él, aplica la sustitución paralingüística y va por el mismo camino con `--stream` y `--stdout`.
4. **Se retiran el procesamiento por lotes y el servidor concurrente**: `batch.rs`, `--batch*`, `--workers`, `--batch-size` y sus planificadores. El servidor queda con un único trabajador que atiende cada conexión en línea, con un solo enrutado y un solo calentamiento.
5. **Un solo destino de audio con rampas de borde** (`EdgeRampSink<W: PcmWrite>`), que no aplica ganancia; la ganancia la aplica `synthesize` justo antes de escribir. Los escritores (WAV progresivo, `stdout`, memoria, HTTP *chunked*) solo escriben bytes.
6. **Preparación de la petición en un solo sitio** de `qwen-app` (`from_cli` y `from_http`), y una resolución única del modo de la CLI (`CliMode`).

---

## Alternativas descartadas

- **Corregir y mantener el lote**: arreglar la doble ganancia, la decodificación de una vez, los topes propios y las proyecciones no bit-exactas dejaba un segundo camino que hay que optimizar y probar aparte, para un caso de uso (varios clientes simultáneos) que el motor no tiene. El servidor concurrente se retiró por la misma razón.
- **Mantener los caminos paralelos y solo unificar la preparación**: elimina la triple copia de la configuración pero no la divergencia del PCM ni la duplicación del generador y de la composición.
- **Conservar `--stream` con marcado por la versión en buffer**: mantiene el comportamiento anterior, pero contradice el contrato de que los dos modos dan el mismo PCM.

---

## Consecuencias

### Positivas

- Toda optimización del camino principal se propaga a todos los casos de uso (CLI, *streaming*, servidor, composición) sin replicarla ni probarla por separado.
- El mismo guion da el mismo PCM en la entrega completa y en *streaming*; lo comprueba `qwen-engine/tests/synthesis_paths_tests.rs` en texto plano, con ICL y con marcado.
- Menos código y menos superficie de API: sin tipos sustituidos ni alias, y sin `#[allow(clippy::too_many_arguments)]` en el camino de síntesis.

### Cambios observables

- **Idioma por defecto español en todos los modos**: sin `-l` ni `language`, la voz `default` habla español; antes la CLI usaba inglés y el servidor no enviaba idioma. `GET /v1/voices` devuelve `"language":"Spanish"` para la voz `default`, y el log `lang=` de las peticiones muestra el idioma efectivo.
- **`--stream` y `--stdout` con marcado** dan el mismo PCM que el WAV; `--stdout` con marcado sale por `stdout`.
- **`--compose` en buffer es igual a `--text` con el mismo marcado**: aplica la paralingüística (`[laugh]` y `[sigh]` son onomatopeyas como en el texto normal).
- **Una composición solo de pausas es un error** (`nothing to synthesize`) también en la entrega completa; antes daba silencio.
- **`-S` gana a `-D`**: con los dos, no se imprimen diagnósticos.
- Los flags `--batch`, `--batch-size`, `--workers` y los demás del lote se rechazan con el error genérico del *parser*.
- La API Rust de `qwen-engine` cambia sin compatibilidad hacia atrás: `Voice`, `SynthesisRequest`, `Synthesizer`, `Delivery`, `Script` y `EdgeRampSink` sustituyen a `VoiceBindings`, `RequestConfig`, `synthesize_to_samples`, `synthesize_to_sink` y los destinos anteriores ([`INTERFACE_CONTRACT.md`](../INTERFACE_CONTRACT.md) §3).

### Riesgos y mitigación

- Quien necesite atender varias peticiones a la vez debe ejecutar varios procesos o retomar el servidor concurrente desde el historial de git (véase [`ROADMAP.md`](../../ROADMAP.md)).
- El audio de la entrega completa y el del *streaming* solo coinciden mientras compartan el generador; la prueba de equivalencia entre entregas lo vigila.
