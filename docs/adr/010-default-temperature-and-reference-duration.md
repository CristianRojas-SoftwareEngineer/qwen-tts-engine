# ADR-010: Temperatura por defecto y duración recomendada de la referencia de clonación

Relacionado: [ADR-007](007-natural-voice-stepwise-text-int8-talker.md), que decide la clonación solo por x-vector, fijaba la temperatura 0,325 (decisión 8, que este ADR sustituye) y documenta la laringalización de la última sílaba como limitación del modelo.

> Registro de dos decisiones de calidad de la voz: la temperatura de muestreo por defecto pasa de 0,325 a 0,45, y la documentación recomienda una referencia de clonación de al menos 5 s de habla continua.

- **Estado**: Aceptado
- **Fecha**: 2026-11-27

---

## Contexto

### Voz crujiente al cierre de las frases

A la escucha de textos largos con la voz `default`, las frases que terminan con el tono descendente y poca energía cierran con voz crujiente (*creaky voice*, *vocal fry*, laringalización); las que terminan subiendo el tono cierran limpias. Es la limitación que la sección «Limitación conocida» de ADR-007 analiza en la última sílaba: la decide el modelo, no este motor, y ninguna regla de muestreo la evita del todo. Este ADR no la corrige; elige, entre temperaturas cercanas a la anterior, la que menos la produce.

- **Medición.** Se usó Creapy (TU Graz; clasificador de bosque aleatorio, modelo «all», bloques de 40 ms, salto de 10 ms y umbral de 0,75), sobre audio mono a 16 kHz. La métrica es el porcentaje del tiempo sonoro marcado como crujiente.
- **No viene de la referencia.** `Speech Sampler.wav`, de la que sale la voz `default`, tiene un 0,49 % de crujido (un solo tramo de 30 ms, en mitad de una frase y en ningún cierre). Treinta síntesis en español de 100 a 1000 caracteres, con temperatura 0,35, tienen un 2,12 % (de 0,69 a 5,45 % según el audio, ninguno sin crujido), y treinta en inglés un 1,63 %. El modelo añade el crujido al sintetizar.
- **Efecto de la temperatura.** Se sintetizaron 3 textos en español de 500 caracteres con la voz `default` y la semilla 4, a 9 temperaturas entre 0,300 y 0,500:

  | Temperatura | Crujido (3 textos) | Por texto |
  |---|---|---|
  | 0,300 | 2,47 % | 1,02 · 5,97 · 0,95 |
  | 0,325 | 2,88 % | 3,01 · 4,42 · 1,36 |
  | 0,350 | 2,38 % | 2,08 · 4,20 · 1,00 |
  | 0,375 | 1,99 % | 1,99 · 2,76 · 1,23 |
  | 0,400 | 1,81 % | 2,75 · 1,33 · 1,31 |
  | 0,425 | 2,48 % | 2,43 · 3,53 · 1,43 |
  | 0,450 | **1,51 %** | 1,21 · 1,59 · 1,75 |
  | 0,475 | 1,64 % | 0,94 · 2,07 · 1,94 |
  | 0,500 | 2,37 % | 1,09 · 4,10 · 2,05 |

  No hay una tendencia monótona: a una misma temperatura, la diferencia entre textos supera a la diferencia entre temperaturas, y ninguna baja al nivel de la referencia. 0,450 es el mínimo medido, la mitad que 0,325 y con la menor dispersión entre textos.
- **Escucha.** Comparados los tres textos a 0,325, 0,400 y 0,450, 0,450 suena más natural y con menos crujido.

### Duración de la referencia de clonación

El motor clona con cualquier duración (la referencia se usa entera), pero no decía cuánta hace falta.

- **Medición.** De una grabación continua del mismo locutor que la voz `default`, limpiada con DPDFNet y pasada por `preprocess_ref_audio`, se clonaron fragmentos de dos tramos distintos (A y B, separados 15 s) con 39 duraciones: de 0,1 a 1 s cada 0,1 s y de 1 a 30 s cada 1 s. Cada x-vector se comparó, por similitud coseno, con el de la referencia de 30 s de su tramo. La clonación es determinista: tres duraciones repetidas dieron `.qvoice` idénticos byte a byte.

  | Duración | Tramo A | Tramo B |
  |---|---|---|
  | 0,1 s | 0,8927 | 0,9135 |
  | 0,5 s | 0,9471 | 0,9664 |
  | 1 s | 0,9770 | 0,9868 |
  | 2 s | 0,9848 | 0,9909 |
  | 3 s | 0,9930 | 0,9949 |
  | 4 s | 0,9941 | 0,9965 |
  | 5 s | 0,9955 | 0,9974 |
  | 10 s | 0,9986 | 0,9990 |
  | 20 s | 0,9997 | 0,9997 |
  | 30 s | 1,0000 | 1,0000 |

- **Criterio, fijado antes de medir.** La duración mínima recomendada es la más corta a partir de la cual los dos tramos quedan, en todos los puntos siguientes, con una similitud de al menos 0,995. El tramo B cumple desde 4 s y el A desde 5 s.

---

## Decisiones

1. **`DEFAULT_TEMPERATURE = 0.45`** (`qwen-core/src/config.rs`), antes 0,325. Se aplica cuando la CLI, el HTTP o la API Rust no indican temperatura. Cambia el audio de toda petición sin temperatura explícita, incluidas las cuatro frases de la tabla «Audio de referencia (x86-64)» de [`NUMERIC_CONTRACT.md`](../NUMERIC_CONTRACT.md), así que sube la versión menor.
2. **Duración recomendada de la referencia: al menos 5 s de habla continua y limpia.** Es una recomendación, no un límite: el motor sigue aceptando cualquier duración. Por encima de 10 s la huella apenas cambia (similitud ≥ 0,9986 con la de 30 s). La recomiendan el README y el contrato de `--ref-audio`.

---

## Consecuencias

- **El crujido baja, pero no desaparece.** A 0,45 sigue en torno a tres veces el de la referencia, y la limitación de ADR-007 sigue vigente. Quitarlo del todo exige otras vías (modificar el x-vector, posprocesar el audio o reentrenar el modelo), fuera del alcance de esta decisión. ADR-007 ya comprobó que temperaturas mucho más altas (0,7 y 0,9) no lo corrigen.
- **Alcance de las mediciones.** Una sola voz y dos idiomas, con un detector entrenado con habla conversacional en alemán: los porcentajes absolutos son orientativos, pero la comparación entre temperaturas, con el mismo detector y el mismo procesado, es válida. No se midió la inteligibilidad a 0,45 por separado de la escucha.
- **La similitud coseno mide la estabilidad del timbre, no la calidad percibida.** Con referencias ruidosas o de otros locutores la curva puede desplazarse.
