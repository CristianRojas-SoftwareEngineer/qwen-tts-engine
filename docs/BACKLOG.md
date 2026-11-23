# Backlog

Funcionalidades pedidas o diseñadas que se dejaron para más adelante, con lo necesario para retomarlas sin conocer la conversación en que surgieron.

## Convenciones

- **Una entrada por funcionalidad**, con estas secciones: qué es, por qué no se hace ahora, cuándo merece la pena retomarla, lo que ya está decidido, lo que falta por analizar y el trabajo que supone.
- **Autocontenida y en términos generales:** describe la capacidad del motor para cualquier cliente, sin nombrar a un integrador concreto.
- **Código citado por archivo y símbolo**, no por número de línea.
- **Lo decidido, separado de lo pendiente.** Lo que no se ha analizado no se presenta como resuelto.
- **Al empezar una entrada**, pasa a un plan de trabajo y se borra de aquí; al descartarla, se borra y el motivo queda en el commit.

## Selección de voz por petición en el servidor HTTP

### Qué es

Hoy el servidor (`--serve`) carga una sola voz al arrancar: la de `--load-voice <archivo>` o, sin esa opción, la voz `default` incorporada. El campo `voice` de una petición solo puede repetir el nombre de esa voz; si no coincide, la respuesta es `400` (`ValidatedTtsRequest::parse` en `crates/qwen-app/src/server/validation.rs`). Un cliente que necesita otra voz tiene que reiniciar el servidor.

La funcionalidad consiste en que el campo `voice` elija la voz de cada petición, de modo que un solo servidor atienda varias voces sin reiniciarse.

### Por qué no se hace ahora

- **Ningún cliente necesita hoy alternar voces a menudo.** El uso previsto es una sola voz activa, que cambia de forma ocasional.
- **El beneficio de cada cambio es pequeño:** evita reiniciar el servidor, es decir, la carga del modelo (624 ms en el equipo de referencia, Ryzen 7 5825U) y el calentamiento previo a aceptar conexiones (~1 s); unos 1,5 s en total, según esa suma.
- **El coste no lo es:** una opción nueva, un cambio en la validación de la petición y varios casos límite sin analizar (abajo).

### Cuándo retomarla

Cuando un cliente necesite alternar voces entre peticiones seguidas, por ejemplo una voz distinta por cada interlocutor, y el reinicio por cambio de voz se note en cada síntesis.

### Lo que ya está decidido

- Nueva opción `--voices-dir <carpeta>`, solo válida con `--serve` (si no, el proceso sale con `1` antes de cargar el modelo); la carpeta debe existir al arrancar.
- El campo `voice` es un **nombre** que resuelve a `<carpeta>/<nombre>.qvoice`, también para `default`. El archivo se lee en cada petición (un `.qvoice` v6 ocupa 4112 bytes), después de validar la petición y antes de enviar la cabecera, para que una voz recién creada sirva sin reiniciar.
- Nombres admitidos: letras ASCII, dígitos, `.`, `_` y `-`, distintos de `.` y `..`, tal cual y sin pasar a minúsculas. Así no hay forma de leer fuera de la carpeta.
- Con `voice` y sin `--voices-dir`, `400`. Un nombre con otra forma, un archivo inexistente o ilegible y una versión de `.qvoice` no válida también son `400`, con la clave `error`: son errores de la petición.
- Sin `voice`, se usa la voz de arranque, y el audio de esas peticiones no cambia.
- El idioma sigue la regla única del motor: el pedido; si no hay, el de la voz; y si tampoco, español.
- Descartado que `voice` lleve la ruta absoluta del archivo (expone rutas del cliente) y un registro de voces fijado al arrancar (añadir una voz exigiría reiniciar).

### Lo que falta por analizar

- **Mayúsculas en Windows:** el sistema de archivos no las distingue, así que `Default` y `default` abrirían el mismo archivo, y en Linux no.
- **Nombres reservados de Windows** (`CON`, `NUL`, `AUX`, `COM1`…), que abren dispositivos.
- **Enlaces simbólicos** dentro de la carpeta que apunten fuera de ella.
- **Archivo a medio escribir** si el cliente reemplaza una voz mientras el servidor la lee, y si el archivo abierto bloquea ese reemplazo en Windows.
- **`GET /v1/voices`:** hoy devuelve la voz cargada; con una carpeta de voces, devolver solo la de arranque sería incoherente.
- **Calentamiento:** hoy se hace con la voz de arranque; comprobar que basta para la primera petición con otra voz.

### Trabajo que supone

- `crates/qwen-app/src/cli.rs`: la opción, su regla y la ayuda.
- `crates/qwen-app/src/server/validation.rs`: `ValidatedTtsRequest::parse` deja de comparar `voice` con la voz cargada (pierde su parámetro `loaded_voice_name`) y valida la forma del nombre.
- `crates/qwen-app/src/server/mod.rs` y `http.rs`: `ServerContext` guarda además la carpeta de voces, y `prepare` construye el `Synthesizer` con la voz de la petición o con la de arranque.
- Pruebas de integración con el binario: dos peticiones con voces distintas sin reiniciar dan cada una el mismo PCM que la CLI con `--load-voice` de ese archivo; una petición sin `voice` vuelve a la voz de arranque; un archivo reemplazado entre dos peticiones se usa en la segunda; `voice` sin `--voices-dir` y `voice: "../x"` dan `400`.
- Documentación: `docs/INTERFACE_CONTRACT.md` (opción, campo `voice` y validación), `docs/ARCHITECTURE.md` (servidor) y el README (sección del servidor y su ejemplo de `curl`, que envía `"voice":"default"`).
- Versión: compatible para los clientes que no envían `voice`, así que es una versión menor. Los que envían `voice` con el nombre de la voz cargada y arrancan el servidor sin `--voices-dir` recibirían `400`; antes de publicarla hay que decidir si eso se mantiene (versión mayor) o si, sin `--voices-dir`, se sigue aceptando el nombre de la voz de arranque.
