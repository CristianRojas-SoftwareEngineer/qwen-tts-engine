// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Equivalencia entre entregas (`Buffer` frente a `Stream`), construcción de voces y bytes del
//! perfil `.qvoice` que produce `clone_voice`.

use std::fs;
use std::path::PathBuf;

use qwen_core::config::{DEFAULT_SEED, ENC_DIM, NUM_CODEBOOKS, SAMPLE_RATE};
use qwen_core::{plan_script, MarkupMode, QVoiceFile, SamplingParams, Script};
use qwen_engine::{
    clone_voice, encode_wav_bytes, write_wav, AbortHandle, CloneOptions, Delivery, EdgeRampSink,
    LogLevel, Model06B, ModelLoadOptions, SessionScratch, SynthesisRequest, Synthesizer, Voice,
    VoiceLoadOptions,
};
use qwen_kernels::ComputeRuntime;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("qwen_engine_synthesis_paths_tests")
        .join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn load_model(runtime: &ComputeRuntime) -> Option<Model06B> {
    let model_dir = repo_root().join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        return None;
    }
    Some(
        Model06B::load(
            &model_dir,
            runtime,
            ModelLoadOptions {
                quant_mixed: true,
                silent: true,
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

/// Un perfil `.qvoice` en memoria, con `n_frames` fotogramas ICL deterministas.
fn sample_qvoice(name: &str, lang_id: u32, ref_text: &str, n_frames: usize) -> QVoiceFile {
    QVoiceFile {
        speaker_embedding: (0..ENC_DIM)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
            .collect(),
        voice_name: name.to_string(),
        lang_id,
        ref_text: ref_text.to_string(),
        ref_codes: (0..n_frames * NUM_CODEBOOKS)
            .map(|i| ((i * 37 + 11) % 2048) as i32)
            .collect(),
    }
}

/// Bytes esperados de un `.qvoice` v4, escritos a mano campo a campo.
fn expected_qvoice_bytes(qv: &QVoiceFile) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"QVCE");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&(ENC_DIM as u32).to_le_bytes());
    for v in &qv.speaker_embedding {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&(qv.voice_name.len() as u32).to_le_bytes());
    b.extend_from_slice(qv.voice_name.as_bytes());
    b.extend_from_slice(&qv.lang_id.to_le_bytes());
    b.extend_from_slice(&(qv.ref_text.len() as u32).to_le_bytes());
    b.extend_from_slice(qv.ref_text.as_bytes());
    b.extend_from_slice(&(qv.n_ref_frames() as u32).to_le_bytes());
    for c in &qv.ref_codes {
        b.extend_from_slice(&c.to_le_bytes());
    }
    b
}

fn icl_cap_env_defined() -> bool {
    std::env::var("QWEN_ICL_FRAMES").is_ok()
}

#[test]
fn test_voice_load_and_icl_cap_on_construction() {
    let dir = temp_dir("voice_load");
    let ref_text = "Texto de referencia.";

    // Con ICL, nombre propio e idioma
    let icl_qv = sample_qvoice("mi_voz", 2050, ref_text, 10);
    let icl_path = dir.join("icl.qvoice");
    icl_qv.write_to_file(&icl_path).unwrap();
    // Solo x-vector, sin nombre ni idioma
    let xv_qv = sample_qvoice("", 0, "", 0);
    let xv_path = dir.join("solo_x.qvoice");
    xv_qv.write_to_file(&xv_path).unwrap();

    let opts = VoiceLoadOptions {
        use_icl: true,
        log: LogLevel::Silent,
        ..Default::default()
    };
    let full = Voice::load(&icl_path, &opts).unwrap();
    assert_eq!(full.name, "mi_voz");
    assert_eq!(full.language_id, Some(2050));
    assert_eq!(full.xvector, icl_qv.speaker_embedding);

    let xv = Voice::load(&xv_path, &opts).unwrap();
    assert_eq!(xv.name, "solo_x", "sin nombre guardado, el del archivo");
    assert_eq!(xv.language_id, None);
    assert!(xv.icl.is_none());

    // Sin `.qvoice` en la ruta: error
    assert!(Voice::load(&dir.join("voz.bin"), &opts).is_err());

    // `use_icl = false` descarta el ICL; `ref_text_override` sustituye el texto
    let no_icl = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            use_icl: false,
            ..opts.clone()
        },
    )
    .unwrap();
    assert!(no_icl.icl.is_none());
    let over = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            ref_text_override: Some("Otra transcripción.".to_string()),
            ..opts.clone()
        },
    )
    .unwrap();
    assert_eq!(over.icl.unwrap().ref_text, "Otra transcripción.");

    // `QWEN_ICL_FRAMES` manda sobre el tope: si está definida, estas comprobaciones no aplican.
    if icl_cap_env_defined() {
        eprintln!("QWEN_ICL_FRAMES definida: se omite el tope de fotogramas");
        return;
    }
    let full_icl = full.icl.as_ref().unwrap();
    assert_eq!(full_icl.ref_text, ref_text);
    assert_eq!(full_icl.ref_codes, icl_qv.ref_codes, "tope 0 = sin recorte");

    // El tope se aplica al construir: los primeros N fotogramas, sin tocar el archivo
    let capped = Voice::load(
        &icl_path,
        &VoiceLoadOptions {
            icl_frames_cap: 4,
            ..opts.clone()
        },
    )
    .unwrap();
    let capped_icl = capped.icl.unwrap();
    assert_eq!(capped_icl.ref_codes.len(), 4 * NUM_CODEBOOKS);
    assert_eq!(
        capped_icl.ref_codes[..],
        icl_qv.ref_codes[..4 * NUM_CODEBOOKS]
    );
    assert_eq!(QVoiceFile::read_from_file(&icl_path).unwrap(), icl_qv);

    // Un tope mayor que el ICL no recorta
    let loose = Voice::from_qvoice(
        icl_qv.clone(),
        &VoiceLoadOptions {
            icl_frames_cap: 99,
            ..opts
        },
    );
    assert_eq!(loose.icl.unwrap().ref_codes, icl_qv.ref_codes);
}

#[test]
fn test_buffer_and_stream_deliver_same_pcm_and_clone_bytes() {
    let runtime = ComputeRuntime::new(Some(4));
    let Some(model) = load_model(&runtime) else {
        eprintln!(
            "Se omite test_buffer_and_stream_deliver_same_pcm_and_clone_bytes (falta el modelo)"
        );
        return;
    };
    let dir = temp_dir("delivery");
    let abort = AbortHandle::new();
    let mut scratch = SessionScratch::new();
    let default_voice = Voice::builtin_default();
    let req = SynthesisRequest {
        language_id: 2054,
        sampling: SamplingParams {
            seed: DEFAULT_SEED,
            ..SamplingParams::default()
        },
        max_frames: 120,
        volume: 0.7,
        log: LogLevel::Silent,
        ..Default::default()
    };

    // Referencia con la voz `default`, para clonar una voz con ICL
    let ref_text = "Esta es una referencia para clonar con contexto.";
    let ref_req = SynthesisRequest {
        max_frames: 8192,
        volume: 1.0,
        ..req.clone()
    };
    let ref_samples = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &default_voice,
    }
    .synthesize(
        &mut scratch,
        &ref_req,
        &abort,
        &Script::Plain(ref_text.to_string()),
        Delivery::Buffer,
    )
    .unwrap()
    .samples
    .unwrap();
    let ref_wav = dir.join("ref.wav");
    write_wav(&ref_wav, &ref_samples, SAMPLE_RATE as u32).unwrap();

    // Bytes del `QVoiceFile` de `clone_voice`: archivo completo, sin recortar
    let (qv, built) = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            ref_text: Some(ref_text.to_string()),
            max_ref_seconds: 30.0,
            name: None,
            language_id: None,
            icl_frames_cap: 5,
            log: LogLevel::Silent,
        },
    )
    .unwrap();
    assert_eq!(qv.speaker_embedding.len(), ENC_DIM);
    assert_eq!(
        qv.voice_name, "",
        "sin nombre dado, el archivo guarda el vacío"
    );
    assert_eq!(qv.lang_id, 0, "sin idioma dado, el archivo guarda 0");
    assert_eq!(qv.ref_text, ref_text);
    assert!(
        qv.n_ref_frames() > 5,
        "el archivo conserva todos los fotogramas"
    );
    assert_eq!(built.name, "ref", "la voz construida usa el nombre del WAV");
    assert_eq!(built.language_id, None);
    assert_eq!(built.xvector, qv.speaker_embedding);
    let built_icl = built.icl.as_ref().unwrap();
    assert_eq!(built_icl.ref_text, ref_text);
    if !icl_cap_env_defined() {
        assert_eq!(built_icl.ref_codes.len(), 5 * NUM_CODEBOOKS);
        assert_eq!(built_icl.ref_codes[..], qv.ref_codes[..5 * NUM_CODEBOOKS]);
    }
    let qv_path = dir.join("clone.qvoice");
    qv.write_to_file(&qv_path).unwrap();
    assert_eq!(fs::read(&qv_path).unwrap(), expected_qvoice_bytes(&qv));
    let (named, named_voice) = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            ref_text: None,
            max_ref_seconds: 30.0,
            name: Some("nombrada".to_string()),
            language_id: Some(2050),
            icl_frames_cap: 0,
            log: LogLevel::Silent,
        },
    )
    .unwrap();
    assert_eq!(
        (named.voice_name.as_str(), named.lang_id),
        ("nombrada", 2050)
    );
    assert!(named.ref_codes.is_empty() && named.ref_text.is_empty());
    assert_eq!(
        (named_voice.name.as_str(), named_voice.language_id),
        ("nombrada", Some(2050))
    );
    assert!(named_voice.icl.is_none());
    assert_eq!(named.speaker_embedding, qv.speaker_embedding);
    let named_path = dir.join("named.qvoice");
    named.write_to_file(&named_path).unwrap();
    assert_eq!(
        fs::read(&named_path).unwrap(),
        expected_qvoice_bytes(&named)
    );

    // Buffer frente a Stream: mismos bytes PCM en texto plano, ICL y marcado
    let icl_voice = Voice::load(
        &qv_path,
        &VoiceLoadOptions {
            use_icl: true,
            icl_frames_cap: 5,
            log: LogLevel::Silent,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(icl_voice.icl.is_some());
    let marked = plan_script("Hola. [pause:0.3] Adiós.", MarkupMode::Auto);
    assert!(matches!(marked.script, Script::Spans(_)));
    let cases = [
        (
            "texto plano",
            &default_voice,
            Script::Plain("Hola mundo.".to_string()),
        ),
        ("ICL", &icl_voice, Script::Plain("Hola mundo.".to_string())),
        ("marcado", &default_voice, marked.script),
    ];
    for (label, voice, script) in &cases {
        let synth = Synthesizer {
            runtime: &runtime,
            model: &model,
            voice,
        };
        let buffered = synth
            .synthesize(&mut scratch, &req, &abort, script, Delivery::Buffer)
            .unwrap()
            .samples
            .unwrap();
        // La entrega completa aplica las rampas de borde al codificar el WAV (sin cabecera: 44 B).
        let buffer_pcm = encode_wav_bytes(&buffered, SAMPLE_RATE as u32)[44..].to_vec();

        let mut sink = EdgeRampSink::new(Vec::<u8>::new(), SAMPLE_RATE as u32);
        let streamed = synth
            .synthesize(
                &mut scratch,
                &req,
                &abort,
                script,
                Delivery::Stream(&mut sink),
            )
            .unwrap();
        assert!(
            streamed.samples.is_none(),
            "{label}: Stream no devuelve muestras"
        );
        let stream_pcm = sink.into_writer();

        assert!(!buffer_pcm.is_empty(), "{label}: audio vacío");
        assert_eq!(
            buffer_pcm.len(),
            stream_pcm.len(),
            "{label}: distinta longitud de PCM"
        );
        assert!(
            buffer_pcm == stream_pcm,
            "{label}: Buffer y Stream deben dar los mismos bytes PCM"
        );
    }
}
