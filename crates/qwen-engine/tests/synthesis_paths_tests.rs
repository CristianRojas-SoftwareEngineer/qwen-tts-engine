// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Equivalencia entre entregas (`Buffer` frente a `Stream`), construcción de voces y bytes del
//! perfil `.qvoice` que produce `clone_voice`.

use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use qwen_core::config::{DEFAULT_SEED, ENC_DIM};
use qwen_core::{plan_script, MarkupMode, QVoiceFile, SamplingParams, Script};
use qwen_engine::{
    clone_voice, encode_wav_bytes, write_wav, AbortHandle, CloneOptions, Delivery, EdgeRampSink,
    Model06B, PcmWrite, SessionScratch, SynthesisRequest, Synthesizer, Voice,
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

/// Destino de PCM compartido: permite leer lo escrito después de ceder el sink al motor.
struct SharedBuf(Arc<Mutex<Vec<u8>>>);

impl PcmWrite for SharedBuf {
    fn write_pcm(&mut self, b: &[u8]) -> std::io::Result<()> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
            true,
        )
        .unwrap(),
    )
}

/// Un perfil `.qvoice` en memoria con un x-vector determinista.
fn sample_qvoice(lang_id: u32) -> QVoiceFile {
    QVoiceFile {
        speaker_embedding: (0..ENC_DIM)
            .map(|i| ((i % 37) as f32 - 18.0) * 0.01)
            .collect(),
        lang_id,
    }
}

/// Bytes esperados de un `.qvoice` v6 (4112 bytes), escritos a mano campo a campo.
fn expected_qvoice_bytes(qv: &QVoiceFile) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"QVCE");
    b.extend_from_slice(&6u32.to_le_bytes());
    b.extend_from_slice(&(ENC_DIM as u32).to_le_bytes());
    for v in &qv.speaker_embedding {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&qv.lang_id.to_le_bytes());
    assert_eq!(b.len(), 4112);
    b
}

#[test]
fn test_voice_load_on_construction() {
    let dir = temp_dir("voice_load");

    // Con idioma
    let lang_qv = sample_qvoice(2050);
    let lang_path = dir.join("con_idioma.qvoice");
    lang_qv.write_to_file(&lang_path).unwrap();
    // Sin idioma
    let no_lang_qv = sample_qvoice(0);
    let no_lang_path = dir.join("sin_idioma.qvoice");
    no_lang_qv.write_to_file(&no_lang_path).unwrap();

    let with_lang = Voice::load(&lang_path, true).unwrap();
    assert_eq!(
        with_lang.name, "con_idioma",
        "la voz se llama como su archivo"
    );
    assert_eq!(with_lang.language_id, Some(2050));
    assert_eq!(with_lang.xvector, lang_qv.speaker_embedding);

    let no_lang = Voice::load(&no_lang_path, true).unwrap();
    assert_eq!(
        no_lang.name, "sin_idioma",
        "la voz se llama como su archivo"
    );
    assert_eq!(no_lang.language_id, None);
    assert_eq!(no_lang.xvector, no_lang_qv.speaker_embedding);

    // Sin `.qvoice` en la ruta: error
    assert!(Voice::load(&dir.join("voz.bin"), true).is_err());
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
        silent: true,
        ..Default::default()
    };

    // Referencia con la voz `default`, para clonar una voz
    let ref_phrase = "Esta es una referencia para clonar.";
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
        &Script::Plain(ref_phrase.to_string()),
        Delivery::Buffer,
    )
    .unwrap()
    .samples
    .unwrap();
    let ref_wav = dir.join("ref.wav");
    write_wav(&ref_wav, &ref_samples).unwrap();

    // Bytes del `QVoiceFile` de `clone_voice`: archivo completo
    let qv = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            language_id: None,
            silent: true,
        },
    )
    .unwrap();
    assert_eq!(qv.speaker_embedding.len(), ENC_DIM);
    assert_eq!(qv.lang_id, 0, "sin idioma dado, el archivo guarda 0");
    let built = Voice::from_qvoice(qv.clone(), "ref".to_string());
    assert_eq!(built.name, "ref");
    assert_eq!(built.language_id, None);
    assert_eq!(built.xvector, qv.speaker_embedding);
    let qv_path = dir.join("clone.qvoice");
    qv.write_to_file(&qv_path).unwrap();
    assert_eq!(fs::read(&qv_path).unwrap(), expected_qvoice_bytes(&qv));
    let named = clone_voice(
        &runtime,
        &model,
        &ref_wav,
        &CloneOptions {
            language_id: Some(2050),
            silent: true,
        },
    )
    .unwrap();
    assert_eq!(named.lang_id, 2050);
    assert_eq!(named.speaker_embedding, qv.speaker_embedding);
    let named_path = dir.join("named.qvoice");
    named.write_to_file(&named_path).unwrap();
    assert_eq!(
        fs::read(&named_path).unwrap(),
        expected_qvoice_bytes(&named)
    );

    // Buffer frente a Stream: mismos bytes PCM en texto plano, con voz clonada y marcado
    let cloned_voice = Voice::load(
        &qv_path,
        true,
    )
    .unwrap();
    let marked = plan_script("Hola. [pause:0.3] Adiós.", MarkupMode::Auto);
    assert!(matches!(marked, Script::Spans(_)));
    let cases = [
        (
            "texto plano",
            &default_voice,
            Script::Plain("Hola mundo.".to_string()),
        ),
        (
            "voz clonada",
            &cloned_voice,
            Script::Plain("Hola mundo.".to_string()),
        ),
        ("marcado", &default_voice, marked),
    ];
    for (label, voice, script) in &cases {
        let synth = Synthesizer {
            runtime: &runtime,
            model: &model,
            voice,
        };
        let buffered_out = synth
            .synthesize(&mut scratch, &req, &abort, script, Delivery::Buffer)
            .unwrap();
        let buffered = buffered_out.samples.unwrap();
        assert_eq!(
            buffered_out.timing.samples_produced,
            buffered.len(),
            "{label}: samples_produced de la entrega completa"
        );
        // La entrega completa aplica las rampas de borde al codificar el WAV (sin cabecera: 44 B).
        let buffer_pcm = encode_wav_bytes(&buffered)[44..].to_vec();

        let shared = Arc::new(Mutex::new(Vec::<u8>::new()));
        let mut sink = EdgeRampSink::new(SharedBuf(Arc::clone(&shared)));
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
        assert_eq!(
            streamed.timing.samples_produced,
            buffered.len(),
            "{label}: samples_produced de la entrega por Stream"
        );
        let stream_pcm = shared.lock().unwrap().clone();

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
