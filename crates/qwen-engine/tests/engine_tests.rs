// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use qwen_core::config::{
    lookup_language, DEFAULT_SEED, DEFAULT_TEMPERATURE, NUM_CODEBOOKS, SAMPLE_RATE,
};
use qwen_engine::model::{DecoderWorkspace, SdStreamState};
use qwen_engine::{
    encode_wav_bytes, speech_decoder_decode_streaming,
    AbortHandle, Delivery, EngineError, LogLevel, MemorySink, Model06B,
    SessionScratch, SynthesisRequest, Synthesizer, Voice,
};
use qwen_core::Script;
use qwen_kernels::ComputeRuntime;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_e2e_determinism_and_abort() {
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Se omite test_e2e_determinism_and_abort (falta el modelo)");
        return;
    }

    let runtime = ComputeRuntime::new(Some(4));
    let model = Model06B::load(
        &model_dir,
        &runtime,
        true,
    )
    .unwrap();

    let lang = lookup_language("Spanish").unwrap();
    let voice = Voice::builtin_default();

    let req = SynthesisRequest {
        language_id: lang,
        log: LogLevel::Silent,
        ..Default::default()
    };
    assert_eq!(req.sampling.temperature, DEFAULT_TEMPERATURE);
    assert_eq!(req.sampling.seed, DEFAULT_SEED);

    let abort = AbortHandle::new();
    let mut scratch1 = SessionScratch::new();
    let script = Script::Plain("Hola, ¿cómo estás?".to_string());
    let synth = Synthesizer {
        runtime: &runtime,
        model: &model,
        voice: &voice,
    };
    let out1 = synth
        .synthesize(&mut scratch1, &req, &abort, &script, Delivery::Buffer)
        .unwrap();
    let (s1, t1) = (out1.samples.unwrap(), out1.timing);
    assert!(t1.frames >= 10 && !s1.is_empty());

    let mut scratch2 = SessionScratch::new();
    let out2 = synth
        .synthesize(&mut scratch2, &req, &abort, &script, Delivery::Buffer)
        .unwrap();
    let (s2, t2) = (out2.samples.unwrap(), out2.timing);
    assert_eq!(t1.frames, t2.frames);
    assert_eq!(
        encode_wav_bytes(&s1, SAMPLE_RATE as u32),
        encode_wav_bytes(&s2, SAMPLE_RATE as u32),
        "Dos ejecuciones idénticas de Rust deben producir bytes WAV idénticos bit a bit"
    );

    // Comprobación de cancelación mediante AbortHandle
    let abort_now = AbortHandle::new();
    abort_now.cancel();
    let mut sink = MemorySink::new(0);
    let err = synth
        .synthesize(
            &mut scratch2,
            &req,
            &abort_now,
            &script,
            Delivery::Stream(&mut sink),
        )
        .err()
        .unwrap();
    assert!(matches!(err, EngineError::Aborted));

    // Rechazo explícito de buffers de códigos truncados en SpeechDecoder
    let bad_codes = vec![0i32; NUM_CODEBOOKS - 1];
    let mut bad_st = SdStreamState::new();
    let mut bad_ws = DecoderWorkspace::new();
    assert!(matches!(
        speech_decoder_decode_streaming(&runtime, &model.decoder, &mut bad_st, &mut bad_ws, &bad_codes, 1),
        Err(EngineError::InvalidArg(_))
    ));
}
