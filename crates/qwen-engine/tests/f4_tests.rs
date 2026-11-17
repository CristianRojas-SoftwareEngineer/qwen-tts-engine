use std::fs;
use std::path::PathBuf;

use qwen_core::config::{ENC_DIM, NUM_CODEBOOKS, SAMPLE_RATE};
use qwen_core::{write_bin_embedding, MarkupPlanner, QVoiceFile};
use qwen_engine::{
    apply_emotion_direction, attach_emo_ref, build_voice_from_ref_audio, encode_wav_bytes,
    load_voice_from_file, mel_spectrogram, prepare_markup_06b, read_wav_24khz,
    render_spans_buffer, resolve_emovoice_06b, synthesize_to_samples, trim_trailing_silence,
    validate_wav_path_or_header, write_wav, EmoVoice06BAction, Model06B, ModelLoadOptions,
    RequestConfig, RequestExecution, SaveVoiceOptions, SessionScratch, VoiceBindings,
    VoiceIdentity, VoiceLoadOptions,
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
    let dir = std::env::temp_dir().join("qwen_engine_f4_tests").join(name);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_wav_validation_and_ffmpeg_hints() {
    let dir = temp_dir("wav_validation");

    // 1. Extension rejection (.mp3, .mp4, .flac, .ogg)
    for ext in ["mp3", "mp4", "ogg", "flac"] {
        let p = dir.join(format!("fake.{ext}"));
        fs::write(&p, b"dummy").unwrap();
        let err = validate_wav_path_or_header(&p).unwrap_err().to_string();
        assert!(err.contains("ffmpeg -i"), "Expected ffmpeg hint for .{ext}: {err}");
        assert!(err.contains("24000"), "Expected 24000 Hz hint for .{ext}: {err}");
    }

    // 2. Magic-byte rejection even with .wav extension
    let bad_mp3 = dir.join("disguised_mp3.wav");
    fs::write(&bad_mp3, b"ID3\x03\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
    let err = read_wav_24khz(&bad_mp3).unwrap_err().to_string();
    assert!(err.contains("MP3") && err.contains("ffmpeg -i"), "got: {err}");

    let bad_ogg = dir.join("disguised_ogg.wav");
    fs::write(&bad_ogg, b"OggS\x00\x02\x00\x00\x00\x00\x00\x00").unwrap();
    let err = read_wav_24khz(&bad_ogg).unwrap_err().to_string();
    assert!(err.contains("OGG") && err.contains("ffmpeg -i"), "got: {err}");

    // 3. Valid WAV at wrong sample rate (16000 Hz) must fail explicitly
    let wav_16k = dir.join("ref_16k.wav");
    let samples = vec![0.1f32; 16000];
    write_wav(&wav_16k, &samples, 16000).unwrap();
    let err = read_wav_24khz(&wav_16k).unwrap_err().to_string();
    assert!(
        err.contains("24000 Hz") && err.contains("16000 Hz") && err.contains("ffmpeg -i"),
        "Expected explicit 24kHz error, got: {err}"
    );

    // 4. Valid 24 kHz WAV succeeds
    let wav_24k = dir.join("ref_24k.wav");
    let samples24: Vec<f32> = (0..24000).map(|i| ((i as f32) * 0.05).sin() * 0.2).collect();
    write_wav(&wav_24k, &samples24, 24000).unwrap();
    let read_back = read_wav_24khz(&wav_24k).unwrap();
    assert_eq!(read_back.len(), 24000);
}

#[test]
fn test_trim_trailing_silence_and_mel_shape() {
    let mut clip = vec![0.0f32; 24000 * 5];
    for (i, v) in clip[..24000 * 3].iter_mut().enumerate() {
        *v = ((i as f32) * 0.05).sin() * 0.3;
    }
    trim_trailing_silence(&mut clip, 24000, true);
    assert!(clip.len() < 24000 * 5 && clip.len() >= 24000 * 3);

    let (mel, n_frames) = mel_spectrogram(&clip, 24000).unwrap();
    assert!(n_frames > 250);
    assert_eq!(mel.len(), n_frames * 128);
    assert!(mel.iter().all(|v| v.is_finite()));
}

#[test]
fn test_emovoice_resolution_and_l2_norm_direction() {
    let dir = temp_dir("emovoice");
    let preset_dir = dir.join("presets");
    fs::create_dir_all(&preset_dir).unwrap();

    // Unknown emotion -> error
    let err = resolve_emovoice_06b("nonexistent_mood", None, false, Some("ryan"), &[&preset_dir])
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a known emotion"), "got: {err}");

    // Missing asset -> error
    let err = resolve_emovoice_06b("joy", None, false, Some("ryan"), &[&preset_dir])
        .unwrap_err()
        .to_string();
    assert!(err.contains("no 'joy' voice asset"), "got: {err}");

    // Direction fallback on clone preserves L2 norm
    let dir_bin = preset_dir.join("dir_sad.bin");
    let mut unit_dir = vec![0.0f32; ENC_DIM];
    unit_dir[0] = 1.0;
    write_bin_embedding(&dir_bin, &unit_dir).unwrap();

    let clone_qvoice = dir.join("alice_graft.qvoice");
    fs::write(&clone_qvoice, b"dummy").unwrap();
    let act = resolve_emovoice_06b(
        "sadness",
        Some(&clone_qvoice),
        true,
        None,
        &[&preset_dir],
    )
    .unwrap();
    assert!(matches!(
        act,
        EmoVoice06BAction::ApplyDirection { tok: "sad", .. }
    ));

    let mut emb = vec![0.25f32; ENC_DIM];
    let n0 = emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
    apply_emotion_direction(&mut emb, &dir_bin, 0.25).unwrap();
    let n1 = emb.iter().map(|&v| v * v).sum::<f32>().sqrt();
    assert!((n0 - n1).abs() < 1e-4, "L2 norm drifted: {n0} -> {n1}");

    // Dedicated .qvoice wins over .bin and dir_*.bin
    let dedicated_qv = preset_dir.join("alice_sad.qvoice");
    fs::write(&dedicated_qv, b"dummy").unwrap();
    let act2 = resolve_emovoice_06b(
        "sad",
        Some(&clone_qvoice),
        true,
        None,
        &[&preset_dir],
    )
    .unwrap();
    assert_eq!(act2, EmoVoice06BAction::LoadQVoiceIclOnly(dedicated_qv));
}

#[test]
fn test_clone_wdelta_q4_error_and_e2e_f4() {
    let root = repo_root();
    let cv_dir = root.join("models/qwen3-tts-0.6b");
    let base_dir = root.join("models/qwen3-tts-0.6b-base");
    if !cv_dir.exists() || !base_dir.exists() {
        eprintln!("Skipping test_clone_wdelta_q4_error_and_e2e_f4 (models absent)");
        return;
    }

    let dir = temp_dir("e2e_f4");
    let runtime = ComputeRuntime::new(Some(4));

    // 1. Generate a 24 kHz reference WAV using 0.6B-CustomVoice
    let ref_wav = dir.join("ref.wav");
    let ref_text = "The weather is beautiful today, perfect for a walk in the park.";
    let mut cv_model = Model06B::load(
        &cv_dir,
        &runtime,
        ModelLoadOptions {
            quant_mixed: true,
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();
    let mut scratch = SessionScratch::new();
    let preset_voice = VoiceBindings::preset(3061);
    let exec_ref = RequestExecution::new(RequestConfig {
        speaker_id: 3061,
        language_id: Some(2050),
        seed: 42,
        silent: true,
        ..Default::default()
    });
    let (ref_samples, _) = synthesize_to_samples(
        &runtime,
        &cv_model,
        &preset_voice,
        &mut scratch,
        &exec_ref,
        ref_text,
    )
    .unwrap();
    write_wav(&ref_wav, &ref_samples, SAMPLE_RATE as u32).unwrap();

    // --ref-audio on CustomVoice must fail explicitly
    let err = build_voice_from_ref_audio(
        &runtime,
        &cv_model,
        &ref_wav,
        None,
        &SaveVoiceOptions {
            ref_text: Some(ref_text),
            silent: true,
            ..Default::default()
        },
    );
    assert!(err.is_err());

    // --emo-ref without --emo-ref-text must fail explicitly
    let mut emo_voice = VoiceBindings::preset(3061);
    let err = attach_emo_ref(&runtime, &cv_model, &mut emo_voice, &ref_wav, "   ", true, false);
    assert!(err.is_err());

    // --emo-ref with text succeeds on CustomVoice
    attach_emo_ref(
        &runtime,
        &cv_model,
        &mut emo_voice,
        &ref_wav,
        ref_text,
        true,
        false,
    )
    .unwrap();
    assert!(emo_voice.emo_ref_codes.as_ref().unwrap().len() >= NUM_CODEBOOKS);

    // 2. Build .qvoice with WDELTA from 0.6B-Base -> CustomVoice
    let base_model = Model06B::load(
        &base_dir,
        &runtime,
        ModelLoadOptions {
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();
    let wdelta_qv = dir.join("clone_wdelta.qvoice");
    let base_voice = build_voice_from_ref_audio(
        &runtime,
        &base_model,
        &ref_wav,
        Some(&wdelta_qv),
        &SaveVoiceOptions {
            ref_text: Some(ref_text),
            max_ref_seconds: 30.0,
            language: Some("English"),
            voice_name: Some("test_clone"),
            target_cv_dir: Some(&cv_dir),
            silent: true,
            debug: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(matches!(base_voice.identity, VoiceIdentity::IclGraft { .. }));

    let parsed_qv = QVoiceFile::read_from_file(&wdelta_qv, false).unwrap();
    assert!(parsed_qv.weights.is_some());
    assert!(parsed_qv.ref_codes.len() >= NUM_CODEBOOKS);

    // 3. Loading WDELTA .qvoice without --icl-only on Q4_0 (--quant-mixed) MUST FAIL explicitly
    let err = load_voice_from_file(
        &runtime,
        &mut cv_model,
        &wdelta_qv,
        None,
        &VoiceLoadOptions {
            icl_only: false,
            silent: true,
            ..Default::default()
        },
    );
    assert!(
        err.is_err(),
        "WDELTA on Q4_0 model without --icl-only must fail"
    );
    let err_msg = err.err().unwrap().to_string();
    assert!(err_msg.contains("q4_0") || err_msg.contains("Q4_0"), "got: {err_msg}");

    // 4. Loading same WDELTA .qvoice WITH --icl-only on Q4_0 succeeds and synthesizes!
    let (icl_voice, _) = load_voice_from_file(
        &runtime,
        &mut cv_model,
        &wdelta_qv,
        None,
        &VoiceLoadOptions {
            icl_only: true,
            silent: true,
            ..Default::default()
        },
    )
    .unwrap();
    let (clone_samples, t_clone) = synthesize_to_samples(
        &runtime,
        &cv_model,
        &icl_voice,
        &mut scratch,
        &RequestExecution::new(RequestConfig {
            speaker_id: 3061,
            language_id: Some(2050),
            seed: 42,
            silent: true,
            ..Default::default()
        }),
        "Hello, this is a test of the cloned voice running with the new engine.",
    )
    .unwrap();
    assert!(t_clone.frames >= 10 && !clone_samples.is_empty());

    // 5. Roughness (Q2_0 down_proj blend) + C6_para MarkupPlanner multi-span synthesis
    let prep = prepare_markup_06b(
        "Che bella notizia! [pause:400ms] [sad] Devo andare... [sigh] [neutral] Ciao.",
        &preset_voice,
        false,
        Some(42),
        None,
        0.9,
    );
    assert!(prep.did_para_substitute);
    assert!(prep.use_compose);
    let spans = MarkupPlanner::parse_spans(&prep.text);
    assert!(spans.len() >= 3);
    let (composed, n_spoken) = render_spans_buffer(
        &runtime,
        &cv_model,
        &preset_voice,
        &mut scratch,
        &RequestExecution::new(RequestConfig {
            speaker_id: 3061,
            language_id: Some(2070),
            seed: prep.seed,
            temperature: prep.temperature,
            cp_roughness: 0.35,
            silent: true,
            ..Default::default()
        }),
        &spans,
        0.12,
    )
    .unwrap();
    assert!(n_spoken >= 2 && composed.len() > SAMPLE_RATE);
    let _ = encode_wav_bytes(&composed, SAMPLE_RATE as u32);
}
