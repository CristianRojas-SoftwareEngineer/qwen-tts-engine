use std::path::PathBuf;

use qwen_core::config::{lookup_language, lookup_speaker, HIDDEN_SIZE, TEXT_HIDDEN_SIZE};
use qwen_core::{
    decode_wdelta_bf16, emotion_name_to_tok, encode_wdelta_bf16, read_bin_embedding,
    write_bin_embedding, BpeTokenizer, LcgRng, MarkupPlanner, MmapStore, ModelConfig06B,
    QVoiceFile, Sampler, SpanSpec, VoiceClass,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn test_config_load_06b_and_reject_17b_voicedesign() {
    let root = repo_root();
    let cv_dir = root.join("models/qwen3-tts-0.6b");
    let base_dir = root.join("models/qwen3-tts-0.6b-base");

    let raw_ok = if cv_dir.exists() && base_dir.exists() {
        let cfg_cv = ModelConfig06B::load_and_validate(&cv_dir).expect("0.6b CustomVoice valid");
        assert!(!cfg_cv.is_base_model);
        assert_eq!(cfg_cv.tts_model_type, "custom_voice");
        assert_eq!(cfg_cv.spk_ids.get("ryan"), Some(&3061));

        let cfg_base = ModelConfig06B::load_and_validate(&base_dir).expect("0.6b Base valid");
        assert!(cfg_base.is_base_model);
        assert_eq!(cfg_base.tts_model_type, "base");

        std::fs::read_to_string(cv_dir.join("config.json")).unwrap()
    } else {
        r#"{"model_type":"qwen3_tts","tts_model_size":"0.6b","tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"text_hidden_size":2048,"text_vocab_size":151936,"vocab_size":3072,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"code_predictor_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":5,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"vocab_size":2048,"num_code_groups":16,"rms_norm_eps":1e-6}}}"#.to_string()
    };

    // Negative validation: 1.7B & VoiceDesign
    let tmp = std::env::temp_dir().join("qwen_core_cfg_neg_test");
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let bad_17b = raw_ok.replace("\"hidden_size\":1024", "\"hidden_size\":2048").replace("\"hidden_size\": 1024", "\"hidden_size\": 2048");
    std::fs::write(tmp.join("config.json"), bad_17b).unwrap();
    let err = ModelConfig06B::load_and_validate(&tmp).unwrap_err();
    assert!(err.to_string().contains("1.7B") || err.to_string().contains("2048"));

    let bad_vd = raw_ok.replace("\"custom_voice\"", "\"voice_design\"");
    std::fs::write(tmp.join("config.json"), bad_vd).unwrap();
    let err = ModelConfig06B::load_and_validate(&tmp).unwrap_err();
    assert!(err.to_string().contains("VoiceDesign"));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_mmap_store_06b() {
    let root = repo_root();
    let cv_dir = root.join("models/qwen3-tts-0.6b");
    if !cv_dir.exists() {
        eprintln!("Skipping test_mmap_store_06b (models absent)");
        return;
    }
    let store = MmapStore::open_model(&cv_dir).expect("open 0.6b model.safetensors");
    let tok_emb = store
        .tensor_bf16("talker.model.text_embedding.weight")
        .unwrap();
    assert_eq!(tok_emb.len(), 151_936 * TEXT_HIDDEN_SIZE);
    let norm = store.tensor_f32("talker.model.norm.weight").unwrap();
    assert_eq!(norm.len(), HIDDEN_SIZE);

    let sd_store = MmapStore::open_model(&cv_dir.join("speech_tokenizer"))
        .expect("open speech_tokenizer/model.safetensors");
    assert!(sd_store.contains("decoder.pre_conv.conv.weight"));
}

#[test]
fn test_bpe_exact_parity_with_c() {
    let root = repo_root();
    let cv_dir = root.join("models/qwen3-tts-0.6b");
    if !cv_dir.exists() {
        eprintln!("Skipping test_bpe_exact_parity_with_c (models absent)");
        return;
    }
    let tok = BpeTokenizer::load_from_dir(&cv_dir).unwrap();

    let cases: &[(&str, &[i32])] = &[
        ("Hola, ¿cómo estás?", &[68012, 11, 28286, 66, 71461, 1788, 7061, 30]),
        (
            "The quick brown fox jumps over the lazy dog on a sunny afternoon.",
            &[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 389, 264, 39698, 13354, 13],
        ),
        (
            "Buongiorno a tutti, oggi è una splendida giornata di primavera e voglio raccontarvi una storia davvero speciale. Camminando lungo il sentiero del parco, si sente il profumo dei fiori freschi e il canto leggero degli uccelli tra gli alberi antichi. È il momento perfetto per fermarsi, respirare profondamente e godersi la tranquillità di questo luogo meraviglioso.",
            &[
                59808, 644, 72, 11121, 264, 43280, 11, 88644, 11422, 5093, 12503, 408, 4744, 59010,
                459, 1853, 8860, 402, 2416, 384, 62939, 25147, 8971, 772, 277, 9971, 5093, 98889,
                40769, 88810, 1398, 19893, 13, 8055, 1065, 4883, 20622, 78, 3815, 3208, 43588, 1594,
                1346, 1015, 11, 4403, 3208, 68, 3815, 2778, 64625, 21805, 282, 2462, 72, 48873,
                14604, 384, 3815, 272, 13561, 2472, 1389, 78, 46569, 575, 638, 20508, 9586, 27932,
                452, 652, 72, 3196, 40552, 13, 131390, 3815, 30121, 21784, 52879, 817, 80582, 60014,
                11, 32415, 546, 2778, 2111, 8980, 384, 9886, 388, 72, 1187, 40564, 483, 23940, 1853,
                33460, 24743, 23500, 4704, 2957, 742, 27147, 13,
            ],
        ),
        (
            "The weather is beautiful today, perfect for a walk in the park.",
            &[785, 9104, 374, 6233, 3351, 11, 4727, 369, 264, 4227, 304, 279, 6118, 13],
        ),
        (
            "Hello, this is a test of the cloned voice running with the new engine.",
            &[9707, 11, 419, 374, 264, 1273, 315, 279, 54119, 7743, 4303, 448, 279, 501, 4712, 13],
        ),
        (
            "Che bella notizia! , 唉, Devo andare... Ciao.",
            &[26843, 96460, 537, 62041, 0, 1154, 4891, 242, 231, 11, 1581, 3334, 323, 546, 1112, 356, 22516, 13],
        ),
        (
            "Oh! , 哈哈哈, That is really wonderful news!",
            &[11908, 0, 1154, 4891, 241, 230, 100323, 11, 2938, 374, 2167, 11117, 3669, 0],
        ),
        (
            "I'm sure we've got 12345 things to do at 400ms!",
            &[40, 2776, 2704, 582, 3003, 2684, 220, 16, 17, 18, 19, 20, 2513, 311, 653, 518, 220, 19, 15, 15, 1011, 0],
        ),
    ];

    for (idx, &(text, expected)) in cases.iter().enumerate() {
        let got = tok.encode(text);
        assert_eq!(got.as_slice(), expected, "BPE mismatch on case {idx}: {text}");
    }
    assert_eq!(tok.decode(cases[0].1), cases[0].0);
    assert_eq!(tok.decode(cases[2].1), cases[2].0);
}

#[test]
fn test_qvoice_and_bin_roundtrip() {
    let emb: Vec<f32> = (0..1024).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect();
    let tmp_bin = std::env::temp_dir().join("qwen_core_roundtrip.bin");
    write_bin_embedding(&tmp_bin, &emb).unwrap();
    let emb2 = read_bin_embedding(&tmp_bin).unwrap();
    assert_eq!(emb, emb2);
    let _ = std::fs::remove_file(&tmp_bin);

    // .qvoice v3 (META + TPAD + WOVR) roundtrip
    let qv = QVoiceFile {
        version: 3,
        enc_dim: 1024,
        speaker_embedding: emb.clone(),
        ref_text: "Hello test".to_string(),
        ref_codes: vec![12; 16 * 4],
        meta: Some(qwen_core::QVoiceMeta {
            lang_id: 2050,
            lang_name: "English".to_string(),
            model_size: 1024,
            enc_dim: 1024,
            ref_dur: 0.32,
            voice_name: "test_voice".to_string(),
            flags: 0,
        }),
        tpad: Some(qwen_core::QVoiceTpad {
            hidden: 1024,
            tts_pad_embed: vec![0.1; 1024],
            tts_bos_embed: vec![0.2; 1024],
            tts_eos_embed: vec![0.3; 1024],
        }),
        wovr: Some(qwen_core::QVoiceWovr {
            wh: 1024,
            wth: 2048,
            wcv: 2048,
            fc1_bf16: vec![0x3F80; 2048 * 2048],
            fc1_bias: vec![0.01; 2048],
            fc2_bf16: vec![0x3F80; 1024 * 2048],
            fc2_bias: vec![0.02; 1024],
            codec_emb_bf16: vec![0x3F00; 2048 * 1024],
        }),
        weights: None,
    };
    let tmp_qv = std::env::temp_dir().join("qwen_core_roundtrip.qvoice");
    qv.write_to_file(&tmp_qv).unwrap();
    let qv2 = QVoiceFile::read_from_file(&tmp_qv, false).unwrap();
    assert_eq!(qv, qv2);
    let _ = std::fs::remove_file(&tmp_qv);

    // WDELTA LZ4 roundtrip
    let cv: Vec<u16> = (0..1024).map(|i| 0x3F00u16.wrapping_add((i % 7) as u16)).collect();
    let base: Vec<u16> = (0..1024).map(|i| 0x3F00u16.wrapping_add((i % 11) as u16)).collect();
    let wt = encode_wdelta_bf16("talker.model.layers.12.self_attn.q_proj.weight", &base, &cv).unwrap();
    let decoded = decode_wdelta_bf16(&cv, &wt, 1.0, 11, 27).unwrap();
    assert_eq!(decoded, base);
}

#[test]
fn test_markup_sampling_and_lookups() {
    assert_eq!(lookup_speaker("RYAN").unwrap().id, 3061);
    assert!(lookup_speaker("nobody").is_none());
    assert_eq!(lookup_language("Italian"), Some(2070));
    assert!(lookup_language("Klingon").is_none());
    assert_eq!(emotion_name_to_tok("Angry"), Some("ang"));

    let sub = MarkupPlanner::para_substitute("Oh! [laugh] Great!", VoiceClass::PresetOther);
    assert!(sub.did_substitute);
    assert_eq!(sub.seed, 2024);
    assert!((sub.temperature - 1.1).abs() < 1e-6);
    assert!(sub.text.contains("哈哈哈"));

    let spans = MarkupPlanner::parse_spans(
        "Che bella notizia! [pause:400ms] [sad] Devo andare... [neutral] Ciao.",
    );
    assert_eq!(spans.len(), 4);
    assert_eq!(
        spans[1],
        SpanSpec::Pause {
            seconds: 0.4
        }
    );

    let mut sampler = Sampler::new();
    let mut rng1 = LcgRng::new(42);
    let mut rng2 = LcgRng::new(42);
    let mut l1 = vec![0.1f32, 2.0, -0.5, 1.5, 0.3];
    let mut l2 = l1.clone();
    let s1 = sampler.sample(&mut l1, 0.9, 50, 1.0, 1.05, &[1], &mut rng1);
    let s2 = sampler.sample(&mut l2, 0.9, 50, 1.0, 1.05, &[1], &mut rng2);
    assert_eq!(s1, s2);
    assert_eq!(rng1.seed, rng2.seed);
}
