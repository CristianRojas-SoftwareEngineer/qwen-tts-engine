// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::path::PathBuf;

use qwen_core::config::{
    effective_language, language_name, lookup_language, DEFAULT_LANGUAGE_ID, HIDDEN_SIZE,
    TEXT_HIDDEN_SIZE,
};
use qwen_core::{
    BpeTokenizer, LcgRng, MarkupPlanner, MmapStore, ModelConfig06B, QVoiceFile, Sampler, SpanSpec,
};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// `config.json` mínimo y válido de Qwen3-TTS 0.6B; `{TYPE}` y `{SPK_ENC}` se sustituyen en cada caso.
const CONFIG_TEMPLATE: &str = r#"{"model_type":"qwen3_tts","tts_model_size":"0b6","tts_model_type":"{TYPE}",{SPK_ENC}"talker_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"text_hidden_size":2048,"text_vocab_size":151936,"vocab_size":3072,"rms_norm_eps":1e-6,"rope_theta":1000000.0,"code_predictor_config":{"hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":5,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"vocab_size":2048,"num_code_groups":16,"rms_norm_eps":1e-6}}}"#;
const SPK_ENC: &str = r#""speaker_encoder_config":{"enc_dim":1024,"sample_rate":24000},"#;

fn synthetic_config(model_type: &str, with_speaker_encoder: bool) -> String {
    CONFIG_TEMPLATE
        .replace("{TYPE}", model_type)
        .replace("{SPK_ENC}", if with_speaker_encoder { SPK_ENC } else { "" })
}

fn load_synthetic(name: &str, config_json: &str) -> Result<ModelConfig06B, qwen_core::CoreError> {
    let tmp = std::env::temp_dir().join(format!("qwen_core_cfg_{name}"));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::write(tmp.join("config.json"), config_json).unwrap();
    let res = ModelConfig06B::load_and_validate(&tmp);
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

#[test]
fn test_config_only_base_loads_and_rejects_others() {
    // Base carga (sin distinguir mayúsculas en `tts_model_type`)
    let cfg = load_synthetic("base_ok", &synthetic_config("base", true)).expect("Base valid");
    assert_eq!(cfg.rms_norm_eps, qwen_core::config::RMS_NORM_EPS);
    load_synthetic("base_upper", &synthetic_config("BASE", true)).expect("BASE valid");

    // CustomVoice se rechaza con el mensaje exacto
    let msg = "This engine only loads Qwen3-TTS 0.6B Base; CustomVoice is not supported";
    let err = load_synthetic("cv", &synthetic_config("custom_voice", false)).unwrap_err();
    assert_eq!(err.to_string(), msg);
    // Un checkpoint sin `tts_model_type` tampoco es Base
    let no_type = synthetic_config("base", true).replace("\"tts_model_type\":\"base\",", "");
    assert_eq!(load_synthetic("no_type", &no_type).unwrap_err().to_string(), msg);
    // `base` sin codificador de locutor tampoco
    let err = load_synthetic("base_no_spk", &synthetic_config("base", false)).unwrap_err();
    assert_eq!(err.to_string(), msg);

    // Validación negativa: rechaza 1.7B y VoiceDesign
    let base_ok = synthetic_config("base", true);
    let bad_17b = base_ok.replace("\"hidden_size\":1024", "\"hidden_size\":2048");
    let err = load_synthetic("17b", &bad_17b).unwrap_err();
    assert!(err.to_string().contains("1.7B") || err.to_string().contains("2048"));
    let bad_vd = base_ok.replace("\"base\"", "\"voice_design\"");
    let err = load_synthetic("vd", &bad_vd).unwrap_err();
    assert!(err.to_string().contains("VoiceDesign"));

    // El checkpoint Base real (si está presente) carga
    let base_dir = repo_root().join("models/qwen3-tts-0.6b-base");
    if base_dir.exists() {
        let cfg_base = ModelConfig06B::load_and_validate(&base_dir).expect("0.6b Base valid");
        assert_eq!(cfg_base.rms_norm_eps, qwen_core::config::RMS_NORM_EPS);
    } else {
        eprintln!("Skipping real Base config check (model absent)");
    }
}

#[test]
fn test_effective_language_precedence() {
    assert_eq!(DEFAULT_LANGUAGE_ID, 2054);
    assert_eq!(effective_language(Some(2050), Some(2070)), 2050);
    assert_eq!(effective_language(None, Some(2070)), 2070);
    assert_eq!(effective_language(None, None), 2054);
}

#[test]
fn test_language_lookup_without_dialects() {
    assert_eq!(lookup_language("Spanish"), Some(2054));
    assert_eq!(lookup_language("Italian"), Some(2070));
    assert!(lookup_language("sichuan_dialect").is_none());
    assert!(lookup_language("beijing_dialect").is_none());
    assert!(lookup_language("Klingon").is_none());
    for name in [
        "English", "German", "Spanish", "Chinese", "Japanese", "French", "Korean", "Russian",
        "Italian", "Portuguese",
    ] {
        let id = lookup_language(name).expect(name);
        assert_eq!(language_name(id), Some(name));
    }
    assert_eq!(language_name(2062), None);
    assert_eq!(language_name(2074), None);
    assert_eq!(language_name(0), None);
}

#[test]
fn test_mmap_store_06b() {
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Skipping test_mmap_store_06b (models absent)");
        return;
    }
    let store = MmapStore::open_model(&model_dir).expect("open 0.6b model.safetensors");
    let tok_emb = store
        .tensor_bf16("talker.model.text_embedding.weight")
        .unwrap();
    assert_eq!(tok_emb.len(), 151_936 * TEXT_HIDDEN_SIZE);
    let norm = store.tensor_f32_ref("talker.model.norm.weight").unwrap();
    assert_eq!(norm.len(), HIDDEN_SIZE);

    let sd_store = MmapStore::open_model(&model_dir.join("speech_tokenizer"))
        .expect("open speech_tokenizer/model.safetensors");
    assert!(sd_store.contains("decoder.pre_conv.conv.weight"));
}

#[test]
fn test_bpe_golden_tokens() {
    let root = repo_root();
    let model_dir = root.join("models/qwen3-tts-0.6b-base");
    if !model_dir.exists() {
        eprintln!("Skipping test_bpe_golden_tokens (models absent)");
        return;
    }
    let tok = BpeTokenizer::load_from_dir(&model_dir).unwrap();

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

fn qvoice_emb() -> Vec<f32> {
    (0..1024).map(|i| ((i % 37) as f32 - 18.0) * 0.01).collect()
}

/// Bytes de un `.qvoice` v4 escritos a mano, para probar el lector sin pasar por el escritor.
fn qvoice_v4_bytes(name: &str, lang_id: u32, ref_text: &str, n_frames: u32) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(b"QVCE");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&1024u32.to_le_bytes());
    for v in qvoice_emb() {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b.extend_from_slice(&(name.len() as u32).to_le_bytes());
    b.extend_from_slice(name.as_bytes());
    b.extend_from_slice(&lang_id.to_le_bytes());
    b.extend_from_slice(&(ref_text.len() as u32).to_le_bytes());
    b.extend_from_slice(ref_text.as_bytes());
    b.extend_from_slice(&n_frames.to_le_bytes());
    for i in 0..(n_frames as usize * 16) {
        b.extend_from_slice(&(i as i32).to_le_bytes());
    }
    b
}

fn read_qvoice_bytes(name: &str, bytes: &[u8]) -> Result<QVoiceFile, qwen_core::CoreError> {
    let path = std::env::temp_dir().join(format!("qwen_core_qv4_{name}.qvoice"));
    std::fs::write(&path, bytes).unwrap();
    let res = QVoiceFile::read_from_file(&path);
    let _ = std::fs::remove_file(&path);
    res
}

#[test]
fn test_qvoice_v4_roundtrip_without_and_with_icl() {
    // Sin ICL: tamaño exacto 4124 + len(nombre)
    let plain = QVoiceFile {
        speaker_embedding: qvoice_emb(),
        voice_name: "plain_voice".to_string(),
        lang_id: 0,
        ref_text: String::new(),
        ref_codes: Vec::new(),
    };
    let p1 = std::env::temp_dir().join("qwen_core_rt_plain.qvoice");
    plain.write_to_file(&p1).unwrap();
    assert_eq!(std::fs::metadata(&p1).unwrap().len() as usize, 4124 + "plain_voice".len());
    let back = QVoiceFile::read_from_file(&p1).unwrap();
    assert_eq!(back, plain);
    assert!(!back.has_icl());
    let _ = std::fs::remove_file(&p1);

    // Con ICL: 4124 + len(nombre) + len(ref_text) + 64 * n_frames
    let icl = QVoiceFile {
        speaker_embedding: qvoice_emb(),
        voice_name: "icl_voz".to_string(),
        lang_id: 2054,
        ref_text: "Hello test, ¿qué tal?".to_string(),
        ref_codes: (0..16 * 4).collect(),
    };
    let p2 = std::env::temp_dir().join("qwen_core_rt_icl.qvoice");
    icl.write_to_file(&p2).unwrap();
    let expected = 4124 + icl.voice_name.len() + icl.ref_text.len() + 64 * 4;
    assert_eq!(std::fs::metadata(&p2).unwrap().len() as usize, expected);
    let back = QVoiceFile::read_from_file(&p2).unwrap();
    assert_eq!(back, icl);
    assert!(back.has_icl());
    assert_eq!(back.n_ref_frames(), 4);
    let _ = std::fs::remove_file(&p2);

    // El escritor y el lector manual coinciden byte a byte
    let by_hand = qvoice_v4_bytes("icl_voz", 2054, "Hello test, ¿qué tal?", 4);
    let p3 = std::env::temp_dir().join("qwen_core_rt_hand.qvoice");
    icl.write_to_file(&p3).unwrap();
    assert_eq!(std::fs::read(&p3).unwrap(), by_hand);
    let _ = std::fs::remove_file(&p3);
}

#[test]
fn test_qvoice_v4_rejects_invalid_files() {
    let good = qvoice_v4_bytes("v", 2050, "text", 2);
    assert!(read_qvoice_bytes("good", &good).is_ok());

    // v3 (y cualquier versión distinta de 4) se rechaza con un mensaje que pide clonar de nuevo
    let mut v3 = Vec::new();
    v3.extend_from_slice(b"QVCE");
    v3.extend_from_slice(&3u32.to_le_bytes());
    v3.extend_from_slice(&1024u32.to_le_bytes());
    let err = read_qvoice_bytes("v3", &v3).unwrap_err().to_string();
    assert!(err.contains("version 3") && err.contains("version 4 required"), "got: {err}");
    assert!(err.contains("qwen_core_qv4_v3.qvoice"), "the message must name the path: {err}");

    // magic incorrecto
    let mut bad_magic = good.clone();
    bad_magic[0] = b'X';
    let err = read_qvoice_bytes("magic", &bad_magic).unwrap_err().to_string();
    assert!(err.contains("bad magic"), "got: {err}");

    // enc_dim incorrecto
    let mut bad_dim = good.clone();
    bad_dim[8..12].copy_from_slice(&512u32.to_le_bytes());
    assert!(read_qvoice_bytes("dim", &bad_dim).is_err());

    // truncado (en cada punto) y bytes sobrantes
    for cut in [0usize, 3, 11, 100, 4096, good.len() - 1] {
        let err = read_qvoice_bytes("trunc", &good[..cut]).unwrap_err().to_string();
        assert!(err.contains("truncated") || err.contains("bad magic"), "cut {cut}: {err}");
    }
    let mut extra = good.clone();
    extra.push(0);
    let err = read_qvoice_bytes("extra", &extra).unwrap_err().to_string();
    assert!(err.contains("trailing bytes"), "got: {err}");

    // tamaños excesivos (sin reservar memoria)
    let mut huge_frames = qvoice_v4_bytes("v", 0, "", 0);
    let n = huge_frames.len();
    huge_frames[n - 4..].copy_from_slice(&1_000_001u32.to_le_bytes());
    assert!(read_qvoice_bytes("huge_frames", &huge_frames).is_err());
    let mut huge_name = qvoice_v4_bytes("v", 0, "", 0);
    huge_name[4108..4112].copy_from_slice(&257u32.to_le_bytes());
    assert!(read_qvoice_bytes("huge_name", &huge_name).is_err());

    // UTF-8 inválido en el nombre
    let mut bad_utf8 = qvoice_v4_bytes("v", 0, "", 0);
    bad_utf8[4112] = 0xFF;
    assert!(read_qvoice_bytes("utf8", &bad_utf8).is_err());

    // lang_id inválido
    assert!(read_qvoice_bytes("lang", &qvoice_v4_bytes("v", 9999, "", 0)).is_err());

    // ICL incoherente: texto sin códigos, o códigos sin texto
    assert!(read_qvoice_bytes("text_only", &qvoice_v4_bytes("v", 0, "text", 0)).is_err());
    assert!(read_qvoice_bytes("codes_only", &qvoice_v4_bytes("v", 0, "", 3)).is_err());

    // El escritor tampoco emite ICL incoherente
    let incoherent = QVoiceFile {
        speaker_embedding: qvoice_emb(),
        voice_name: String::new(),
        lang_id: 0,
        ref_text: "text".to_string(),
        ref_codes: Vec::new(),
    };
    let p = std::env::temp_dir().join("qwen_core_incoherent.qvoice");
    assert!(incoherent.write_to_file(&p).is_err());
    let _ = std::fs::remove_file(&p);
}

#[test]
fn test_markup_sampling_and_lookups() {
    assert_eq!(lookup_language("Italian"), Some(2070));
    assert!(lookup_language("Klingon").is_none());

    let sub = MarkupPlanner::paralinguistic_substitute("Oh! [laugh] Great!");
    assert!(sub.did_substitute);
    assert_eq!(sub.seed, 2024);
    assert!((sub.temperature - 1.1).abs() < 1e-6);
    assert!(sub.text.contains("哈哈哈"));

    let spans = MarkupPlanner::parse_spans("Che bella notizia! [pause:400ms] Devo andare... Ciao.");
    assert_eq!(spans.len(), 3);
    assert_eq!(
        spans[1],
        SpanSpec::Pause {
            seconds: 0.4
        }
    );

    // Una etiqueta que ya no es de marcado (`[sad]`) queda como texto, igual que `[xyz]`
    for tag in ["[sad]", "[neutral]", "[xyz]"] {
        let text = format!("Devo andare {tag} Ciao.");
        assert!(!MarkupPlanner::has_markup(&text), "{tag} must not count as markup");
        let spans = MarkupPlanner::parse_spans(&text);
        assert_eq!(spans.len(), 1);
        let SpanSpec::Utterance { text: got, .. } = &spans[0] else {
            panic!("expected an utterance");
        };
        assert!(got.contains(tag), "{tag} must stay in the text: {got}");
    }
    // `[laugh]` (onomatopeya) y `|` siguen funcionando
    assert!(MarkupPlanner::has_markup("Oh [laugh] great"));
    assert!(qwen_core::is_paralinguistic_event_tag("sigh"));

    let mut sampler = Sampler::new();
    let mut rng1 = LcgRng::new(42);
    let mut rng2 = LcgRng::new(42);
    let mut l1 = vec![0.1f32, 2.0, -0.5, 1.5, 0.3];
    let mut l2 = l1.clone();
    let s1 = sampler.sample(&mut l1, 0.9, 50, 1.0, 1.05, &[1], &mut rng1);
    let s2 = sampler.sample(&mut l2, 0.9, 50, 1.0, 1.05, &[1], &mut rng2);
    assert_eq!(s1, s2);
    assert_eq!(rng1.seed, rng2.seed);

    // Muestreo nucleus (top_p < 1.0) determinista y sin bloqueo O(N^2)
    let mut lp1 = vec![0.1f32, 3.0, 2.5, -1.0, 0.2];
    let mut lp2 = lp1.clone();
    let mut rng_p1 = LcgRng::new(123);
    let mut rng_p2 = LcgRng::new(123);
    let sp1 = sampler.sample(&mut lp1, 0.8, 3, 0.6, 1.0, &[], &mut rng_p1);
    let sp2 = sampler.sample(&mut lp2, 0.8, 3, 0.6, 1.0, &[], &mut rng_p2);
    assert_eq!(sp1, sp2);
    assert!(sp1 == 1 || sp1 == 2);

    // Manejo seguro de NaN en logits sin devolver índice inválido ni propagar NaN
    let mut nan_logits = vec![f32::NAN, 1.0, 2.0, -3.0];
    let g_idx = sampler.sample(&mut nan_logits, 0.0, 1, 1.0, 1.0, &[], &mut rng1);
    assert_eq!(g_idx, 2);
    let mut nan_softmax = vec![f32::NAN; 4];
    let s_idx = sampler.sample(&mut nan_softmax, 0.9, 50, 1.0, 1.0, &[], &mut rng1);
    assert!(s_idx < 4);
}

#[test]
fn test_markup_pauses_and_pipe_separator() {
    // Pausas negativas o no finitas se sanean a 0.0
    let p_spans = MarkupPlanner::parse_spans("A [pause:-5] [pause:NaN] B");
    assert_eq!(p_spans[1], SpanSpec::Pause { seconds: 0.0 });
    assert_eq!(p_spans[2], SpanSpec::Pause { seconds: 0.0 });

    // Separador '|' detectado por has_markup y dividido en spans
    assert!(MarkupPlanner::has_markup("Hola. | Adiós."));
    let pipe_spans = MarkupPlanner::parse_spans("Hola. | Adiós.");
    assert_eq!(pipe_spans.len(), 2);
}
