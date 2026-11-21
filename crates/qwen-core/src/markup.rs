// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use crate::sampling::{time_based_seed, SamplingParams, SeedChoice};

#[derive(Debug, Clone, PartialEq)]
pub enum SpanSpec {
    Utterance {
        text: String,
        is_filler: bool,
        rate: f32,
        volume: f32,
    },
    Pause {
        seconds: f32,
    },
}

/// Cómo se interpreta el marcado del texto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkupMode {
    /// Sustitución paralingüística y composición si queda marcado.
    Auto,
    /// `--no-compose`: el texto pasa tal cual y los tags se leen como texto.
    Literal,
    /// `--compose`: el texto es una especificación de tramos.
    Compose,
}

/// Guion a sintetizar: un texto plano o una lista de tramos.
#[derive(Debug, Clone, PartialEq)]
pub enum Script {
    Plain(String),
    Spans(Vec<SpanSpec>),
}

/// Resultado de planificar el marcado.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedScript {
    pub script: Script,
    /// Semilla y temperatura del primer tag paralingüístico sustituido, si hubo.
    pub paralinguistic: Option<(u32, f32)>,
    /// Texto tras la sustitución paralingüística (solo para logs).
    pub text: String,
}

/// Único sitio que decide la sustitución paralingüística y los tramos.
pub fn plan_script(text: &str, mode: MarkupMode) -> PlannedScript {
    match mode {
        MarkupMode::Literal => PlannedScript {
            script: Script::Plain(text.to_string()),
            paralinguistic: None,
            text: text.to_string(),
        },
        // `--compose`: igual que el texto normal (sustitución paralingüística), pero siempre
        // como lista de tramos.
        MarkupMode::Compose => {
            let sub = MarkupPlanner::paralinguistic_substitute(text);
            let paralinguistic = sub.did_substitute.then_some((sub.seed, sub.temperature));
            PlannedScript {
                script: Script::Spans(MarkupPlanner::parse_spans(&sub.text)),
                paralinguistic,
                text: sub.text,
            }
        }
        MarkupMode::Auto => {
            let sub = MarkupPlanner::paralinguistic_substitute(text);
            let paralinguistic = sub.did_substitute.then_some((sub.seed, sub.temperature));
            let script = if MarkupPlanner::has_markup(&sub.text) {
                Script::Spans(MarkupPlanner::parse_spans(&sub.text))
            } else {
                Script::Plain(sub.text.clone())
            };
            PlannedScript {
                script,
                paralinguistic,
                text: sub.text,
            }
        }
    }
}

/// Resuelve el muestreo final. Precedencia: lo explícito del usuario, después el tag
/// paralingüístico y por último el valor de `base`. `Clock` se resuelve con `time_based_seed()`.
pub fn resolve_sampling(
    base: SamplingParams,
    seed: SeedChoice,
    temperature: Option<f32>,
    paralinguistic: Option<(u32, f32)>,
) -> SamplingParams {
    let seed = match seed {
        SeedChoice::Fixed(s) => s,
        SeedChoice::Clock => time_based_seed(),
        SeedChoice::Default => paralinguistic.map_or(base.seed, |(s, _)| s),
    };
    let temperature =
        temperature.unwrap_or_else(|| paralinguistic.map_or(base.temperature, |(_, t)| t));
    SamplingParams {
        temperature,
        seed,
        ..base
    }
}

struct ComposeMacro {
    tag: &'static str,
    text: &'static str,
    rate: f32,
    volume: f32,
}

const COMPOSE_MACROS: &[ComposeMacro] = &[
    ComposeMacro { tag: "ahh", text: "Haaa...", rate: 0.90, volume: 0.70 },
    ComposeMacro { tag: "relief", text: "Haaa...", rate: 0.90, volume: 0.70 },
    ComposeMacro { tag: "phew", text: "Uao...", rate: 1.00, volume: 0.82 },
    ComposeMacro { tag: "hmm", text: "Hmmm...", rate: 0.88, volume: 0.65 },
    ComposeMacro { tag: "mmm", text: "\u{55ef}", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "hmpf", text: "Hmpf...", rate: 1.00, volume: 0.75 },
    ComposeMacro { tag: "mah", text: "Mah...", rate: 0.95, volume: 0.78 },
    ComposeMacro { tag: "uhm", text: "Uhm...", rate: 0.95, volume: 0.72 },
    ComposeMacro { tag: "haha", text: "Haha!", rate: 1.00, volume: 0.80 },
    ComposeMacro { tag: "heh", text: "Hehhh...", rate: 0.95, volume: 0.70 },
    ComposeMacro { tag: "ouch", text: "Ouch!", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "ahi", text: "Ahi!", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "huff", text: "Uff...", rate: 1.00, volume: 0.78 },
    ComposeMacro { tag: "ugh", text: "Ugh...", rate: 1.00, volume: 0.78 },
];

fn pick_paralinguistic(tag: &str) -> Option<(&'static str, u32, f32)> {
    match tag.to_ascii_lowercase().as_str() {
        "laugh" | "laughs" => Some(("\u{54c8}\u{54c8}\u{54c8}", 2024, 1.1)),
        "sigh" | "sighs" => Some(("\u{5509}", 42, 1.1)),
        "yawn" | "yawns" => Some(("\u{54c8}\u{554a}", 42, 1.1)),
        "wow" => Some(("\u{54c7}", 2024, 1.1)),
        "giggle" | "giggles" => Some(("\u{563f}\u{563f}", 42, 1.1)),
        "scoff" => Some(("\u{5207}", 42, 1.0)),
        _ => None,
    }
}

pub fn is_paralinguistic_event_tag(tag: &str) -> bool {
    pick_paralinguistic(tag).is_some()
}

pub struct ParalinguisticSubstitution {
    pub text: String,
    pub did_substitute: bool,
    pub seed: u32,
    pub temperature: f32,
}

pub struct MarkupPlanner;

impl MarkupPlanner {
    /// Sustitución paralingüística en línea.
    /// Reemplaza `[laugh]`, `[sigh]`, etc. por onomatopeyas delimitadas por comas
    /// en una única generación, fijando la semilla y temperatura del primer tag.
    pub fn paralinguistic_substitute(text: &str) -> ParalinguisticSubstitution {
        let mut out = String::with_capacity(text.len() + 48);
        let mut did = false;
        let mut seed = 7u32;
        let mut temp = 1.1f32;
        let bytes = text.as_bytes();
        let mut p = 0usize;

        while p < bytes.len() {
            if bytes[p] == b'[' {
                if let Some(rel_close) = bytes[p..].iter().position(|&b| b == b']') {
                    let close = p + rel_close;
                    let tl = close - p - 1;
                    if tl < 32 {
                        if let Ok(raw_tag) = std::str::from_utf8(&bytes[p + 1..close]) {
                            let tag = raw_tag.trim();
                            if let Some((onom, sd, td)) = pick_paralinguistic(tag) {
                                while out.ends_with(' ') {
                                    out.pop();
                                }
                                if out.ends_with(',') {
                                    out.pop();
                                }
                                if !out.is_empty() {
                                    out.push_str(", ");
                                }
                                out.push_str(onom);
                                out.push_str(", ");
                                if !did {
                                    seed = sd;
                                    temp = td;
                                }
                                did = true;
                                p = close + 1;
                                while p < bytes.len() && bytes[p] == b' ' {
                                    p += 1;
                                }
                                if p < bytes.len() && bytes[p] == b',' {
                                    p += 1;
                                }
                                while p < bytes.len() && bytes[p] == b' ' {
                                    p += 1;
                                }
                                continue;
                            }
                        }
                    }
                }
            }
            let Some(ch) = text[p..].chars().next() else {
                break;
            };
            out.push(ch);
            p += ch.len_utf8();
        }

        ParalinguisticSubstitution {
            text: out,
            did_substitute: did,
            seed,
            temperature: temp,
        }
    }

    pub fn has_markup(text: &str) -> bool {
        if text.contains('|') {
            return true;
        }
        let mut search = text;
        while let Some(open) = search.find('[') {
            let after = &search[open + 1..];
            if let Some(close) = after.find(']') {
                if close < 64 {
                    let t = after[..close].trim();
                    let tl = t.to_ascii_lowercase();
                    if tl.starts_with("pause") || tl.starts_with("break") {
                        return true;
                    }
                    if t.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
                        return true;
                    }
                    if is_paralinguistic_event_tag(t) {
                        return true;
                    }
                    if COMPOSE_MACROS.iter().any(|m| m.tag == tl) {
                        return true;
                    }
                }
            }
            search = &search[open + 1..];
        }
        false
    }

    fn parse_duration_s(s: &str) -> f32 {
        let s = s.trim_start();
        let mut end = 0usize;
        for (idx, b) in s.bytes().enumerate() {
            if b.is_ascii_digit() || matches!(b, b'.' | b'+' | b'-') {
                end = idx + 1;
            } else {
                break;
            }
        }
        let v = s[..end].parse::<f32>().unwrap_or(0.0);
        if !v.is_finite() || v <= 0.0 {
            return 0.0;
        }
        let unit = s[end..].trim_start();
        if unit.len() >= 2 && unit[..2].eq_ignore_ascii_case("ms") {
            v / 1000.0
        } else {
            v
        }
    }

    /// Parsea marcado en línea / `--compose` en `Vec<SpanSpec>`.
    /// Los tags desconocidos (por ejemplo `[xyz]`) quedan como texto.
    pub fn parse_spans(input: &str) -> Vec<SpanSpec> {
        let mut spans = Vec::new();
        let mut seg = String::new();

        let flush = |seg: &mut String, spans: &mut Vec<SpanSpec>| {
            let trimmed = seg.trim_matches(|c| matches!(c, ' ' | '\t' | '\n'));
            if !trimmed.is_empty() {
                spans.push(SpanSpec::Utterance {
                    text: trimmed.to_string(),
                    is_filler: false,
                    rate: 0.0,
                    volume: 0.0,
                });
            }
            seg.clear();
        };

        let bytes = input.as_bytes();
        let mut p = 0usize;
        while p < bytes.len() {
            if bytes[p] == b'|' {
                flush(&mut seg, &mut spans);
                p += 1;
                continue;
            }
            if bytes[p] == b'[' {
                if let Some(rel_close) = bytes[p..].iter().position(|&b| b == b']') {
                    let close = p + rel_close;
                    let tl = close - p - 1;
                    if tl < 64 {
                        if let Ok(raw_tag) = std::str::from_utf8(&bytes[p + 1..close]) {
                            let t = raw_tag.trim();
                            let tlow = t.to_ascii_lowercase();
                            let mut handled = false;
                            if tlow.starts_with("pause") || tlow.starts_with("break") {
                                let num = if let Some(idx) = t.find(':') {
                                    &t[idx + 1..]
                                } else if let Some(idx) = t.find('=') {
                                    &t[idx + 1..]
                                } else {
                                    t.get(5..).unwrap_or("")
                                };
                                flush(&mut seg, &mut spans);
                                spans.push(SpanSpec::Pause {
                                    seconds: Self::parse_duration_s(num),
                                });
                                handled = true;
                            } else if t.starts_with(|c: char| c.is_ascii_digit() || c == '.') {
                                flush(&mut seg, &mut spans);
                                spans.push(SpanSpec::Pause {
                                    seconds: Self::parse_duration_s(t),
                                });
                                handled = true;
                            } else if let Some(m) = COMPOSE_MACROS.iter().find(|m| m.tag == tlow) {
                                flush(&mut seg, &mut spans);
                                spans.push(SpanSpec::Utterance {
                                    text: m.text.to_string(),
                                    is_filler: true,
                                    rate: m.rate,
                                    volume: m.volume,
                                });
                                handled = true;
                            }
                            if handled {
                                p = close + 1;
                                continue;
                            }
                        }
                    }
                }
            }
            let Some(ch) = input[p..].chars().next() else {
                break;
            };
            seg.push(ch);
            p += ch.len_utf8();
        }
        flush(&mut seg, &mut spans);
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DEFAULT_SEED, DEFAULT_TEMPERATURE};

    #[test]
    fn plan_script_modes() {
        let lit = plan_script("Hola [laugh] mundo.", MarkupMode::Literal);
        assert_eq!(lit.script, Script::Plain("Hola [laugh] mundo.".to_string()));
        assert!(lit.paralinguistic.is_none());

        let auto = plan_script("Hola [laugh] mundo.", MarkupMode::Auto);
        assert_eq!(auto.paralinguistic, Some((2024, 1.1)));
        assert!(matches!(auto.script, Script::Plain(_)));

        let plain = plan_script("Hola.", MarkupMode::Auto);
        assert!(plain.paralinguistic.is_none());
        assert_eq!(plain.script, Script::Plain("Hola.".to_string()));

        let pause = plan_script("Hola. [pause:0.4] Adiós.", MarkupMode::Auto);
        assert!(matches!(pause.script, Script::Spans(_)));

        // `Compose` aplica la paralingüística como el texto normal: `[laugh]` es onomatopeya.
        let comp = plan_script("Hola. [laugh] Adiós.", MarkupMode::Compose);
        assert_eq!(comp.paralinguistic, Some((2024, 1.1)));
        assert_eq!(
            comp.text,
            plan_script("Hola. [laugh] Adiós.", MarkupMode::Auto).text
        );
        let Script::Spans(spans) = comp.script else {
            panic!("se esperaba Spans");
        };
        assert_eq!(spans.len(), 1);
        let SpanSpec::Utterance { text, is_filler, .. } = &spans[0] else {
            panic!("se esperaba Utterance");
        };
        assert!(!is_filler);
        assert!(text.contains('\u{54c8}'), "se obtuvo: {text}");

        // `[hmm]` sigue siendo muletilla; `[yawn]` se sustituye; `[pause]` es una pausa.
        let mix = plan_script("Uno [hmm] dos [yawn] tres [pause:0.2]", MarkupMode::Compose);
        assert_eq!(mix.paralinguistic, Some((42, 1.1)));
        let Script::Spans(spans) = mix.script else {
            panic!("se esperaba Spans");
        };
        assert!(spans
            .iter()
            .any(|s| matches!(s, SpanSpec::Utterance { is_filler: true, .. })));
        assert!(spans.iter().any(|s| matches!(s, SpanSpec::Pause { .. })));
    }

    #[test]
    fn resolve_sampling_precedence() {
        let base = SamplingParams::default();
        let paralinguistic = Some((2024, 1.1));

        let d = resolve_sampling(base, SeedChoice::Default, None, None);
        assert_eq!((d.seed, d.temperature), (DEFAULT_SEED, DEFAULT_TEMPERATURE));

        let p = resolve_sampling(base, SeedChoice::Default, None, paralinguistic);
        assert_eq!((p.seed, p.temperature), (2024, 1.1));

        let u = resolve_sampling(base, SeedChoice::Fixed(9), Some(0.5), paralinguistic);
        assert_eq!((u.seed, u.temperature), (9, 0.5));

        let mixed = resolve_sampling(base, SeedChoice::Fixed(9), None, paralinguistic);
        assert_eq!((mixed.seed, mixed.temperature), (9, 1.1));

        // `Clock` manda sobre el tag; el resto de campos viene de `base`.
        let c = resolve_sampling(base, SeedChoice::Clock, None, paralinguistic);
        assert_eq!(c.top_k, base.top_k);
        assert_eq!(c.temperature, 1.1);
    }
}
