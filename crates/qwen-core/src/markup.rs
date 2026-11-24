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
    /// Composición si el texto lleva marcado.
    Auto,
    /// `--no-compose`: el texto pasa tal cual y los tags se leen como texto.
    Literal,
}

/// Guion a sintetizar: un texto plano o una lista de tramos.
#[derive(Debug, Clone, PartialEq)]
pub enum Script {
    Plain(String),
    Spans(Vec<SpanSpec>),
}

/// Único sitio que decide los tramos. También fija el punto final: al texto plano o al último tramo hablado.
pub fn plan_script(text: &str, mode: MarkupMode) -> Script {
    match mode {
        MarkupMode::Literal => Script::Plain(close_final_sentence(text)),
        MarkupMode::Auto => {
            if MarkupPlanner::has_markup(text) {
                let mut spans = MarkupPlanner::parse_spans(text);
                // Solo el último tramo hablado cierra la frase; las pausas finales se ignoran.
                if let Some(SpanSpec::Utterance { text, .. }) = spans
                    .iter_mut()
                    .rev()
                    .find(|s| matches!(s, SpanSpec::Utterance { .. }))
                {
                    *text = close_final_sentence(text);
                }
                Script::Spans(spans)
            } else {
                Script::Plain(close_final_sentence(text))
            }
        }
    }
}

/// Cierra el texto con un punto final: el modelo cierra la entonación de una frase terminada
/// cuando el texto acaba en puntuación final. Se ignoran los espacios y los signos de cierre
/// (`"`, `'`, `»`, `”`, `’`, `)`) del final. Si el último carácter restante es puntuación final
/// (`.`, `!`, `?`, `…`, `。`, `！`, `？`) o no hay ninguno, el texto no cambia; si es `,`, `;` o `:`,
/// se sustituye por `.`; en otro caso se inserta `.` tras el último carácter que no es espacio,
/// conservando los espacios finales.
fn close_final_sentence(text: &str) -> String {
    const CLOSING: [char; 6] = ['"', '\'', '»', '”', '’', ')'];
    let Some((idx, last)) = text
        .char_indices()
        .rev()
        .find(|&(_, c)| !c.is_whitespace() && !CLOSING.contains(&c))
    else {
        return text.to_string();
    };
    match last {
        '.' | '!' | '?' | '…' | '。' | '！' | '？' => text.to_string(),
        ',' | ';' | ':' => {
            let mut out = String::with_capacity(text.len());
            out.push_str(&text[..idx]);
            out.push('.');
            out.push_str(&text[idx + last.len_utf8()..]);
            out
        }
        _ => {
            let end = text.trim_end().len();
            format!("{}.{}", &text[..end], &text[end..])
        }
    }
}

/// Resuelve el muestreo final: lo explícito del usuario y, si no, el valor de `base`. `Clock` se resuelve con `time_based_seed()`.
pub fn resolve_sampling(
    base: SamplingParams,
    seed: SeedChoice,
    temperature: Option<f32>,
) -> SamplingParams {
    let seed = match seed {
        SeedChoice::Fixed(s) => s,
        SeedChoice::Clock => time_based_seed(),
        SeedChoice::Default => base.seed,
    };
    let temperature = temperature.unwrap_or(base.temperature);
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

pub struct MarkupPlanner;

impl MarkupPlanner {
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

    /// Parsea el marcado en línea en `Vec<SpanSpec>`.
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
        assert_eq!(lit, Script::Plain("Hola [laugh] mundo.".to_string()));

        // Un tag desconocido se lee como texto, igual que en `Literal`.
        let auto = plan_script("Hola [laugh] mundo.", MarkupMode::Auto);
        assert_eq!(auto, Script::Plain("Hola [laugh] mundo.".to_string()));

        let plain = plan_script("Hola.", MarkupMode::Auto);
        assert_eq!(plain, Script::Plain("Hola.".to_string()));

        let pause = plan_script("Hola. [pause:0.4] Adiós.", MarkupMode::Auto);
        assert!(matches!(pause, Script::Spans(_)));

        // `[hmm]` y `[uhm]` son muletillas; `[pause]` es una pausa.
        let mix = plan_script("Uno [hmm] dos [uhm] tres [pause:0.2]", MarkupMode::Auto);
        let Script::Spans(spans) = mix else {
            panic!("se esperaba Spans");
        };
        assert!(spans
            .iter()
            .any(|s| matches!(s, SpanSpec::Utterance { is_filler: true, .. })));
        assert!(spans.iter().any(|s| matches!(s, SpanSpec::Pause { .. })));
    }

    #[test]
    fn close_final_sentence_cases() {
        let changed = [
            ("Hola mundo", "Hola mundo."),
            ("Tengo 3", "Tengo 3."),
            ("Hola mundo ", "Hola mundo. "),
            ("Hola, \n", "Hola. \n"),
            ("Hola mundo,", "Hola mundo."),
            ("Hola;", "Hola."),
            ("Nota:", "Nota."),
            ("Dijo «hola»", "Dijo «hola»."),
        ];
        for (input, expected) in changed {
            assert_eq!(close_final_sentence(input), expected, "entrada {input:?}");
        }
        let unchanged = [
            "Hola.",
            "¿Vienes?",
            "¡Hola!",
            "Hmmm...",
            "Bien…",
            "你好。",
            "Dijo «hola».",
            "(ok.)",
            "Hola mundo.  ",
            "",
            "   ",
            ")",
        ];
        for input in unchanged {
            assert_eq!(close_final_sentence(input), input, "entrada {input:?}");
        }
    }

    #[test]
    fn plan_script_closes_final_sentence() {
        assert_eq!(
            plan_script("Hola [laugh] mundo", MarkupMode::Literal),
            Script::Plain("Hola [laugh] mundo.".to_string())
        );
        assert_eq!(
            plan_script("Hola mundo", MarkupMode::Auto),
            Script::Plain("Hola mundo.".to_string())
        );

        // Solo el último tramo hablado cambia; la muletilla y las pausas quedan igual.
        let Script::Spans(spans) = plan_script("Uno [hmm] dos [pause:0.2]", MarkupMode::Auto)
        else {
            panic!("se esperaba Spans");
        };
        let texts: Vec<(&str, bool)> = spans
            .iter()
            .filter_map(|s| match s {
                SpanSpec::Utterance {
                    text, is_filler, ..
                } => Some((text.as_str(), *is_filler)),
                SpanSpec::Pause { .. } => None,
            })
            .collect();
        assert_eq!(texts.len(), 3, "tramos hablados: {texts:?}");
        assert_eq!(texts[0], ("Uno", false));
        assert!(texts[1].1, "el segundo tramo debe ser la muletilla");
        assert_eq!(texts[2], ("dos.", false));
        assert!(matches!(spans.last(), Some(SpanSpec::Pause { .. })));

        // Un último tramo que ya termina en `!` no cambia.
        let Script::Spans(spans) = plan_script("Uno [pause:0.2] dos!", MarkupMode::Auto) else {
            panic!("se esperaba Spans");
        };
        assert!(spans
            .iter()
            .any(|s| matches!(s, SpanSpec::Utterance { text, .. } if text == "dos!")));

        // Sin tramos hablados no hay nada que cerrar.
        let Script::Spans(spans) = plan_script("[pause:0.3]", MarkupMode::Auto) else {
            panic!("se esperaba Spans");
        };
        assert!(spans.iter().all(|s| matches!(s, SpanSpec::Pause { .. })));
    }

    #[test]
    fn resolve_sampling_precedence() {
        let base = SamplingParams::default();

        let d = resolve_sampling(base, SeedChoice::Default, None);
        assert_eq!((d.seed, d.temperature), (DEFAULT_SEED, DEFAULT_TEMPERATURE));

        let u = resolve_sampling(base, SeedChoice::Fixed(9), Some(0.5));
        assert_eq!((u.seed, u.temperature), (9, 0.5));

        // `Clock` se resuelve con la hora; el resto de campos viene de `base`.
        let c = resolve_sampling(base, SeedChoice::Clock, None);
        assert_eq!(c.top_k, base.top_k);
    }
}
