// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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

struct ComposeMacro {
    tag: &'static str,
    text: &'static str,
    rate: f32,
    volume: f32,
}

const COMPOSE_MACROS: &[ComposeMacro] = &[
    ComposeMacro { tag: "sigh", text: "Hah...", rate: 1.12, volume: 0.67 },
    ComposeMacro { tag: "sighs", text: "Hah...", rate: 1.12, volume: 0.67 },
    ComposeMacro { tag: "ahh", text: "Haaa...", rate: 0.90, volume: 0.70 },
    ComposeMacro { tag: "relief", text: "Haaa...", rate: 0.90, volume: 0.70 },
    ComposeMacro { tag: "phew", text: "Uao...", rate: 1.00, volume: 0.82 },
    ComposeMacro { tag: "hmm", text: "Hmmm...", rate: 0.88, volume: 0.65 },
    ComposeMacro { tag: "mmm", text: "\u{55ef}", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "hmpf", text: "Hmpf...", rate: 1.00, volume: 0.75 },
    ComposeMacro { tag: "mah", text: "Mah...", rate: 0.95, volume: 0.78 },
    ComposeMacro { tag: "uhm", text: "Uhm...", rate: 0.95, volume: 0.72 },
    ComposeMacro { tag: "laugh", text: "Eheh...", rate: 0.95, volume: 0.78 },
    ComposeMacro { tag: "laughs", text: "Eheh...", rate: 0.95, volume: 0.78 },
    ComposeMacro { tag: "haha", text: "Haha!", rate: 1.00, volume: 0.80 },
    ComposeMacro { tag: "heh", text: "Hehhh...", rate: 0.95, volume: 0.70 },
    ComposeMacro { tag: "ouch", text: "Ouch!", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "ahi", text: "Ahi!", rate: 1.00, volume: 0.85 },
    ComposeMacro { tag: "huff", text: "Uff...", rate: 1.00, volume: 0.78 },
    ComposeMacro { tag: "ugh", text: "Ugh...", rate: 1.00, volume: 0.78 },
];

fn para_pick_06b(tag: &str) -> Option<(&'static str, u32, f32)> {
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

pub fn is_para_event_tag(tag: &str) -> bool {
    para_pick_06b(tag).is_some()
}

pub struct ParaSubResult {
    pub text: String,
    pub did_substitute: bool,
    pub seed: u32,
    pub temperature: f32,
}

pub struct MarkupPlanner;

impl MarkupPlanner {
    /// Sustitución paralingüística inline para `0.6B`.
    /// Reemplaza `[laugh]`, `[sigh]`, etc. por onomatopeyas delimitadas por comas
    /// en una única generación, fijando la semilla y temperatura de la primera etiqueta.
    pub fn para_substitute(text: &str) -> ParaSubResult {
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
                            if let Some((onom, sd, td)) = para_pick_06b(tag) {
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

        ParaSubResult {
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
                    if is_para_event_tag(t) {
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

    /// Parsea marcado inline / `--compose` en `Vec<SpanSpec>`.
    /// Las etiquetas desconocidas (por ejemplo `[xyz]`) quedan como texto.
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
