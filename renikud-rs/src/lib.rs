use std::collections::HashMap;
use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

const ALEF: u32 = 0x05D0;
const TAF: u32 = 0x05EA;
const STRESS: &str = "ˈ";

fn is_hebrew(c: char) -> bool {
    let cp = c as u32;
    cp >= ALEF && cp <= TAF
}

pub struct G2P {
    session: Session,
    vocab: HashMap<char, i64>,
    consonant_vocab: HashMap<i64, String>,
    vowel_vocab: HashMap<i64, String>,
    cls_id: i64,
    sep_id: i64,
}

impl G2P {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let (vocab_json, consonant_vocab_json, vowel_vocab_json, cls_id, sep_id) = {
            let meta = session.metadata()?;
            let vocab_json = meta.custom("vocab").ok_or_else(|| anyhow::anyhow!("missing vocab"))?;
            let consonant_vocab_json = meta.custom("consonant_vocab").ok_or_else(|| anyhow::anyhow!("missing consonant_vocab"))?;
            let vowel_vocab_json = meta.custom("vowel_vocab").ok_or_else(|| anyhow::anyhow!("missing vowel_vocab"))?;
            let cls_id: i64 = meta.custom("cls_token_id").ok_or_else(|| anyhow::anyhow!("missing cls_token_id"))?.parse()?;
            let sep_id: i64 = meta.custom("sep_token_id").ok_or_else(|| anyhow::anyhow!("missing sep_token_id"))?.parse()?;
            (vocab_json, consonant_vocab_json, vowel_vocab_json, cls_id, sep_id)
        };

        let raw_vocab: HashMap<String, i64> = serde_json::from_str(&vocab_json)?;
        let vocab: HashMap<char, i64> = raw_vocab
            .into_iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
            .collect();

        let raw_consonants: HashMap<String, String> = serde_json::from_str(&consonant_vocab_json)?;
        let consonant_vocab: HashMap<i64, String> = raw_consonants
            .into_iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
            .collect();

        let raw_vowels: HashMap<String, String> = serde_json::from_str(&vowel_vocab_json)?;
        let vowel_vocab: HashMap<i64, String> = raw_vowels
            .into_iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
            .collect();

        Ok(Self { session, vocab, consonant_vocab, vowel_vocab, cls_id, sep_id })
    }

    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>, Vec<(usize, usize)>) {
        let normalized: String = text.nfd().collect();
        let unk_id = 0i64;
        let mut ids = vec![self.cls_id];
        let mut offsets = vec![(0usize, 0usize)]; // CLS
        for (i, c) in normalized.char_indices() {
            ids.push(*self.vocab.get(&c).unwrap_or(&unk_id));
            offsets.push((i, i + c.len_utf8()));
        }
        ids.push(self.sep_id);
        offsets.push((0, 0)); // SEP
        let mask = vec![1i64; ids.len()];
        (ids, mask, offsets)
    }

    pub fn phonemize(&mut self, text: &str) -> anyhow::Result<String> {
        let normalized: String = text.nfd().collect();
        let (ids, mask, offsets) = self.tokenize(text);
        let len = ids.len();

        let input_ids = Tensor::<i64>::from_array(([1, len], ids.into_boxed_slice()))?;
        let attention_mask = Tensor::<i64>::from_array(([1, len], mask.into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])?;

        let (cons_shape, cons_data) = outputs["consonant_logits"].try_extract_tensor::<f16>()?;
        let (vowel_shape, vowel_data) = outputs["vowel_logits"].try_extract_tensor::<f16>()?;
        let (_, stress_data) = outputs["stress_logits"].try_extract_tensor::<f16>()?;

        let num_consonants = cons_shape[2] as usize;
        let num_vowels = vowel_shape[2] as usize;

        let argmax = |data: &[f16], offset: usize, size: usize| -> i64 {
            data[offset..offset + size]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0)
        };

        let mut result = String::new();
        let mut prev_end = 0usize;
        for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
            // Only process single-char tokens
            let char_len = end.saturating_sub(start);
            if char_len == 0 {
                continue;
            }

            // Pass through skipped bytes
            if start > prev_end {
                result.push_str(&normalized[prev_end..start]);
            }

            let c = normalized[start..end].chars().next().unwrap();
            prev_end = end;

            if !is_hebrew(c) {
                result.push(c);
                continue;
            }

            let consonant_id = argmax(&cons_data, tok_idx * num_consonants, num_consonants);
            let vowel_id = argmax(&vowel_data, tok_idx * num_vowels, num_vowels);
            let stress_id = argmax(&stress_data, tok_idx * 2, 2);

            let consonant = self.consonant_vocab.get(&consonant_id).map(String::as_str).unwrap_or("∅");
            let vowel = self.vowel_vocab.get(&vowel_id).map(String::as_str).unwrap_or("∅");
            let stressed = stress_id == 1;

            if consonant != "∅" {
                result.push_str(consonant);
            }
            if stressed {
                result.push_str(STRESS);
            }
            if vowel != "∅" {
                result.push_str(vowel);
            }
        }

        if prev_end < normalized.len() {
            result.push_str(&normalized[prev_end..]);
        }

        drop(outputs);
        Ok(result)
    }
}
