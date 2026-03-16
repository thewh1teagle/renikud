use std::collections::HashMap;
use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

pub struct G2P {
    session: Session,
    vocab: HashMap<char, i64>,
    ipa_vocab: HashMap<i64, String>,
    cls_id: i64,
    sep_id: i64,
}

impl G2P {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let (vocab_json, ipa_vocab_json, cls_id, sep_id) = {
            let meta = session.metadata()?;
            let vocab_json = meta.custom("vocab").ok_or_else(|| anyhow::anyhow!("missing vocab"))?;
            let ipa_vocab_json = meta.custom("ipa_vocab").ok_or_else(|| anyhow::anyhow!("missing ipa_vocab"))?;
            let cls_id: i64 = meta.custom("cls_token_id").ok_or_else(|| anyhow::anyhow!("missing cls_token_id"))?.parse()?;
            let sep_id: i64 = meta.custom("sep_token_id").ok_or_else(|| anyhow::anyhow!("missing sep_token_id"))?.parse()?;
            (vocab_json, ipa_vocab_json, cls_id, sep_id)
        };

        let raw_vocab: HashMap<String, i64> = serde_json::from_str(&vocab_json)?;
        let vocab: HashMap<char, i64> = raw_vocab
            .into_iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
            .collect();

        let raw_ipa: HashMap<String, String> = serde_json::from_str(&ipa_vocab_json)?;
        let ipa_vocab: HashMap<i64, String> = raw_ipa
            .into_iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
            .collect();

        Ok(Self { session, vocab, ipa_vocab, cls_id, sep_id })
    }

    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>) {
        let normalized: String = text.nfd().collect();
        let unk_id = 0i64;
        let mut ids = vec![self.cls_id];
        for c in normalized.chars() {
            ids.push(*self.vocab.get(&c).unwrap_or(&unk_id));
        }
        ids.push(self.sep_id);
        let mask = vec![1i64; ids.len()];
        (ids, mask)
    }

    fn decode(&self, token_ids: &[i64]) -> String {
        let mut result = String::new();
        let mut prev: Option<i64> = None;
        for &t in token_ids {
            if t == 0 {
                prev = None;
                continue;
            }
            if Some(t) != prev {
                if let Some(tok) = self.ipa_vocab.get(&t) {
                    if tok != "<pad>" && tok != "<unk>" && tok != "<blank>" {
                        result.push_str(tok);
                    }
                }
            }
            prev = Some(t);
        }
        result
    }

    pub fn phonemize(&mut self, text: &str) -> anyhow::Result<String> {
        let (ids, mask) = self.tokenize(text);
        let len = ids.len();

        let input_ids = Tensor::<i64>::from_array(([1, len], ids.into_boxed_slice()))?;
        let attention_mask = Tensor::<i64>::from_array(([1, len], mask.into_boxed_slice()))?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids,
            "attention_mask" => attention_mask
        ])?;

        let (_, lengths_data) = outputs["input_lengths"].try_extract_tensor::<i64>()?;
        let length = lengths_data[0] as usize;

        let (logits_shape, logits_data) = outputs["logits"].try_extract_tensor::<f16>()?;
        let vocab_size = logits_shape[2] as usize;

        let pred_ids: Vec<i64> = (0..length)
            .map(|t| {
                let offset = t * vocab_size;
                let frame = &logits_data[offset..offset + vocab_size];
                frame
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as i64)
                    .unwrap_or(0)
            })
            .collect();

        drop(outputs);
        Ok(self.decode(&pred_ids))
    }
}
