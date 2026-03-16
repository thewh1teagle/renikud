Here is the architectural design document in English. You can use this as your official blueprint, share it with collaborators, or use it to apply for compute grants.

---

# Architectural Design Document: Hebrew Grapheme-to-Phoneme (G2P) Foundation Model

## 1. Project Objective

To build a State-of-the-Art (SOTA), real-time neural model capable of converting raw Modern Hebrew text into accurate IPA (International Phonetic Alphabet) transcriptions. The model must handle the inherent complexities of Hebrew (unvoweled text, deep semantic ambiguity, modern slang, and loan words) while maintaining ultra-fast inference speeds (non-autoregressive) for production environments.

---

## 2. Dataset & Knowledge Distillation Strategy

Given the absence of a large-scale, open-source IPA dataset for Hebrew, we are utilizing a **Knowledge Distillation** approach to generate a high-quality, synthetic dataset efficiently:

* **The Baseline (Pre-training Data):** Utilizing existing rule-based/hybrid engines (e.g., `Phonikud`, which yields ~85% accuracy) to generate 1M-5M sentences. This provides the model with the basic phonetic mapping and statistical structure of the language at zero cost.
* **The Oracle / Teacher Model (Fine-tuning Data):** Leveraging a frontier LLM (Gemini 3.1 Pro / Gemini 1.5 Flash) via strict JSON structured outputs (`temperature=0.0`) to generate 50,000–100,000 "Gold Standard" sentences. This targets edge cases, complex morphology, and deep contextual disambiguation where baseline models fail.
* **Evaluation:** A curated benchmark set of 5,000 human/LLM-verified sentences to serve as the definitive test set.

---

## 3. Architectural Evolution & R&D

We evaluated several architectural paradigms before finalizing the pipeline, navigating significant technical constraints specific to Hebrew NLP:

### Phase 1: Training From Scratch

* **Hypothesis:** Build a custom Character-level Transformer or Conformer (~50M parameters) and train it with a CTC (Connectionist Temporal Classification) loss.
* **Conclusion:** Rejected. Training a model from scratch to understand Hebrew morphology requires immense data and weeks of GPU compute, making it highly inefficient compared to transfer learning.

### Phase 2: Standard DictaBERT + CTC

* **Hypothesis:** Attach a CTC head to a pre-trained `dicta-il/dictabert` (185M parameters).
* **The WordPiece Collision:** Standard LLMs use sub-word tokenization. The word "שלום" might be a single token. CTC mathematically requires the input length ($T$) to be $\ge$ the output length. Forcing 5 IPA phonemes out of 1 WordPiece token causes an immediate tensor shape crash. Swapping the tokenizer post-training destroys the model's pre-trained weights.

### Phase 3: The Breakthrough (DictaBERT-Large-Char)

* **Hypothesis:** Utilize Mila/Dicta's `dicta-il/dictabert-large-char-menaked` (300M parameters). Being a character-level model, it solves the WordPiece issue while retaining deep semantic understanding.
* **The "Abjad" Problem:** Hebrew is written with missing vowels. The word 'ספר' is 3 input characters, but its phonetic representation 'sefer' is 5 phonemes. This still violates the CTC length requirement ($T_{in} \ge T_{out}$).

---

## 4. The Final Architecture: Upsampled CTC Pipeline

To resolve the Abjad problem without resorting to a slow, autoregressive Attention Decoder, we implemented a structural manipulation technique.

**The Pipeline Flow:**

1. **Input:** Raw Hebrew text tokenized at the character level.
2. **Encoder:** Passed through `DictaBERT-large-char-menaked` (Output shape: `Batch, SeqLen, 1024`).
3. **Linear Projection:** Dimensionality reduction from `1024` to `256` to optimize GPU memory footprint during upsampling.
4. **The Magic Trick (2x Upsample):** Using `torch.repeat_interleave(2)`, we double the sequence length of both the hidden states and the attention mask (e.g., `ס פ ר` $\rightarrow$ `ס ס פ פ ר ר`). **This guarantees the input sequence is always longer than the IPA output, satisfying the CTC strict requirement.**
5. **CTC Head:** A linear classifier mapping the upsampled vectors to the IPA vocabulary logits.
6. **Inference:** Pure Greedy Decoding. The CTC algorithm collapses consecutive duplicates and blanks in $O(1)$ steps, providing ultra-fast, real-time generation.

*(Note: We deliberately stripped away the joint Attention Decoder used in standard academic papers. DictaBERT's 300M parameters provide more than enough contextual understanding, rendering the autoregressive decoder an unnecessary bottleneck for production inference).*

---

## 5. Training Infrastructure

The model will be trained using the **Hugging Face `Trainer` API** for maximal efficiency and stability:

* **Mixed Precision (FP16):** Mandatory for training a 300M parameter model to halve VRAM usage and double training speed.
* **Discriminative Learning Rates:**
* **High LR** ($\sim 1e^{-4}$): Applied to the newly initialized Projection and CTC layers to accelerate learning.
* **Low LR** ($\sim 2e^{-5}$): Applied to the `DictaBERT` encoder layers (Unfrozen) to gently fine-tune the phonetic mappings without triggering Catastrophic Forgetting of its pre-trained Hebrew semantics.


* **The Tokenizer Bug Fix:** Native `AutoTokenizer` breaks character-level Hugging Face models by defaulting to `BertTokenizer` logic. The pipeline explicitly bypasses this by downloading the raw `tokenizer.json` and wrapping it in `PreTrainedTokenizerFast`.
* **Trainer Arguments:** `remove_unused_columns=False` is strictly enforced so the HF Trainer does not strip the custom `label_lengths` tensor required by PyTorch's `CTCLoss`.

---

**Next Steps:** Proceed to code the dataset preparation scripts (`prepare_tokens.py`) and initiate the local Proof of Concept (PoC) training loop.