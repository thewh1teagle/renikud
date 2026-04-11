#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "model_config.h"
#include "weights.h"
#include "model.h"

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------
#define H   HIDDEN_SIZE        // 96
#define NH  NUM_HEADS          // 4
#define HD  HEAD_DIM           // 24
#define IL  INTERMEDIATE_SIZE  // 192
#define NC  NUM_CONSONANTS     // 25
#define NV  NUM_VOWELS         // 6
#define NS  NUM_STRESS         // 2

// Max input tokens (CLS + chars + SEP)
#define MAX_TOKENS 64

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------

// Decode one UTF-8 codepoint from s, advance *s past it. Returns 0 at end.
static uint32_t utf8_next(const char **s) {
    const unsigned char *p = (const unsigned char *)*s;
    if (!*p) return 0;
    uint32_t cp;
    int len;
    if (*p < 0x80)       { cp = *p;               len = 1; }
    else if (*p < 0xE0)  { cp = *p & 0x1F;        len = 2; }
    else if (*p < 0xF0)  { cp = *p & 0x0F;        len = 3; }
    else                 { cp = *p & 0x07;        len = 4; }
    for (int i = 1; i < len; i++) cp = (cp << 6) | (p[i] & 0x3F);
    *s += len;
    return cp;
}

// Write a UTF-8 codepoint to buf, return bytes written.
static int utf8_write(char *buf, uint32_t cp) {
    if (cp < 0x80)       { buf[0] = cp; return 1; }
    if (cp < 0x800)      { buf[0]=0xC0|(cp>>6); buf[1]=0x80|(cp&0x3F); return 2; }
    if (cp < 0x10000)    { buf[0]=0xE0|(cp>>12); buf[1]=0x80|((cp>>6)&0x3F); buf[2]=0x80|(cp&0x3F); return 3; }
    buf[0]=0xF0|(cp>>18); buf[1]=0x80|((cp>>12)&0x3F); buf[2]=0x80|((cp>>6)&0x3F); buf[3]=0x80|(cp&0x3F); return 4;
}

static int is_hebrew(uint32_t cp) {
    return cp >= 0x05D0 && cp <= 0x05EA;  // א–ת
}

// ---------------------------------------------------------------------------
// Vocab lookup: binary search on sorted VOCAB_CODEPOINTS
// ---------------------------------------------------------------------------
static int vocab_lookup(uint32_t cp) {
    int lo = 0, hi = VOCAB_CHAR_COUNT - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (VOCAB_CODEPOINTS[mid] == cp) return VOCAB_TOKEN_IDS[mid];
        if (VOCAB_CODEPOINTS[mid] < cp)  lo = mid + 1;
        else                             hi = mid - 1;
    }
    return UNK_TOKEN_ID;
}

// ---------------------------------------------------------------------------
// Letter constraint lookup
// ---------------------------------------------------------------------------
static const uint8_t *get_constraints(uint32_t cp, int *len_out) {
    for (int i = 0; i < NUM_HEBREW_LETTERS; i++) {
        if (HEBREW_LETTER_CODEPOINTS[i] == cp) {
            *len_out = CONSTRAINT_LEN[i];
            return &CONSTRAINT_IDS[CONSTRAINT_OFFSET[i]];
        }
    }
    *len_out = 0;
    return NULL;
}

// ---------------------------------------------------------------------------
// Linear layer: y = x @ W^T + b
// W shape: [out_dim, in_dim], x shape: [in_dim], y shape: [out_dim]
// int8 variant dequantizes on the fly with scale
// ---------------------------------------------------------------------------
static void linear_f32(const float *x, const float *W, const float *b,
                        float *y, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float acc = b ? b[o] : 0.0f;
        const float *row = W + o * in_dim;
        for (int i = 0; i < in_dim; i++) acc += x[i] * row[i];
        y[o] = acc;
    }
}

static void linear_i8(const float *x, const int8_t *W, float scale,
                       const float *b, float *y, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        float acc = b ? b[o] : 0.0f;
        const int8_t *row = W + o * in_dim;
        for (int i = 0; i < in_dim; i++) acc += x[i] * (row[i] * scale);
        y[o] = acc;
    }
}

// ---------------------------------------------------------------------------
// LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
// ---------------------------------------------------------------------------
static void layernorm(const float *x, const float *w, const float *b,
                      float *y, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    float var = 0.0f;
    for (int i = 0; i < dim; i++) { float d = x[i] - mean; var += d * d; }
    var /= dim;
    float inv = 1.0f / sqrtf(var + 1e-12f);
    for (int i = 0; i < dim; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// ---------------------------------------------------------------------------
// GELU approximation (matches PyTorch's tanh variant)
// ---------------------------------------------------------------------------
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// ---------------------------------------------------------------------------
// Softmax in-place
// ---------------------------------------------------------------------------
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// ---------------------------------------------------------------------------
// Argmax
// ---------------------------------------------------------------------------
static int argmax(const float *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

// ---------------------------------------------------------------------------
// BERT layer
// Each layer struct just holds pointers into the generated weight arrays.
// ---------------------------------------------------------------------------
typedef struct {
    const int8_t *q_w, *k_w, *v_w, *out_w;
    float q_s, k_s, v_s, out_s;
    const float *q_b, *k_b, *v_b, *out_b;
    const float *attn_ln_w, *attn_ln_b;
    const int8_t *ffn1_w, *ffn2_w;
    float ffn1_s, ffn2_s;
    const float *ffn1_b, *ffn2_b;
    const float *ffn_ln_w, *ffn_ln_b;
} BertLayer;

// Scratch buffers (static to avoid stack overflow on MCU)
static float _q[MAX_TOKENS][H];
static float _k[MAX_TOKENS][H];
static float _v[MAX_TOKENS][H];
static float _attn_w[MAX_TOKENS][MAX_TOKENS];
static float _attn_out[MAX_TOKENS][H];
static float _tmp[MAX_TOKENS][H];
static float _ffn_mid[MAX_TOKENS][IL];

static void bert_layer(float hidden[MAX_TOKENS][H], int seq_len, const BertLayer *L) {
    // --- Self-attention ---
    for (int t = 0; t < seq_len; t++) {
        // Pre-norm happens inside attention output layernorm in BERT (post-norm)
        linear_i8(hidden[t], L->q_w, L->q_s, L->q_b, _q[t], H, H);
        linear_i8(hidden[t], L->k_w, L->k_s, L->k_b, _k[t], H, H);
        linear_i8(hidden[t], L->v_w, L->v_s, L->v_b, _v[t], H, H);
    }

    // Attention scores per head
    float scale = 1.0f / sqrtf((float)HD);
    for (int h = 0; h < NH; h++) {
        int off = h * HD;
        // scores
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < HD; d++) dot += _q[i][off+d] * _k[j][off+d];
                _attn_w[i][j] = dot * scale;
            }
            softmax(_attn_w[i], seq_len);
        }
        // weighted sum of V
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < HD; d++) {
                float acc = 0.0f;
                for (int j = 0; j < seq_len; j++) acc += _attn_w[i][j] * _v[j][off+d];
                _attn_out[i][off+d] = acc;
            }
        }
    }

    // Output projection + residual + LayerNorm
    for (int t = 0; t < seq_len; t++) {
        linear_i8(_attn_out[t], L->out_w, L->out_s, L->out_b, _tmp[t], H, H);
        for (int i = 0; i < H; i++) _tmp[t][i] += hidden[t][i];  // residual
        layernorm(_tmp[t], L->attn_ln_w, L->attn_ln_b, hidden[t], H);
    }

    // --- FFN ---
    for (int t = 0; t < seq_len; t++) {
        linear_i8(hidden[t], L->ffn1_w, L->ffn1_s, L->ffn1_b, _ffn_mid[t], H, IL);
        for (int i = 0; i < IL; i++) _ffn_mid[t][i] = gelu(_ffn_mid[t][i]);
        linear_i8(_ffn_mid[t], L->ffn2_w, L->ffn2_s, L->ffn2_b, _tmp[t], IL, H);
        for (int i = 0; i < H; i++) _tmp[t][i] += hidden[t][i];  // residual
        layernorm(_tmp[t], L->ffn_ln_w, L->ffn_ln_b, hidden[t], H);
    }
}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

// hidden state buffer
static float _hidden[MAX_TOKENS][H];

// logit buffers
static float _consonant_logits[NC];
static float _vowel_logits[NV];
static float _stress_logits[NS];
static float _vowel_in[H + NC];
static float _stress_in[H + NC + NV];

static const BertLayer LAYERS[3] = {
    {
        layer0_q_w, layer0_k_w, layer0_v_w, layer0_attn_out_w,
        layer0_q_w_scale, layer0_k_w_scale, layer0_v_w_scale, layer0_attn_out_w_scale,
        layer0_q_b, layer0_k_b, layer0_v_b, layer0_attn_out_b,
        layer0_attn_ln_w, layer0_attn_ln_b,
        layer0_ffn1_w, layer0_ffn2_w,
        layer0_ffn1_w_scale, layer0_ffn2_w_scale,
        layer0_ffn1_b, layer0_ffn2_b,
        layer0_ffn_ln_w, layer0_ffn_ln_b,
    },
    {
        layer1_q_w, layer1_k_w, layer1_v_w, layer1_attn_out_w,
        layer1_q_w_scale, layer1_k_w_scale, layer1_v_w_scale, layer1_attn_out_w_scale,
        layer1_q_b, layer1_k_b, layer1_v_b, layer1_attn_out_b,
        layer1_attn_ln_w, layer1_attn_ln_b,
        layer1_ffn1_w, layer1_ffn2_w,
        layer1_ffn1_w_scale, layer1_ffn2_w_scale,
        layer1_ffn1_b, layer1_ffn2_b,
        layer1_ffn_ln_w, layer1_ffn_ln_b,
    },
    {
        layer2_q_w, layer2_k_w, layer2_v_w, layer2_attn_out_w,
        layer2_q_w_scale, layer2_k_w_scale, layer2_v_w_scale, layer2_attn_out_w_scale,
        layer2_q_b, layer2_k_b, layer2_v_b, layer2_attn_out_b,
        layer2_attn_ln_w, layer2_attn_ln_b,
        layer2_ffn1_w, layer2_ffn2_w,
        layer2_ffn1_w_scale, layer2_ffn2_w_scale,
        layer2_ffn1_b, layer2_ffn2_b,
        layer2_ffn_ln_w, layer2_ffn_ln_b,
    },
};

void g2p_phonemize(const char *utf8_in, char *out_buf, int out_size) {
    // --- Tokenize ---
    int token_ids[MAX_TOKENS];
    uint32_t codepoints[MAX_TOKENS];  // original codepoint per token (for constraints)
    int seq_len = 0;

    token_ids[seq_len] = CLS_TOKEN_ID;
    codepoints[seq_len] = 0;
    seq_len++;

    const char *p = utf8_in;
    uint32_t cp;
    while ((cp = utf8_next(&p)) != 0 && seq_len < MAX_TOKENS - 1) {
        token_ids[seq_len] = vocab_lookup(cp);
        codepoints[seq_len] = cp;
        seq_len++;
    }

    token_ids[seq_len] = SEP_TOKEN_ID;
    codepoints[seq_len] = 0;
    seq_len++;

    // --- Embeddings: word + position (no type embeddings needed, type=0 → all same) ---
    for (int t = 0; t < seq_len; t++) {
        int wid = token_ids[t];
        // word embedding
        for (int i = 0; i < H; i++)
            _hidden[t][i] = emb_word[wid * H + i] + emb_pos[t * H + i];
        // type embedding is all-same (type_id=0), add it
        for (int i = 0; i < H; i++)
            _hidden[t][i] += emb_type[i];  // token_type_embeddings[0]
    }
    // Embedding LayerNorm
    for (int t = 0; t < seq_len; t++)
        layernorm(_hidden[t], emb_ln_w, emb_ln_b, _hidden[t], H);

    // --- Transformer layers ---
    for (int l = 0; l < NUM_LAYERS; l++)
        bert_layer(_hidden, seq_len, &LAYERS[l]);

    // --- Decode each Hebrew character ---
    // stress: pick highest stress[1] logit per word
    // First pass: compute all logits
    float stress_score[MAX_TOKENS];
    int consonant_pred[MAX_TOKENS];
    int vowel_pred[MAX_TOKENS];

    // word boundary tracking for stress selection
    // a "word" is a run of non-space tokens
    int word_id[MAX_TOKENS];
    int cur_word = -1;
    int prev_space = 1;
    for (int t = 1; t < seq_len - 1; t++) {  // skip CLS/SEP
        uint32_t c = codepoints[t];
        int is_space = (c == 0x20);
        if (!is_space && prev_space) cur_word++;
        word_id[t] = is_space ? -1 : cur_word;
        prev_space = is_space;
    }
    int num_words = cur_word + 1;

    // Per-token predictions
    for (int t = 1; t < seq_len - 1; t++) {
        if (!is_hebrew(codepoints[t])) {
            consonant_pred[t] = -1;
            vowel_pred[t] = -1;
            stress_score[t] = -1e9f;
            continue;
        }

        // Consonant head: [H] -> [NC]
        linear_i8(_hidden[t], head_consonant_w, head_consonant_w_scale, head_consonant_b, _consonant_logits, H, NC);

        // Apply letter constraint
        int n_allowed;
        const uint8_t *allowed = get_constraints(codepoints[t], &n_allowed);
        if (n_allowed > 0) {
            // zero out disallowed
            float best = -1e30f;
            int best_id = allowed[0];
            for (int a = 0; a < n_allowed; a++) {
                if (_consonant_logits[allowed[a]] > best) {
                    best = _consonant_logits[allowed[a]];
                    best_id = allowed[a];
                }
            }
            consonant_pred[t] = best_id;
        } else {
            consonant_pred[t] = argmax(_consonant_logits, NC);
        }

        // Vowel head: [H + NC] -> [NV]
        memcpy(_vowel_in, _hidden[t], H * sizeof(float));
        memcpy(_vowel_in + H, _consonant_logits, NC * sizeof(float));
        linear_i8(_vowel_in, head_vowel_w, head_vowel_w_scale, head_vowel_b, _vowel_logits, H + NC, NV);
        vowel_pred[t] = argmax(_vowel_logits, NV);

        // Stress head: [H + NC + NV] -> [NS]
        memcpy(_stress_in, _vowel_in, (H + NC) * sizeof(float));
        memcpy(_stress_in + H + NC, _vowel_logits, NV * sizeof(float));
        linear_i8(_stress_in, head_stress_w, head_stress_w_scale, head_stress_b, _stress_logits, H + NC + NV, NS);
        stress_score[t] = _stress_logits[1];  // logit for "stressed"
    }

    // Per-word: pick the token with highest stress score
    float best_stress[32] = {0};
    int   stressed_tok[32];
    for (int w = 0; w < num_words && w < 32; w++) stressed_tok[w] = -1;
    for (int t = 1; t < seq_len - 1; t++) {
        int w = word_id[t];
        if (w < 0 || w >= 32) continue;
        if (stressed_tok[w] < 0 || stress_score[t] > best_stress[w]) {
            best_stress[w] = stress_score[t];
            stressed_tok[w] = t;
        }
    }

    // --- Assemble output ---
    char *out = out_buf;
    int remaining = out_size - 1;

    for (int t = 1; t < seq_len - 1; t++) {
        uint32_t c = codepoints[t];

        if (!is_hebrew(c)) {
            // pass through non-Hebrew (spaces, punctuation) except apostrophe/quote
            if (c != '\'' && c != '"') {
                int n = utf8_write(out, c);
                out += n; remaining -= n;
            }
            continue;
        }

        int cid = consonant_pred[t];
        int vid = vowel_pred[t];
        int w   = word_id[t];
        int stressed = (w >= 0 && w < 32 && stressed_tok[w] == t);

        // Furtive patah: word-final ח with vowel 'a' → emit aχ
        int word_final = (t + 1 >= seq_len - 1) || !is_hebrew(codepoints[t+1]);
        if (c == 0x05D7 && word_final && vid == 1 /* 'a' */) {
            if (stressed && remaining > 3) { memcpy(out, "\xcb\x88", 2); out+=2; remaining-=2; }
            if (remaining > 4) { memcpy(out, "a\xcf\x87", 3); out+=3; remaining-=3; }
            continue;
        }

        // Consonant (skip ∅)
        if (cid != 0) {
            const char *cs = CONSONANT_LABELS[cid];
            int n = strlen(cs);
            if (n <= remaining) { memcpy(out, cs, n); out+=n; remaining-=n; }
        }
        // Stress mark ˈ (U+02C8, 2 bytes)
        if (stressed && remaining > 2) { memcpy(out, "\xcb\x88", 2); out+=2; remaining-=2; }
        // Vowel (skip ∅)
        if (vid != 0) {
            const char *vs = VOWEL_LABELS[vid];
            int n = strlen(vs);
            if (n <= remaining) { memcpy(out, vs, n); out+=n; remaining-=n; }
        }
    }
    *out = '\0';
}
