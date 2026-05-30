# ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion

Maxim Melichov

Yakov Kolani

Morris Alper

*Reichman University*

*Independent Researcher*

*Carnegie Mellon University*

***Abstract*****—Grapheme-to-phoneme** **(G2P)** **conversion** **for** **Mod-** *beer*) or /biʁ'a/ ( *capital city* ). Secondly, nikud convention-

**ern** **Hebrew** **is** **needed** **for** **applications** **like** **text-to-speech** **(TTS),**ally reflects traditional grammatical rules that do not match

**but** **is** **challenging** **due** **to** **the** **language’s** **abjad** **writing** **sys-**everyday spoken Hebrew. For instance, the word ךרדבוis for-

**tem,** **which** **leaves** **vowels** **largely** **unwritten,** **creating** **widespread** mally vocalized and pronounced asubadˈeʁeχ/, but in natural

**ambiguity.** **Standard** **approaches** **first** **predict** **vowel** **diacritics** speech, it is pronounced vebadˈeʁeχ/. Similarly,

םילשוריב

**(nikud)** **to** **produce** **International** **Phonetic** **Alphabet** **(IPA)** **tran-**

**scriptions,** **but** **this** **is** **limited:** **vocalization** **data** **is** **scarce** **and**is formally /biʁuʃalˈajim/, but native speakers typically say

**laborious** **to** **produce,** **it** **does** **not** **specify** **features** **such** **as** **lexical**bejeʁuʃalˈajim/ [4], [5].

**stress,** **and** **it** **reflects** **formal** **grammatical** **rules** **rather** **than** Another issue is a lack of scalable data sources for directly

**everyday** **spoken** **pronunciation.** **Direct** **sequence-to-sequence** **IPA** learning G2P. Vocalization models [6], [7] learn from Hebrew

**prediction,** **meanwhile,** **struggles** **on** **limited** **data** **and** **produces** text with manually annotated vowel diacritics, data which

**frequent** **hallucinations** **due** **to** **a** **lack** **of** **character-level** **alignment.**

**Our** **method,** **ReNikud,** **overcomes** **these** **limitations** **with** **two**is scarce and laborious to produce. Similarly, methods such

**key** **insights:** **(1)** **Weak** **audio** **supervision** **via** **a** **phoneme-based**as [3], [8] that learn from IPA annotations are bottlenecked

**automatic** **speech** **recognition** **(ASR)** **pseudo-labeling** **pipeline** **on** by data availability. Conversely, another abundant source of

**thousands** **of** **hours** **of** **unlabeled** **Hebrew** **audio,** **yielding** **phonemic** data on pronunciation exists—unlabelled Hebrew audio—but

**transcriptions** **that** **reflect** **natural** **spoken** **norms** **without** **manual** existing methods can only learn from text.

**annotation.** **(2)** **A** **pseudo-vocalization** **architecture** **that** **predicts**

**IPA** **phonemes** **at** **each** **character** **position,** **enforcing** **character-** Finally, while methods using vocalization poorly reflect

**level** **alignment** **to** **mitigate** **hallucinations.** **Results** **on** **existing** spoken pronunciation, they have been shown to outperform

**Hebrew** **G2P** **benchmarks** **and** **new** **targeted** **test** **suites** **for** **spoken** direct sequence-to-sequence (seq2seq) prediction of IPA [3],

**Hebrew** **show** **that** **ReNikud** **surpasses** **previous** **state-of-the-art** the latter of which struggles to learn on limited data and

**methods.** **We** **will** **release** **our** **code** **and** **trained** **models** **to** **support**

**further** **work** **on** **Hebrew** **TTS** **and** **speech** **technologies.**

produces frequent hallucinations due to a lack of enforced

***Index*** ***Terms*****—Grapheme-to-Phoneme,** **Text-to-Speech,** **Modern** character-level alignment with the input transcript.

**Hebrew,** **Weakly** **Supervised** **Learning,** **Automatic** **Speech** **Recog-**

2

Our method, *ReNikud* , directly addresses these challenges.

**nition,** **Lexical** **Stress**

To reflect spoken norms and unlock abundant training data, we propose a novel pipeline to train G2P with*weak supervision*

I. Introduction

*from unlabeled Hebrew audio*. By applying an automatic

With increasing interest in text-to-speech (TTS) systems for speech recognition (ASR)-based pseudo-labeling pipeline, we

low-resource languages, the modern Hebrew language poses are able to extract pronunciation information and train on

particular challenges. Hebrew is written as an*abjad*—a writ- thousands of hours of recordings, a scalable approach that

ing system that normally does not indicate vowel sounds [1]. effectively learns spoken Hebrew norms. To enforce character-

Consequently, many words are homographs and require seman- level alignment between Hebrew characters and phonemes, we

tic context for correct pronunciation. For example, the word introduce a *pseudo-vocalization* architecture. Like traditional 1

םירפסמcan be read as mispaʁˈim *numbers*), /mispaʁˈajim vocalization, this predicts phonetic content aligned with each

*scissors*), or mesapʁˈim/ ( *telling*). As this ambiguity con- character position, but rather than predicting traditional*nikud*

founds TTS generations, leading open-source approaches first symbols it directly predicts character-aligned IPA phonemes.

predict *nikud* (vocalization, i.e., conventional vowel diacritics) By using the orthographic structure of Hebrew text, this

to condition synthesis [2], [3].

mitigates hallucinations and increases data efficiency relative

However, there are substantial gaps between vocalization to seq2seq baselines, as we demonstrate.

and fully-specified International Phonetic Alphabet (IPA) tran- Our results on existing Hebrew G2P benchmarks show

scriptions for grapheme-to-phoneme (G2P) conversion.

that ReNikud succeeds in leveraging audio data and better

One issue is the mismatch between written and spoken reflecting Hebrew speech. In addition, we propose additional

Hebrew. Firstly, nikud does not fully specify phonetic features test suites and targeted evaluations to show improvements on

such as lexical stress, e.g., הריבmay be pronounced as b'iʁa difficult phonetic cases reflecting spoken language.

Equal contribution 1We place the IPA stress mark directly before the stressed vowel, following 2

TTS conventions.

Rethinking Nikud

---

We will release our code, data and trained models to spur

work on Hebrew speech technologies.

Figure 1: **System** **overview.** We first pseudo-label audio (left)

by creating a many-to-one FST alignment between unvocalized Hebrew text and IPA phonemes derived from two parallel

ASR runs applied to Hebrew audio. We then train a pseudo- vocalization architecture (right) where unvocalized Hebrew

characters are passed through a character encoder to predict a phonetic triplet (consonant, stress, and vowel) at each position via parallel classification heads.

II. Method

The ReNikud pipeline consists of two stages, shown in Fig- ure 1: audio pseudo-labeling (Section II-A), and our pseudo- vocalization architecture trained on this data (Section II-B).

*A. Audio Pseudo-Labeling*

To learn pronunciation from unlabeled audio at scale, we

construct a pipeline that extracts character-aligned IPA anno- tations using two parallel ASR systems, as shown in Figure 1 (upper left). We extract Hebrew orthographic transcripts with a standard pretrained Hebrew ASR model, and IPA transcripts with a custom ASR model trained to output IPA when applied to Hebrew audio. By applying both ASR systems to a large- scale, unlabeled Hebrew audio corpus, we extract parallel

Hebrew text and IPA transcriptions that serve*pseudo-labels* as

providing weak supervision for our downstream G2P model.unvocalized Importantly, we retain only pairs where the Hebrew and

IPA transcripts agree on the number of words and where

every word passes the FST alignment procedure describedStandard sequence-to-sequence

below. This filtering retains approximately[TODO: XXX%]of

the original samples. While strict, this selection criterion still yields several million[Morris: make precise?]aligned sentence from the large dataset, selecting for more reliably transcribed utterances. Finally, we perform a string alignment process based on

the Hebrew orthography’s abjad structure. In general, abjads encode consonants linearly with interleaved, unwritten vowels, meaning that graphemes map monotonically to (consonant,

vowel) pairs. As Hebrew also has unwritten lexical stress, each

grapheme maps to a*phonetic triplet* encoding such a pair and

a binary stress indicator. We find this alignment with a simple finite state transducer (FST) handling known consonant values, including one-to-many (e.g.,ב b, v/) and many-to-one (e.g.,

ט ת t/) mappings, as well as orthographic complexities such

as:

- **Digraphs:** Loanwords in Modern Hebrew frequently use

an apostrophe (*geresh*) to denote non-native phonemes

(e.g., ׳זfor /ʒ/; the base letterז normally represents /z/).

The FST assigns the digraph phoneme to the base letter and passes over the*geresh*

- Word-finalח χ/) may occur with an additional*preced-*

*ing* a/ vowel (*patah gnuva*), e.g., חולl'uaχ/. We handle

this reordering with a special combinedaχ / value for the

vowel slot.

- **Silent** **letters** : The Hebrew lettersו, י ה א may either

indicate consonant sounds or may be silent*matres lectio-*

*nis*). We handle the latter case with a null) consonant

class. An example of the resulting alignment between Hebrew

characters and phonetic triplets is shown in Table I.

**Word** **Char**

| םולש(/ | ʃal'om/) | ש | /ʃ/ / | a/ 0 |  |
|---|---|---|---|---|---|
|  |  | ל | /l/ / | o/ 1 |  |
|  |  | ו | ∅ ∅ |  | 0 |
|  |  | ם | /m/ | ∅ | 0 |

| ספי׳צ(/tʃ'ips/) | צ | /tʃ/ / | i/ 1 |  |
|---|---|---|---|---|
|  | ׳ | ∅ ∅ |  | 0 |
|  | י | ∅ ∅ |  | 0 |
|  | פ | /p/ | ∅ | 0 |
|  | ס | /s/ | ∅ | 0 |

**Consonant** **Vowel** **Stress**

| חופת(/tap'uaχ | /) ת | /t/ / | a/ 0 |  |
|---|---|---|---|---|
|  | פ | /p/ / | u/ 1 |  |
|  | ו | ∅ ∅ |  | 0 |
|  | ח | ∅ | /aχ/ 0 |  |

Table I: **Examples** **of** **FST-derived** **alignment** between

Hebrew characters and phonetic triplets (consonant, vowel,

stress).

*B. Pseudo-Vocalization Architecture*

Our goal is to create a Hebrew G2P model that maps

Hebrew text directly to IPA strings. Because

Hebrew is an *abjad*, there is a strong, local relationship be-

tween individual written letters and their phonetic realizations. models fail to exploit these

localized properties, leading to hallucinations. To mitigate this, we frame the G2P task as a con-

strained, per-character classification problem—a method we term *Pseudo-Vocalization*, illustrated in Figure 1 (right). While

a single Hebrew character typically corresponds to more than one IPA symbol (e.g., a consonant followed by a vowel), we resolve this by having every Hebrew letter independently pre-

dict exactly one phonetic triplet, as described in Section II-A. The core model is a character-level transformer encoder. On top of this encoder, we attach three parallel, independent

---

classification heads that simultaneously predict the phonetic attributes for each character directly from the encoder’s hidden states:

- **Consonant** **Head:** Selects from 25 IPA consonants or

null (

- **Vowel** **Head:** Selects from 5 vowels, null ), or the

special /aχ/ token (see Section II-A).

- **Stress** **Head:** Binary classifier for lexical stress.

At inference time, realizations are predicted by takingmanually annotated sentences from

the argmax of logits for each head. In addition, we apply

*constrained decoding* to enforce hard constraints on Hebrewreport Character Error Rate (CER), Word Error Rate (WER),

letters and phonetic realizations: the argmax is calculated only over possible consonantal realizations of a letter (e.g.,ב can

only be realized as b/ or /v/). We also enforce a word-level

constraint that exactly one lexical stress is predicted.

*C. Renikud Classifier* The core model is built upondicta-il/dictabert-

large-char, a character-level BERT encoder pre-trained onTable

Hebrew. We selected this base encoder because our method

requires character-level representation. On top of this base

encoder, we attach three independent classification heads that simultaneously predict the following phonetic elements for

every input character:

- **Letter** **Head:** Predicts the base consonant from a prede-

fined set of 25 possible classes.

- **Vowel** **Head:** Predicts the associated vowel from a set of 7 possible classes.

- **Stress** **Head:** A binary classifier indicating whether the

character carries primary lexical stress (true/false).

III. Experiments and Results

*A. Experimental Setup* For our base ASR model, we employ the Whisper Large v3 Turbo checkpoint adapted for Hebrew by ivrit.ai, which we subsequently fine-tune on text-and-IPA objective. Our training process is divided into two distinct stages:

**Dataset** **Size**

**Role**

| SASpeech | [9] | ∼18h Pretraining | (Stage 1) |
|---|---|---|---|
| Recital |  | ∼46h Pretraining | (Stage 1) |
| ILSpeech | [3] | ∼2h Fine-tuning | (Stage 2) |

Table II: **Data** **sources** for IPA ASR training.

**Stage** **1:** **Pretraining.**During the initial pre-training phase,

we utilized two primary corpora. The first is a subset of the *SASpeech* dataset [9]. While the full corpus contains roughly

30 hours of audio, the automatic audio files that contain 26 hours we filtered short files with single word, we used the ivirit’s ASR to filter all the audio with a lot of ”umm..” by

doing this we trim it down to approximately 14 hours

the automatically aligned data. with the 4 manual data we got about 18 hours. The second is the crowd-sourced*Recital*

dataset, consisting of approximately 46 hours of raw audio. we processed the datasets’ existing Hebrew transcripts directly through Phonikud [3] to produce IPA.

**Stage** **2:** **Fine-tuning** **and** **Evaluation** **of** **the** **ASR**Fol-

lowing pre-training, the model was fine-tuned on the manual ipa *ILSpeech* dataset [3], which comprises approximately 2

hours of audio. From the dataset, we extracted exactly 150 recordings for our held-out test set, reserving the remainder for fine-tuning. To assess the ASR-IPA capabilities, we evaluated the fine-

tuned model on two datasets: the held-out test set of 150

*ILSpeech*, and *Michel*, a

synthetic out-of-domain dataset comprising 250 sentences. We

Vowel Error Rate (VER), and Stress Error Rate (SER). Results

are detailed in Table III.

**Test** **Set** **CER**

**WER** **VER** **SER**

ILSpeech (150 seq.) 0.0240 0.1120 0.0224 0.9240

Michel (250 seq.) 0.0478 0.2216 0.0321 0.1767

III: **ASR** **IPA** **evaluation** **results** on the in-domain

ILSpeech and out-of-domain synthetic Michel datasets.

[Max: is 350 sentences enough for showing the results?] [Max: is there a way to explain that michel is a legit dataset?] **Stage** **3:** **Training** **of** **Renikud**For our primary large-scale

dataset, we utilized Knesset Vox [10], specifically drawing

upon approximately 1.7k hours of parliamentary recordings from its training set. Knesset Vox has ready to use transcripts by using ASR and forced alignment between the already made scripts from the knesset parliament transcripts. Because the Whisper ASR model experiences performance degradation on longer audio inputs (exceeding 25 seconds), we bypassed the original Knesset Vox sentence annotations and re-segmented the raw audio into optimized 5–15-second clips to ensure

correct results of phonetic transcriptions with the ASR. We then processed these shorter segments using both a standard Hebrew Whisper model and our custom IPA-tuned Whisper model to extract parallel Hebrew text and IPA transcriptions. After extracting the pseudo-labels and applying the FST align- ment process described in Section II-A,To ensure memory

stability and prevent out-of-memory (OOM) errors during

training, we filtered out sentence-length outliers using the stan- dard 1 5 IQR rule. Specifically, we discarded sequences with

lengths falling outside the bounds ofQ 1 5 IQR, Q 1

3

1 5 IQR . Ultimately, this weak supervision pipeline yielded

a dataset of approximately 1.52 million aligned Hebrew-to-IPA sentences. aligned Hebrew-to-IPA sentences. [Morris: does

Knesset Vox have gold transcripts? do we ignore them?][Max:

Yes, but we create shorter clips of 5-15 and it will be a

headache to use the gold labels also probably were made with whisper] [Morris: should be explained concisely][Max: added

of some explanation about the transcriptions] **Stage** **4:** **Evaluation** **of** **Renikud**To benchmark our pro-

posed ReNikud architecture, we evaluated its grapheme-to-

SASpeech [9] ILSpeech [3]phoneme mapping capabilities against the standard Phonikud baseline.

---

**Cat.** **Input**

Gender Acronyms

**Target**

ךתיארדסבלכה

ן״מאב

**IPA** **Cat.**

ידיגת ךתיאʔitˈaχ/ Names

ן״מאבתרישאוהbeʔamˈan/ Homogr.

Penult.

םחל ירטםחליתינקlˈeχem/ Min.

Rare Ph.

רהמלכאת

ךליבשבספי׳ציתאבהספי׳צtʃˈips/ Slang

Foreign

ביוו

הפשיבוטביווהזיאvˈajb

Table IV: Selected examples from BITUI, split by category (Cat.). Abbreviated category names are penultimate stress (Penult.),

rare phonemes (Rare Ph.), homographs (Homogr.), and minimal stress pairs (Min. Str.).

*B. BITUI Testset*

3

BITUI testset consists of a targeted evaluation corpus de-

signed to systematically assess a model’s capacity for complex linguistic disambiguation in context-dependent orthographies. BITUI testset except of the ilspeech, the generation of the sentences was created using Gemini-3.1-pro and then we fixed it manually. the high result of the gemini in the graphs might be the results of bias towards his sentences. ILspeech was updated from Phonikud [3] mistakes were fixed like e.g. ןכותו

םירטאווא, it was םירטבאןכתו[Morris: how many samples are

in BITUI?] **Targeted** **Evaluation** **Methodology:**

The dataset is strictly categorized into five distinct linguistic challenges:

- **Morphological** **and** **Syntactic** **Ambiguity** **gender**

Hebrew pronouns, verbs, and adjectives frequently share identical orthography but shift in pronunciation based on the gender of the subject or object.

*Example:* ותוא ארקת ,רפסהתאךל ןתונינא

*Targeted Label:* Index 2 ( ךל leχˈa/ (Resolved via

the masculine future verb ארקת

- **Semantic** **Ambiguity** **homographs** This category

tests identically spelled words with completely divergentPerformance is meanings and phonetic structures.

*Example 1:* קרמללצב יתכתח(I cut an **onion** for

the soup) batsˈal

*Example 2:* ץעה לצב ונבשי(We sat **in** **the** **shade**

of the tree) betsˈel/. [Max: not sure about this

example]

- **Colloquialisms** **and** **Slang** **slang** Modern spoken

Hebrew frequently diverges from prescriptive dictionary rules.

*Example:*

לומתאהתייההחידאפוזיא

*Targeted Label:*

החידאפfadˈiχa

- **Foreign** **Loanwords** **and** **Rare** **Phonemesloanwords**

Hebrew orthography adapts to non-native phonemes (e.g.,upper bound for the possible results. /w/, /d/, /t/) using modifying characters (geresh) or

unusual letter combinations.

*Example:*

ספי’צ

*Targeted Label:* ספיצ

3Benchmark for Individual-word Testing of U

ורגרובמהיתנמזה

tʃˈips

nderspecified IPA

**Input**

**Target**

**IPA**

ילשהלוחכההצלוחההפיאיהיליהילlˈihi

בשיאוהשןמזבןיינעמרפסארקאוהרפסsˈefeʁ/~/sapˈaʁ

םישנרפסלצא

Str.

וידלילשםיגשיההמחורתחנולהיהתחנnˈaχat/~/naχˈat

לולסמהלעםולשבתחנסוטמה ירמגלבונגםדאןבאוהבונגɡanˈuv

Figure 2: Word accuracy rate (left) and character accuracy rate (right). [Max: should i add the name Gemini-3.1-pro? in the graph or in caption?][Morris: in caption or paper text is fine.

It should be a gray dotted line, not solid black, and in caption we should say that it provides an upper bound]

- **Baseline** **Control** **ilspeech-v2-test** Main testset of

phonikud and also to test general speech classification correction. *Example:*

םיטקיורפרופסניאבאליממתברועמהתייה

םימדקתמםייאבצ

**Evaluation** **Methodology:** Predictions were compared

against human-annotated gold-standard IPA transcriptions.

reported via Word Error Rate (WER) and

Character Error Rate (CER). The overall model performance (reported as OVERALL) is calculated as a micro-average

of the error rates across all distinct evaluation categories,

encompassing both the challenge suite and the generalized test set in [3] as seen in **Results** **and** **Analysis:**The comparative results are detailed

in Table V. For example, ReNikud accurately predicts the

colloquial slang

החידאפas / fadˈiχa/, whereas the Phonikud

incorrectly outputs the padiχˈa/. Furthermore, ReNikud suc-

cessfully maps non-native phonemes in foreign loanwords,

correctly resolving the English ’w’ in רניווwˈineʁ/) and the

’j’ sound in בוג hadʒˈob/), which the Phonikud misclassifies

as (/vinˈeʁ/ and /hadʒˈov/). Gemini-3.1-pro results used as an

[Morris: do we have a category for things like vebaderex which we discussed in the intro?][Max: nope]

[Morris: add something mentioning that ILSpeech test set is expanded/updated (so it’s clear why Gemini results aren’t identical to the ones in Phonikud paper)] [Morris: add qualitative results – still missing full results]

---

Stress

Phonemes

Gender Acronyms Penult. Rare

| ReNikud | (Ours) | 49 | **17**/ **34** | / **9 20** | / **4 59** | / **18 46** | / **11** | **29** / **9 32** | / **21 38** | / **11 16** | / **4 27** | / **8** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ReNikud | (Phonikud | Data) | **47** / 18 76 | / 26 30 | / 6 82 | / | 30 47 | / 13 33 | / 11 34 | / 21 44 | / 15 19 | / 5 33 |
| Phonikud |  | (Baseline) 62 | / 22 64 | / 24 42 | / 9 83 |  | / 28 66 | / 18 35 | / 11 36 | / 21 59 | / 21 20 | / 5 |

Table V: Detailed evaluation on the Hebrew G2P challenge suite (WER / CER, in %). Our proposed ReNikud model trained on

the Knesset Vox corpus achieves the best overall performance, significantly reducing errors in complex domains such as Slang and Rare Phonemes compared to both the Phonikud baseline and an ablation trained on legacy the ablation]. Best results per row are highlighted in bold.

|  | suite (WER | / CER, | in %). Our | proposed | ReNikud | model |
|---|---|---|---|---|---|---|
|  | significantly | | reducing errors | in | complex | domains such |
|  | and an | ablation trained | on | legacy[Morris: | data don’t | understand |
|  | [Morris: error | rates should | have | one decimal | place | e.g. 49.1 |

just 49]

*C. Vocalization (Nikud Prediction)*

G2P is similar to diacritization which is an important task, but nikud data is not scalable unlike audio. To demonstrate our per-character architecture, we evaluated its capacity to

perform standard Hebrew text diacritization *nikud*). While

our primary G2P pipeline scales by leveraging audio su-

pervision, diacritization remains a foundational task in He-

brew natural language processing (NLP) that shares the exact same core structural challenge: resolving highly localized,

context-dependent ambiguities at the grapheme level. Follow- ing the per-character classification methodology established by DictaBERT [7], [Morris: need better explanation - we’re using the same methodology as dicta-bert, cite them]we retained

our pre-trained ReNikud encoder but replaced the phonetic classification heads with a head tailored to directly predict the correct *nikud* class.

This adapted model was fine-tuned on a dataset of 5 million unvocalized sentences sampled from the Knesset corpus that

was used in phonikud training but without the stress, shva and the prefix tags. Phonikud’s data was generated from Dicta Nakdan API. from the 5m sentences we sampled 10k, 25k, 50k and 100k sentences and other 10k sentence for validation. For evaluation we took 100 sentences from nakdimon test and menually fixed the 100 sentences.[Max: how do I need to

explain the why its only 100 sentences? its so small becuase we just want to prove a point no?][Morris: what does that

mean? llm speak]. To generate the ground-truth diacritics for

this subset, we utilized the Dicta Nakdan API[Max: true but

we just used the phonikud data without stress, shva and the prefix tags] [Morris: why? and how is it the Phonikud corpus? where do the GT diacritics come from? is this just distilling dicta-bert?]. We The comparative results against the state-of-the-art 4

DictaBERT-Menaked model are detailed in Table VI.

Despite the limited fine-tuning data[Morris: already said

that], our adapted model achieved an identical Exact Match (EM) score of 0.29 on the test set. At the corpus level,

ReNikud maintained a Diacritic Error Rate (DER) of 2.07%, closely tracking DictaBERT-Menaked’s 1.65%. This confirms

4dicta-il/dictabert-large-char-menaked

Homog.

Foreign Names Stress

Slang ILSpeech-test **OVERALL**

/ 11 / 12

**Model** **WER**

**CER** **DER** **DER** **EM**

(Corpus) (Corpus) (Sample) (Corpus)

DictaBERT-Menaked 10.21 1.68 1.51 1.65 0.29

**V**

**ReNikud (Ours)** 12.34 2.11 1.88 2.07

**0.29**

Table VI: Diacritization performance on the 100 manually cor-

rected Nakdimon benchmark sentences. Metrics are presented as percentages for Word Error Rate (WER), Character Error Rate (CER), Diacritic Error Rate (DER), and Exact Match

(EM). Despite training on a fraction of the data[Morris: how

V

much?], the adapted ReNikud (ReNikud– ReNikud adapted for vocalization) matches the Exact Match baseline.[Morris:

not so convincing other metrics are still worse; can we

compare to fine-tuning DictaBERT directly on the same data? only need one decimal place for percentages e.g. 10.2 not

10.21. Also EM should be percentage with one decimal place]

that our base encoder serves as a robust initialization for

diacritization tasks, successfully absorbing the distilled

knowledge to achieve competitive accuracy. [Morris: not

that convincing–it performs worse and was just distilling

dicta-bert-menaked] To illustrate the quality of the model’s outputs, Table VII

showcases examples of unvocalized input text successfully

mapped to its fully diacritized form by our adapted model.

*D. Ablation and Architectural Comparison* To isolate the contributions of our core design choices and

validate the efficacy of our approach, we designed evaluations to justify two key components: our novel classification archi- tecture and the necessity of audio-derived weak supervision. The comparative results of these evaluations are detailed in Table VIII. **Architectural** **Ablation:** To demonstrate that our inde-

pendent classification head is superior for Hebrew G2P, we compared it against two standard architectures trained on

the identical weakly-supervised Knesset Vox dataset. The

baselines included a sequence-to-sequence approach ByT5-

Small) and a Connectionist Temporal Classification (CTC)

network built on the samedicta-il/dictabert-large-

char base encoder used in our method. [Morris: can we show qualitative examples to illustrate

ReNikud (Ours) 49 ReNikud (Phonikud Data) Phonikud (Baseline) 62  / 30 47  / 5 36 [Morris: error rates should have one decimal place e.g. 49.1 not  data don’t understandthat seq2seq has types of hallucinations that we don’t?]


---

**V**

**Validation** **Set** **(250)** **Test** **Set** **(1,500)**

**Unvocalized** **Input** **ReNikud**

**Dicta**

**Method**

**WER** **CER** **WER** **CER**

ונחנאובעגרהעיגמ ּונְחַנֲאֹוּבעַגֶרָהַעיִּ גַמ

ּונְחַנֲאֹוּבעַגֶרָהַעיִּ גַמ

ָָּּ יַח ֶׁ

ָָּּ יַח ֶׁ

םיבייחשםישיגרמםיִב שםיִׁשיִּ גְרַמ םיִב שםיִׁשיִּ גְרַמ Seq2Seq 24.10 4.94 32.12 11.00

יונישתושעל יּוּנִׁ שתֹוׂשֲעַל יּוּנִׁ שתֹוׂשֲעַל ReNikud (Phonikud Data) 23.36 6.78 32.59 10.98

CTC Loss Network 21.18 3.76 27.86 8.86

| הלשממשארלככ | ֹויְפֹאְוהָלָׁ | שְמֶמׁשאֹרלכָּ | ֹויְפׇאְוהָלָׁ | ׇּ |
|---|---|---|---|---|
| ויפואו |  |  |  |  |

**New** **Classifier** **Head** **(Ours)** **13.71** **2.69** **26.68** **8.48**

| םירדגומםתאםא | םיִרָּ ֶּ | דְגֻמם תַאםִא | םיִרָּ | דְגֻמםֶּ תַאםִא |
|---|---|---|---|---|
| אנאהאושילוצינכ | אָּ | נָאהָאֹוׁשיֵלֹוציִנכְּ | אָּ | נָאהָאֹוׁשיֵלֹוצִנכְּּ |
| ןאכוצחל | ןאָּ | כּוצֲחַל |  | ןאָּ כּוצֲחַל |

VIII: Performance comparison of generation methods.

(WER) and Character Error Rate (CER), evaluated on both a

Table VII: Examples of the adapted ReNikud model curated validation set of 250 sentences and the full held-out

V

(ReNikud – ReNikud adapted for vocalization) acting as atest set of 1,500 sentences.[Morris: only need one decimal

*nakdan*, demonstrating accurate morphological and contextual place in results e.g. 24.1 not 24.10][Morris: maybe remove

diacritization, compared with Dicta. [Morris: Dicta what? val set]

DictaBERT-Menaked? Dicta API? Missing GT. Also results

are not so convincing, it would be better to compare to training DictaBERT on same data, and highlighting words with errorsclassifier architecture is trained solely on the legacy text-

in red (readers will not notice otherwise)]

derived Phonikud dataset, Test WER degrades from 26.68% to 32.59%. This confirms that while reframing G2P as a

direct IPA classification task structurally prevents generative

Qualitative analysis of the predictions highlights a criticalhallucinations, the architecture alone is insufficient.

flaw in the Seq2Seq baseline: autoregressive hallucination. [Max: im not sure how to add the gemini results]

When encountering modern loanwords or slang that use silent [Morris: these are confusing and need more explanation]

vowel letters (*matres lectionis*), the Seq2Seq model frequently

IV. Related Work

hallucinates non-existent syllables and glottal stopsʔ /).(/ For Explicit G2P conversion for Hebrew was recently introduced

example, for the slang הפאכkˈafa/), ByT5 predicts /kaʔafˈa by Kolani et al. [3], following prior works on Hebrew dia-

for רקאהhˈakeʁ/), it outputs haʔˈekeʁ/; and for הקחאד critization [6], [7]. These approaches are bottlenecked by the

dˈaχka/), it generates daʔaχakˈa/. Conversely, our ReNikud availability of annotated textual data, while we use audio as a

architecture structurally prevents these hallucinations.[Max: scalable source of pronunciation information.

By enforcing a constrained, 1-to-1 per-character mapping, the Among prior works learning pronunciation from audio

classifier seamlessly resolves these silent letters to the null in other language settings, we distinguish between two ap-

class, guaranteeing alignment and generating the exact ground proaches:

truth pronunciations.] [Morris: these are not good examples of hallucinations that we prevent, I thought seq2seq sometimes(1) **Audio-supervised** methods such as ours use audio during

predicts things that don’t match the Hebrew characters?] training to improve pronunciation knowledge. Most similar

**Audio** **Supervision** **vs.** **Text** **Data:**To justify our weak to our work, a few studies use audio to improve G2P for

supervision pipeline, we compared the ReNikud architectureEnglish [11], [12] or in multilingual settings [13]. However, trained on our new Knesset Vox audio pseudo-labels againstthese works train on labeled audio as a supplementary signal

the same architecture trained on, text-derived Phonikud datasetto labelled text, while we use large-scale unlabeled audio as

**Training** **Details:** All models were continuously evaluated our primary training signal. Additionally, we operate on the

using a validation set of 250 gold sentences.[Max: what is Hebrew language which has a high degree of orthographic am-

expected from this part?]

biguity and phonetic features unspecified in text (stress, spoken norms, etc.), and our pseudo-vocalization architecture avoids

**Ablation** **Results:** The results in Table VIII confirm the seq2seq hallucinations by enforcing abjad-style alignment.

independent contributions of both our direct phonetic clas-

sification architecture and our weak-supervision data pipeline.(2) **Audio-guided** methods also take audio along with text at

Structurally, our novel IPA triplet classifier achieves the lowestinference time for G2P [14]–[16] or abjad diacritization [17],

error rates across all splits (26.68% Test WER / 8.48% Test[18]. While adding audio as an additional input can provide a

CER), outperforming alternative generation models trained onricher signal, we focus on the case where only text is available

the exact same Vox dataset, namely the Seq2Seq baselineat inference time.

(32.12% Test WER) and the CTC network (27.86% Test

V. Conclusion

WER). This performance gap is most pronounced on the

curated validation set of complex edge-cases, where our IPA We introduced ReNikud, a*Pseudo-Vocalization* architecture

classifier drops WER to 13.71%, compared to 21.18% for thefor Hebrew G2P conversion. By reframing the generative

CTC approach and 24.10% for Seq2Seq.[Morris: don’t need to sequence-to-sequence problem into a per-character phonetic

repeat all numbers from tables, text is for high-level analysis]triplet classification task (Consonant, Vowel, Stress), our

Furthermore, the dataset ablation isolates the specific impactmethod structurally mitigates the generative hallucinations

Table Error rates are reported as percentages for Word Error Rateof the audio supervision. When the exact same ReNikud IPAobserved in standard Seq2Seq models.[Morris: this paragraph

---

should be changed, renikud is also a method for using audio, not just the architecture] [Morris: change claim to be about reducing degrees of freedom] [Yakov: Add a mention about the limitation where, due to

our strict alignment, the aligner might drop some informal variants such as , which might be pronounced as jixtov.] To train this architecture, we developed a weak-supervision

pipeline utilizing a monotonic FST to align continuous IPA transcriptions from an adapted Whisper ASR model with

unvocalized Hebrew graphemes. Our evaluations demonstrate

that ReNikud lowers Word Error Rates compared to the legacy Phonikud baseline and the evaluated Seq2Seq and CTC archi- tectures [Morris: phrase that part better], particularly on com-

plex edge-cases such as colloquial slang, rare phonemes, and penultimate stress. Furthermore, we validated the extensibility of our core character-level encoder by successfully adapting it for Hebrew text diacritization via knowledge distillation, using only a fraction of standard training data volumes.[Morris: llm

speak]

**Limitations** **and** **Future** **Work:**A primary limitation of our

methodology is its reliance on ASR-generated pseudo-labels, as inherent transcription errors from the ASR model naturally propagate into the G2P training data. Additionally, because the Knesset Vox dataset consists exclusively of parliamentary speeches, the training corpus carries a distinct formal and

demographic bias. [Morris: LLM speak] This restricts the

model’s exposure to diverse, casual, and second-person con- versational phonology—a domain shift that accounts for the observed performance gap in resolving gendered morphology compared to legacy models trained on broader literary text. [Morris: are you sure that’s true?]Future work will focus

on expanding our weak-supervision pipeline to incorporate

more diverse, unstructured conversational audio corpora to

address these domain-specific biases. Furthermore, because

the core challenge of unwritten vowels is shared across other abjad writing systems (such as Arabic), we plan to extend the *Pseudo-Vocalization*architecture to these languages to evaluate

its cross-lingual generalizability.[Max: something do to with תובונגע,ח?] [Morris: what do they limit?]

References

[1] P. T. Daniels and W. Bright,*The world’s writing systems*. Oxford

University Press, 1996.

[2] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu,

A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi*et al.* , “Scaling speech

technology to 1,000+ languages,” *Journal of Machine Learning Re-*

*search*, vol. 25, no. 97, pp. 1–52, 2024.

[3] Y. Kolani, M. Melichov, C. Calev, and M. Alper, “Phonikud: Hebrew grapheme-to-phoneme conversion for real-time text-to-speech,”*arXiv*

*preprint arXiv:2506.12311*, 2025.

[4] A. Aharoni, “Vocalization of modern hebrew,” in*Encyclopedia of*

*Hebrew Language and Linguistics*, G. Khan, S. Bolozky, S. E. Fassberg,

G. A. Rendsburg, A. D. Rubin, O. Schwarzwald, and T. Zewi, Eds. Leiden: Brill, 2013, vol. 3, pp. 944–951. [5] H. Neudecker, “Vocalization of modern hebrew and colloquial pro- nunciation,” in *Encyclopedia of Hebrew Language and Linguistics*

G. Khan, S. Bolozky, S. E. Fassberg, G. A. Rendsburg, A. D. Rubin,
O. Schwarzwald, and T. Zewi, Eds. Leiden: Brill, 2013, vol. 3, pp.

951–953. [6] E. Gershuni and Y. Pinter, “Restoring hebrew diacritics without a dic- tionary,” in *Findings of the Association for Computational Linguistics:*

*NAACL 2022*, 2022, pp. 1010–1018.

[7] S. Shmidman, A. Shmidman, and M. Koppel, “Dictabert: A state-of- the-art bert suite for modern hebrew,” 2023. [8] J. Zhu, C. Zhang, and D. Jurgens, “Byt5 model for massively multi- lingual grapheme-to-phoneme conversion,” in*Proc. Interspeech 2022*

2022, pp. 446–450. [9] O. Sharoni, R. Shenberg, and E. Cooper, “Saspeech: A hebrew single speaker dataset for text to speech and voice conversion,” *Proc.* in

*Interspeech*, 2023.

[10] Y. Marmor, A. Zulti, D. Krongauz, A. Gabet, Y. Snapir, Y. Lifshitz, and E. Segal, “Voxknesset: A large-scale longitudinal hebrew speech dataset for aging speaker modeling,” 2026. [Online]. Available:

[https://arxiv.org/abs/2603.01270](https://arxiv.org/abs/2603.01270)

[11] S. Sun, K. Richmond, and H. Tang, “Improving seq2seq tts frontends with transcribed speech audio,” *IEEE/ACM Transactions on Audio,*

*Speech, and Language Processing*, vol. 31, pp. 1940–1952, 2023.

[12] S. Sun and K. Richmond, “Acquiring pronunciation knowledge from transcribed speech audio via multi-task learning,” 2024. [Online].

Available: [https://arxiv.org/abs/2409.09891](https://arxiv.org/abs/2409.09891)

[13] M. S. Ribeiro, G. Comini, and J. Lorenzo-Trueba, “Improving grapheme- to-phoneme conversion by learning pronunciations from speech record- ings,” *arXiv preprint arXiv:2307.16643*, 2023.

[14] J. Route, S. Hillis, I. C. Etinger, H. Zhang, and A. W. Black, “Multi- modal, multilingual grapheme-to-phoneme conversion for low-resource languages,” in *Proceedings of the 2nd Workshop on Deep Learning*

*Approaches for Low-Resource NLP (DeepLo 2019)*, 2019, pp. 192–201.

[15] H. Gao, M. Hasegawa-Johnson, and C. D. Yoo, “G2pu: grapheme-to- phoneme transducer with speech units,” in*ICASSP 2024-2024 IEEE*

*International Conference on Acoustics, Speech and Signal Processing* *(ICASSP)*. IEEE, 2024, pp. 10 061–10 065.

[16] C.-J. Li, K. Chang, S. Bharadwaj, E. Yeo, K. Choi, J. Zhu, D. Mortensen, and S. Watanabe, “Powsm: A phonetic open whisper-style speech

foundation model,” *arXiv preprint arXiv:2510.24992*, 2025.

[17] S. Shatnawi, S. Alqahtani, and H. Aldarmaki, “Automatic restoration of diacritics for speech data sets,” 2024. [Online]. Available:

[https://arxiv.org/abs/2311.10771](https://arxiv.org/abs/2311.10771)

[18] A. Ghannam, N. Alharthi, F. Alasmary, K. Al Tabash, S. Sadah, and

L. Ghouti, “Abjad ai at nadi 2025: Catt-whisper: Multimodal diacritic restoration using text and speech representations,” pp. 757–761, 2025.