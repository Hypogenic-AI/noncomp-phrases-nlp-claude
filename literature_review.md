# Literature Review: Finding Non-Compositional Phrases via Future Lens

## Research Area Overview

This research sits at the intersection of three areas: (1) mechanistic interpretability of LLMs, specifically how hidden states encode multi-token information; (2) memorization in language models; and (3) non-compositional/idiomatic language processing. The hypothesis is that the Future Lens method can serve as a lower bound detector for phrases that LLMs have memorized as non-compositional units, since predictability of future tokens from a single hidden state indicates the model has encoded the phrase as a single unit rather than composing it from parts.

## Key Papers

### 1. Future Lens (Pal et al., 2023) — Core Method

**Key Contribution**: Demonstrates that individual hidden states in GPT-J-6B encode information about tokens 2+ positions ahead, not just the immediate next token.

**Methodology**:
- Four methods tested: (a) Direct vocabulary prediction via linear model, (b) Linear model approximation (predict future hidden state, decode with pretrained head), (c) Fixed prompt causal intervention (transplant hidden state into unrelated context), (d) Learned prompt causal intervention (optimize soft prompt to extract future token info).
- Learned prompt method is best: 48.4% accuracy at N=1 (one token ahead of next), peaking at middle layers (~layer 14 of 28 in GPT-J).
- Bigram baseline: 20.1% accuracy at N=1.

**Key Results**:
- Future token prediction peaks at **middle layers**, unlike N=0 which peaks at last layer.
- Accuracy correlates with model confidence: 95% accuracy when model confidence is 90-100%.
- Named entities show similar (not higher) accuracy (44%, 42%, 37% for N=1,2,3), suggesting future info is encoded broadly.
- Optimizing for N=1 generalizes well to N=2,3; optimizing for other N does not generalize back to N=1.

**Relevance**: If a hidden state at token t can predict tokens t+2, t+3, etc., this indicates the model has "pre-computed" the entire phrase as a unit. For non-compositional phrases (idioms, named entities), this pre-computation is expected because the model cannot derive the continuation compositionally.

**Datasets Used**: The Pile (100K training tokens, 1K test tokens).
**Code**: https://github.com/KoyenaPal/future-lens

### 2. Token Erasure (Feucht et al., 2024) — Complementary Method

**Key Contribution**: Discovers that last token positions of multi-token words and named entities exhibit "erasure" — information about previous tokens is rapidly forgotten in early layers. Uses this to build an implicit vocabulary reader.

**Methodology**:
- Train linear probes at each layer to predict nearby tokens (offsets -3 to +1).
- Test on COUNTERFACT subjects: last tokens of multi-token entities show dramatic drop in ability to predict preceding tokens, suggesting the model overwrites token-level info with entity-level semantics.
- Same pattern found for multi-token words in Wikipedia.
- Propose erasure score ψ that measures probability drop between layer 1 and layer 9 for predicting within-sequence tokens.
- Algorithm 1 segments documents into high-scoring non-overlapping lexical items.

**Key Results**:
- Llama-2-7b: ~1800 sequences recovered from 500 Wikipedia articles; 44.9% are multi-token words or entities.
- Llama-3-8b (larger vocab): fewer multi-token words recovered, but finds multi-word expressions, LaTeX commands.
- Highest-scoring sequences are plausible non-compositional lexical items.

**Relevance**: The erasure score provides an alternative/complementary signal to Future Lens for detecting non-compositional phrases. Where Future Lens detects "forward prediction" (can predict tokens ahead), Token Erasure detects "backward forgetting" (forgets constituent tokens). Both indicate the model treats the phrase as a single unit.

**Datasets Used**: The Pile, COUNTERFACT, Wikipedia.
**Code**: https://github.com/sfeucht/footprints

### 3. Transformer Memorization via Idioms (Haviv et al., 2022)

**Key Contribution**: First framework for probing memorization recall in transformers using idioms. Discovers two-phase memorization: early-layer candidate promotion + late-layer confidence boosting.

**Methodology**:
- Define criteria for memorization detection: (1) single target independent of context, (2) irreducible prompt (need entire sequence).
- Construct IdioMem dataset: 852 English idioms with metadata.
- Split idioms into memorized/non-memorized sets per model.
- Track predicted token rank and probability across layers using logit lens interpretation.

**Key Results**:
- Memorized idioms: target token promoted to top of distribution by early layers (candidate promotion), then probability sharply increases in final layers (confidence boosting).
- Non-memorized inputs: gradual, less structured prediction construction.
- Zeroing out FFN neurons in early layers disrupts memorization recall; late-layer intervention doesn't.
- Pattern holds across GPT-2 sizes and GPT-Neo architectures.

**Relevance**: Provides ground-truth non-compositional phrases (idioms) for validation. The two-phase memorization profile could serve as a detection signal complementary to Future Lens predictions.

**Datasets Used**: IdioMem (852 idioms), LAMA-UHN (factual statements).
**Code**: https://github.com/adihaviv/idiomem

### 4. MWE Semantics in Transformers Survey (Miletic & Schulte im Walde, 2024)

**Key Finding**: Transformers capture MWE semantics inconsistently, relying on surface patterns and memorized information. MWE meaning is predominantly localized in early layers. Representations benefit from lower semantic idiosyncrasy and ambiguity.

**Relevance**: Confirms that non-compositional processing happens in early layers, consistent with both Future Lens (middle layer peak) and Token Erasure (early layer erasure).

### 5. Frequent Phrases vs. Idioms (Rambelli et al., 2023)

**Key Finding**: Idioms and high-frequency compositional phrases are processed similarly by both humans and neural language models (measured via surprisal). This suggests frequency-based retrieval may confound non-compositionality detection.

**Relevance**: Important caveat — Future Lens predictions may capture both non-compositional phrases AND highly frequent compositional phrases. Need to distinguish between these cases, possibly using compositionality annotations as ground truth.

## Common Methodologies

1. **Logit Lens / Tuned Lens**: Interpreting hidden states as probability distributions over vocabulary by applying the pretrained decoder head. Used in Haviv et al. and related to Future Lens linear methods.
2. **Causal Intervention**: Transplanting hidden states between contexts to test what information they encode (Future Lens, ROME).
3. **Linear Probing**: Training linear classifiers on hidden states to test what information is accessible (Token Erasure).
4. **Erasure/Knockout**: Zeroing or modifying specific components to test necessity (Haviv et al. FFN knockouts).

## Standard Baselines

- **Bigram frequency**: Simple co-occurrence statistics (Future Lens reports 20.1% bigram baseline at N=1).
- **Logit Lens (N=0)**: Standard next-token prediction from hidden states.
- **Random baseline**: Chance-level prediction (~0.002% for 50K vocabulary).

## Evaluation Metrics

- **Precision@k**: Whether the correct token appears in top-k predictions.
- **Surprisal**: Negative log probability of predicted token under the model.
- **Erasure score ψ**: Drop in probe accuracy between early and middle layers (Token Erasure).
- **Compositionality correlation**: Agreement with human compositionality ratings.

## Datasets in the Literature

| Dataset | Papers | Task |
|---------|--------|------|
| The Pile | Future Lens, Token Erasure | General text, training data |
| COUNTERFACT | Token Erasure | Named entity factual recall |
| IdioMem | Haviv et al. | 852 English idioms for memorization probing |
| LAMA-UHN | Haviv et al. | Factual statements |
| Farahmand NCS | (external) | 1042 noun compounds with compositionality labels |
| MAGPIE | (external) | 44K sentences with idiom annotations |
| Reddy et al. | (external) | 90 noun compounds with compositionality scores |

## Gaps and Opportunities

1. **No direct application of Future Lens to non-compositionality detection**: The original paper mentions named entities but does not systematically evaluate on known non-compositional phrases.
2. **Future Lens + Token Erasure combination**: Both methods detect related signals (forward prediction vs. backward erasure) but have not been combined.
3. **Compositionality scoring**: Future Lens accuracy or confidence could serve as a continuous compositionality score, which can be validated against human annotations.
4. **Scale beyond GPT-J**: Future Lens was only tested on GPT-J-6B; Token Erasure used Llama models. Testing on modern models would be valuable.
5. **Distinguishing frequency from non-compositionality**: Rambelli et al. show both are processed similarly; need a method to tease them apart.

## Recommendations for Experiment

### Recommended Approach
1. **Apply Future Lens to known non-compositional phrases** (IdioMem idioms, Farahmand non-compositional compounds) and measure whether prediction accuracy is higher than for compositional phrases.
2. **Compare Future Lens predictions with Token Erasure scores** on the same phrases to see if the signals correlate.
3. **Use compositionality-annotated datasets** (Farahmand, Reddy) as ground truth to validate that Future Lens accuracy correlates with non-compositionality.

### Recommended Datasets
- **Primary**: IdioMem (ground-truth idioms), Farahmand NCS (compositionality labels)
- **Secondary**: COUNTERFACT (named entities), Wikipedia text (general)
- **Baseline text**: Pile samples (compositional text)

### Recommended Baselines
- Bigram/n-gram frequency statistics
- Standard logit lens (N=0 only)
- Random/uniform baseline

### Recommended Metrics
- Future Lens accuracy (Precision@1) at N=1, N=2, N=3 for non-compositional vs. compositional phrases
- Correlation between Future Lens confidence and human compositionality ratings
- Token Erasure score comparison between phrase types
- ROC/AUC for binary non-compositionality classification using Future Lens as a feature

### Methodological Considerations
- Use smaller models (GPT-2, Pythia) for feasibility without GPU cluster
- The learned prompt method requires per-layer training; linear methods are cheaper
- Ensure phrases appear in training data to distinguish non-compositionality from unfamiliarity
- Control for phrase frequency to separate memorization-by-frequency from memorization-by-non-compositionality
