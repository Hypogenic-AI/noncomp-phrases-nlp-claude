# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on finding non-compositional phrases using the Future Lens method. The hypothesis is that Future Lens can lower-bound the set of phrases LLMs have memorized as non-compositional units, since predictability in advance from a hidden state indicates likely memorization as a single unit.

## Papers
Total papers downloaded: 12

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Future Lens | Pal, Sun, Yuan, Wallace, Bau | 2023 | papers/2311.04897_future_lens.pdf | Core method: predict future tokens from hidden states |
| Token Erasure | Feucht, Atkinson, Wallace, Bau | 2024 | papers/2406.20086_token_erasure.pdf | Erasure score for implicit vocabulary items |
| Transformer Memorization via Idioms | Haviv et al. | 2022 | papers/2210.03588_transformer_memorization_idioms.pdf | IdioMem dataset, two-phase memorization |
| Memorization Without Overfitting | Tirumala et al. | 2022 | papers/2310.04691_memorization_without_overfitting.pdf | Training dynamics of memorization |
| SoK: Memorization in LLMs | Hartmann et al. | 2023 | papers/2310.09638_sok_memorization_llms.pdf | Taxonomy of memorization types |
| MWE Semantics in Transformers | Miletic & Schulte im Walde | 2024 | papers/2401.12299_mwe_transformer_survey.pdf | Survey: MWE processing localized in early layers |
| Figurative Language in Narratives | Chakrabarty et al. | 2021 | papers/2109.03400_figurative_language_narratives.pdf | Non-compositional language interpretation |
| FF Layers as Key-Value Memories | Geva et al. | 2020 | papers/2012.04905_ff_layers_key_value_memories.pdf | Foundational: how transformers store knowledge |
| Frequent Phrases vs Idioms | Rambelli et al. | 2023 | papers/2309.13415_frequent_phrases_idioms.pdf | Frequency vs non-compositionality in processing |
| Memorization or Reasoning (Idioms) | Kim et al. | 2025 | papers/2501.00794_memorization_or_reasoning_idioms.pdf | MIDAS dataset, hybrid processing |
| Non-comp Expression Generation | Zhou et al. | 2023 | papers/2310.10013_non_comp_expression_generation.pdf | Curriculum learning for non-comp generation |
| Demystifying Verbatim Memorization | Huang et al. | 2024 | papers/2407.17817_demystifying_verbatim_memorization.pdf | Memorization intertwined with general capabilities |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 6

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| IdioMem | GitHub (adihaviv) | 852 idioms | Memorization probing | datasets/idiomem.jsonl | Ground-truth non-compositional phrases |
| CounterFact Expanded | GitHub (sfeucht) | 54K prompts | Entity knowledge | datasets/counterfact_expanded.csv | Named entity factual recall |
| Wikipedia Test 500 | GitHub (sfeucht) | 500 articles | General text probing | datasets/wikipedia_test_500.csv | Multi-token word analysis |
| Future Lens Test Data | GitHub (KoyenaPal) | 1K samples | Future token prediction | datasets/testing_data_teacher_1000.csv | Pile samples with predictions |
| Farahmand NCS | GitHub (meghdadFar) | 1042 compounds | Compositionality labels | datasets/farahmand_ncs/ | Expert binary annotations |
| IdioTS | HuggingFace | 195 examples | Idiom comprehension | datasets/idiots/ | English + Spanish idioms |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| future-lens | github.com/KoyenaPal/future-lens | Core Future Lens implementation | code/future-lens/ | GPT-J-6B, uses nnsight |
| footprints | github.com/sfeucht/footprints | Token erasure & implicit vocabulary | code/footprints/ | Llama models, pre-trained probes on HF |
| idiomem | github.com/adihaviv/idiomem | Idiom memorization experiments | code/idiomem/ | 852 idioms dataset + analysis |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service (diligent mode) with query: "Future Lens non-compositional phrases language model memorization"
- 44 papers returned, filtered to 12 most relevant
- Focused on: (1) the Future Lens method itself, (2) token erasure/implicit vocabulary, (3) idiom/MWE processing in transformers, (4) memorization mechanisms in LLMs
- Searched for compositionality-annotated datasets via web search

### Selection Criteria
- Papers directly related to the core method (Future Lens, Token Erasure) were prioritized
- Papers providing ground-truth datasets for non-compositional phrases (idioms, MWEs) were included
- Papers on general memorization included for theoretical grounding
- Only papers with available code/data preferred for reproducibility

### Challenges Encountered
- MAGPIE HuggingFace dataset uses deprecated dataset scripts; could not be auto-downloaded
- No single dataset combines compositionality scores with the specific format needed for Future Lens evaluation
- Future Lens was only tested on GPT-J-6B; applying to other models requires adaptation

### Gaps and Workarounds
- **Missing**: A ready-made dataset of phrases with both compositionality scores AND appearance in LLM training data. Workaround: use IdioMem (known non-compositional) + Farahmand NCS (compositionality labels) + general text (compositional baseline).
- **Missing**: Pre-trained Future Lens probes for models other than GPT-J. Workaround: use linear methods (cheaper to train) or adapt existing footprints probes.

## Recommendations for Experiment Design

### Primary Experiment: Future Lens as Non-Compositionality Detector
1. **Primary datasets**: IdioMem (idioms = non-compositional) + Farahmand NCS (labeled compositionality) + Wikipedia text (compositional baseline)
2. **Baseline methods**: Bigram frequency, logit lens (N=0), random baseline
3. **Evaluation metrics**: Precision@1 for Future Lens predictions on non-comp vs. comp phrases; correlation with compositionality ratings; ROC/AUC for binary classification
4. **Code to adapt/reuse**:
   - `code/future-lens/` for linear model approximation and causal intervention
   - `code/footprints/` for erasure score computation (complementary signal)
   - `code/idiomem/` for ground-truth idioms and memorization analysis framework

### Recommended Model
- Start with GPT-2 or Pythia (smaller, faster iteration)
- Future Lens linear methods are cheapest to train
- Token Erasure probes are available pre-trained for Llama-2-7b on HuggingFace

### Key Research Questions
1. Does Future Lens prediction accuracy correlate with non-compositionality?
2. Can erasure score + Future Lens accuracy together improve non-compositionality detection?
3. What is the precision/recall tradeoff when using Future Lens as a binary non-compositionality classifier?
