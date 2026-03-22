# Downloaded Papers

## Core Papers (Deep Read)

1. **Future Lens: Anticipating Subsequent Tokens from a Single Hidden State** (2311.04897_future_lens.pdf)
   - Authors: Koyena Pal, Jiuding Sun, Andrew Yuan, Byron C. Wallace, David Bau
   - Year: 2023 (CoNLL 2023)
   - arXiv: 2311.04897
   - Why relevant: THE core method paper. Shows hidden states encode future tokens; learned prompt intervention achieves 48% accuracy predicting N+1 tokens from a single hidden state at middle layers.

2. **Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs** (2406.20086_token_erasure.pdf)
   - Authors: Sheridan Feucht, David Atkinson, Byron C. Wallace, David Bau
   - Year: 2024 (EMNLP 2024)
   - arXiv: 2406.20086
   - Why relevant: Direct follow-up from same lab. Discovers "erasure" effect where multi-token words/entities lose token-level info in early layers. Proposes erasure score to identify implicit vocabulary items. Directly applicable to finding non-compositional phrases.

3. **Understanding Transformer Memorization Recall Through Idioms** (2210.03588_transformer_memorization_idioms.pdf)
   - Authors: Adi Haviv, Ido Cohen, Jacob Gidron, Roei Schuster, Yoav Goldberg, Mor Geva
   - Year: 2022 (EACL 2023)
   - arXiv: 2210.03588
   - Why relevant: Provides IdioMem dataset (852 idioms) and methodology for detecting memorization recall. Shows memorization is a two-step process: early-layer candidate promotion + late-layer confidence boosting. Provides ground-truth non-compositional phrases.

## Supporting Papers

4. **Memorization Without Overfitting** (2310.04691_memorization_without_overfitting.pdf)
   - Authors: Kushal Tirumala et al.
   - Year: 2022
   - Why relevant: Studies training dynamics of memorization in LLMs.

5. **SoK: Memorization in General-Purpose LLMs** (2310.09638_sok_memorization_llms.pdf)
   - Authors: Valentin Hartmann et al.
   - Year: 2023
   - Why relevant: Comprehensive taxonomy of memorization types in LLMs.

6. **Semantics of Multiword Expressions in Transformer-Based Models: A Survey** (2401.12299_mwe_transformer_survey.pdf)
   - Authors: Filip Miletic, Sabine Schulte im Walde
   - Year: 2024
   - Why relevant: Survey of MWE processing in transformers. Finds meaning is localized in early layers.

7. **It's not Rocket Science: Interpreting Figurative Language in Narratives** (2109.03400_figurative_language_narratives.pdf)
   - Authors: Tuhin Chakrabarty, Yejin Choi, V. Shwartz
   - Year: 2021
   - Why relevant: Non-compositional figurative language (idioms, similes) interpretation.

8. **Transformer Feed-Forward Layers Are Key-Value Memories** (2012.04905_ff_layers_key_value_memories.pdf)
   - Authors: Mor Geva, Roei Schuster, Jonathan Berant, Omer Levy
   - Year: 2020
   - Why relevant: Foundational mechanistic understanding of how transformers store knowledge.

9. **Are Frequent Phrases Directly Retrieved like Idioms?** (2309.13415_frequent_phrases_idioms.pdf)
   - Authors: Giulia Rambelli et al.
   - Year: 2023
   - Why relevant: Compares processing of idioms vs. high-frequency compositional phrases in NLMs.

10. **Memorization or Reasoning? Exploring Idiom Understanding of LLMs** (2501.00794_memorization_or_reasoning_idioms.pdf)
    - Authors: Jisu Kim et al.
    - Year: 2025
    - Why relevant: MIDAS dataset of idioms in 6 languages; hybrid memorization+reasoning in LLMs.

11. **Non-compositional Expression Generation** (2310.10013_non_comp_expression_generation.pdf)
    - Authors: Jianing Zhou et al.
    - Year: 2023
    - Why relevant: Curriculum learning for non-compositional expression generation.

12. **Demystifying Verbatim Memorization in LLMs** (2407.17817_demystifying_verbatim_memorization.pdf)
    - Authors: Jing Huang, Diyi Yang, Christopher Potts
    - Year: 2024
    - Why relevant: Studies mechanisms of verbatim memorization; finds it's intertwined with general capabilities.
