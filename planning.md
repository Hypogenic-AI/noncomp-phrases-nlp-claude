# Research Plan: Finding Non-Compositional Phrases via Future Lens

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs memorize certain multi-token sequences as single units — idioms ("kick the bucket"), named entities ("New York"), and other non-compositional phrases. Identifying these phrases is important for understanding LLM internal representations, improving tokenization, detecting memorized training data, and understanding compositional vs. non-compositional language processing. Currently, there is no scalable method to enumerate what an LLM has memorized as a single unit.

### Gap in Existing Work
Future Lens (Pal et al., 2023) showed that hidden states encode future token information, but never applied this to non-compositionality detection. Token Erasure (Feucht et al., 2024) detects backward forgetting but doesn't directly measure forward predictability. Haviv et al. (2022) studied idiom memorization but used logit lens, not Future Lens probes. No one has used Future Lens prediction accuracy as a compositionality score or validated it against human compositionality annotations.

### Our Novel Contribution
We test whether Future Lens prediction accuracy (the ability of a hidden state at position t to predict tokens at t+2, t+3) serves as a lower bound detector for non-compositional phrases. We validate this using the Farahmand NCS dataset (1042 compounds with compositionality labels) and the IdioMem dataset (852 idioms). This is the first systematic evaluation of Future Lens as a non-compositionality metric.

### Experiment Justification
- **Experiment 1 (Linear Probe Training)**: Train Future Lens probes on general text to establish the baseline capability of hidden states to predict future tokens.
- **Experiment 2 (Compositionality Classification)**: Apply probes to compositional vs. non-compositional phrases from Farahmand NCS. If non-compositional phrases have higher Future Lens accuracy, this supports the hypothesis.
- **Experiment 3 (Idiom Detection)**: Validate on IdioMem idioms as a second dataset of known non-compositional phrases.
- **Experiment 4 (Frequency Control)**: Control for phrase frequency to separate "memorized because frequent" from "memorized because non-compositional."

## Research Question
Can Future Lens prediction accuracy serve as a lower bound for detecting non-compositional phrases memorized by LLMs?

## Background and Motivation
See literature_review.md for full details. Key insight: if a hidden state at token t can predict token t+k, the model has pre-encoded the phrase as a unit. For non-compositional phrases, this pre-encoding is expected because the continuation cannot be derived compositionally.

## Hypothesis Decomposition
1. **H1**: Future Lens prediction accuracy at the first token of a phrase is higher for non-compositional than compositional phrases.
2. **H2**: The accuracy gap persists after controlling for phrase frequency.
3. **H3**: Future Lens accuracy correlates with human compositionality ratings.
4. **H4**: Known idioms (IdioMem) show higher Future Lens accuracy than random bigrams.

## Proposed Methodology

### Approach
Use a pre-trained LLM (GPT-2-xl, 1.5B params) to:
1. Train linear probes mapping hidden_state[layer, position t] → hidden_state[layer, position t+k]
2. Apply probes to phrases from compositionality-annotated datasets
3. Compare prediction accuracy across compositional vs. non-compositional phrases

GPT-2-xl is chosen for fast iteration; results can be validated on larger models.

### Experimental Steps
1. Load GPT-2-xl, extract hidden states from Wikipedia text for probe training
2. Train linear probes for k=1,2,3 (predicting 1,2,3 tokens ahead) at each layer
3. Prepare Farahmand NCS compounds in sentence contexts
4. Prepare IdioMem idioms (use idiom text directly)
5. Evaluate probes on all phrase datasets
6. Compare Future Lens accuracy: non-compositional vs. compositional
7. Compute ROC/AUC for binary classification
8. Control for frequency

### Baselines
- **Bigram frequency**: PMI or raw co-occurrence frequency
- **Model perplexity**: Standard next-token prediction confidence (not Future Lens)
- **Random baseline**: Chance-level prediction

### Evaluation Metrics
- **Precision@1**: Whether probe's top prediction matches actual future token
- **Mean probability**: Average probability assigned to the correct future token
- **ROC AUC**: Binary classification of non-compositional vs. compositional
- **Pearson correlation**: Between Future Lens accuracy and compositionality scores

### Statistical Analysis Plan
- Two-sample t-test or Mann-Whitney U for comparing accuracy distributions
- Bootstrap confidence intervals for AUC
- Significance level: α = 0.05
- Effect size: Cohen's d

## Expected Outcomes
- Non-compositional phrases should have higher Future Lens accuracy (supporting H1)
- The effect should be strongest at middle layers (~layer 24 of 48 for GPT-2-xl)
- AUC for binary classification should be significantly above 0.5

## Timeline and Milestones
1. Environment setup + data preparation: 20 min
2. Hidden state extraction + probe training: 30 min
3. Evaluation on phrase datasets: 30 min
4. Analysis and visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- Frequency confound: frequent compositional phrases may also have high accuracy
- Tokenization: some phrases may tokenize into many subword tokens
- Context sensitivity: probe accuracy may depend on surrounding context
- GPT-2's training data may not contain all target phrases

## Success Criteria
- Statistically significant difference in Future Lens accuracy between compositional and non-compositional phrases
- AUC > 0.6 for binary non-compositionality classification
- Clear visualization showing the separation
