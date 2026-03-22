# Finding Non-Compositional Phrases via Future Lens

Can we use the Future Lens method to identify phrases that LLMs have memorized as non-compositional units?

## Key Findings

- **Non-compositional compounds are more predictable**: On the Farahmand NCS dataset (1042 compounds, expert-annotated), non-compositional compounds show significantly higher within-phrase predictability than compositional ones (p<0.0001, Cohen's d=0.49 for surprisal).
- **ROC AUC = 0.643** for binary non-compositionality classification using next-token probability.
- **Spearman r = 0.23** (p<0.0001) between Future Lens prediction metrics and human compositionality scores.
- **Idioms are dramatically more predictable** than control bigrams (d=1.85), confirming memorization as single units.
- **The within-phrase signal outperforms the from-first-token signal**, especially for longer phrases like idioms.
- **Non-compositional advantage appears from layer 6 onward** in GPT-2-XL's 48-layer architecture.

## Conclusion

Future Lens metrics provide a **statistically significant but moderate** signal for non-compositionality detection. The method works as a noisy lower bound: high predictability strongly suggests memorization as a unit, but the AUC of 0.64 means it misses many non-compositional phrases.

## Reproduction

```bash
# Set up environment
uv venv && source .venv/bin/activate
uv add torch transformers numpy pandas scikit-learn matplotlib seaborn scipy tqdm

# Run experiment (~2 min on A6000 GPU)
python src/future_lens_v3.py
```

## Project Structure

```
├── REPORT.md                          # Full research report with results
├── planning.md                        # Research plan and hypothesis decomposition
├── src/
│   └── future_lens_v3.py             # Main experiment script
├── results/
│   ├── future_lens_v3_results.csv    # Raw per-phrase metrics (1949 phrases)
│   └── statistical_results_v3.csv    # Statistical test results
├── figures/
│   ├── metrics_by_category.png       # Box plots of key metrics
│   ├── layer_profiles.png            # Logit lens across layers
│   ├── compositionality_scatter.png  # Correlation with human scores
│   ├── noncomp_advantage.png         # Layer-by-layer advantage
│   └── distribution_comparison.png   # Idiom vs control distributions
├── datasets/                          # Pre-downloaded datasets
├── papers/                            # Reference papers
├── code/                              # Reference implementations
└── literature_review.md               # Literature synthesis
```

## Model & Environment

- **Model**: GPT-2-XL (1.5B params, 48 layers)
- **GPU**: NVIDIA RTX A6000 (49 GB)
- **Runtime**: ~93 seconds for full evaluation
- **Python**: 3.12.8, PyTorch 2.10.0, Transformers 5.3.0
