# Cloned Repositories

## Repo 1: future-lens
- **URL**: https://github.com/KoyenaPal/future-lens
- **Purpose**: Official implementation of Future Lens (CoNLL 2023). Core method for predicting subsequent tokens from a single hidden state.
- **Location**: code/future-lens/
- **Key files**:
  - `causal_methods/train.py` - Train soft prompt for learned prompt intervention
  - `causal_methods/test.py` - Test learned prompt predictions
  - `linear_methods/linear_hs.py` - Train linear models for hidden state prediction
  - `demo/FutureLensDemonstration.ipynb` - Interactive demo notebook
  - `data/` - Training/test data (Pile samples)
- **Dependencies**: nnsight, torch, transformers (see scripts/colab-reqs/future-env.txt)
- **Model**: GPT-J-6B
- **Notes**: Updated Oct 2025 to use NNsight 0.5.9. Key method: learned prompt causal intervention at middle layers (~layer 14) achieves best prediction of future tokens.

## Repo 2: footprints (Token Erasure)
- **URL**: https://github.com/sfeucht/footprints
- **Purpose**: Token erasure detection and implicit vocabulary reading for LLMs (EMNLP 2024).
- **Location**: code/footprints/
- **Key files**:
  - `scripts/segment.py` - Segment documents into lexical items using erasure score
  - `scripts/readout.py` - Read out implicit vocabulary from a dataset
  - `scripts/train_probe.py` - Train linear probes on hidden states
  - `scripts/test_probe.py` - Test probes on various datasets
  - `data/` - CounterFact, Wikipedia, and Pile datasets
- **Dependencies**: nnsight, torch, transformers (see requirements.txt)
- **Models**: Llama-2-7b, Llama-3-8b
- **Pre-trained probes**: Available at https://huggingface.co/sfeucht/footprints
- **Notes**: The erasure score ψ identifies multi-token sequences that the model treats as single lexical units. Directly applicable to finding non-compositional phrases. Uses probes at layers 1 and 9.

## Repo 3: idiomem
- **URL**: https://github.com/adihaviv/idiomem
- **Purpose**: Dataset and experiments for studying transformer memorization recall through idioms (EACL 2023).
- **Location**: code/idiomem/
- **Key files**:
  - `idiomem.jsonl` - 852 English idioms with memorization metadata
  - `run_experiments.py` - Reproduce paper's memorization analysis
  - `idioms_dataset_collector.py` - Collect idiom datasets
  - `knockouts.py` - FFN neuron intervention experiments
- **Dependencies**: transformers, torch
- **Notes**: Provides ground-truth non-compositional phrases. The `hard_to_guess` field indicates idioms that truly require memorization (not guessable from constituents).
