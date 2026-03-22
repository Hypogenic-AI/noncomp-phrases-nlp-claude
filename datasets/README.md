# Datasets

This directory contains datasets for the research project on finding non-compositional phrases using the Future Lens method. Data files are NOT committed to git due to size.

## Dataset 1: IdioMem (Idiom Memorization Dataset)

### Overview
- **Source**: https://github.com/adihaviv/idiomem
- **Size**: 852 English idioms
- **Format**: JSONL
- **Task**: Probing memorization recall in LMs via idiom completion
- **License**: Research use

### Download Instructions
```bash
git clone https://github.com/adihaviv/idiomem
cp idiomem/idiomem.jsonl datasets/
```

### Loading
```python
import json
idioms = [json.loads(line) for line in open("datasets/idiomem.jsonl")]
```

### Notes
- Each entry has: idiom text, source, prompt compositionality indicator, similarity indicator
- Key field `hard_to_guess`: True means the idiom fulfills criteria for memorization detection
- Useful as ground-truth non-compositional phrases for validation

## Dataset 2: CounterFact (Expanded)

### Overview
- **Source**: https://rome.baulab.info/ (Meng et al., 2022)
- **Size**: 54,464 factual prompts about named entities
- **Format**: CSV
- **Task**: Factual association recall; testing entity knowledge in LLMs

### Download Instructions
```bash
git clone https://github.com/sfeucht/footprints
cp footprints/data/counterfact_expanded.csv datasets/
```

### Loading
```python
import pandas as pd
df = pd.read_csv("datasets/counterfact_expanded.csv")
```

### Notes
- Contains prompts like "The mother tongue of {} is" with subject, true target, false target
- Pre-filtered for Llama-2-7b and Llama-3-8b correctness
- Used in Token Erasure paper for probing implicit vocabulary

## Dataset 3: Wikipedia Test (500 articles)

### Overview
- **Source**: https://github.com/sfeucht/footprints
- **Size**: 500 Wikipedia articles (~256k tokens)
- **Format**: CSV
- **Task**: General text for probing multi-token word processing

### Download Instructions
```bash
git clone https://github.com/sfeucht/footprints
cp footprints/data/wikipedia_test_500.csv datasets/
```

## Dataset 4: Future Lens Test Data

### Overview
- **Source**: https://github.com/KoyenaPal/future-lens
- **Size**: 1,000 samples from the Pile
- **Format**: CSV
- **Task**: Testing hidden state prediction of future tokens

### Download Instructions
```bash
git clone https://github.com/KoyenaPal/future-lens
cp future-lens/data/testing_data_teacher_1000.csv datasets/
```

## Dataset 5: Farahmand Non-Compositionality Dataset

### Overview
- **Source**: https://github.com/meghdadFar/en_ncs_noncompositional_conventionalized
- **Size**: 1,042 English noun-noun compounds with expert annotations
- **Format**: CSV (4 annotators, binary judgments)
- **Task**: Non-compositionality classification of noun compounds
- **License**: CC-BY-SA 3.0

### Download Instructions
```bash
git clone https://github.com/meghdadFar/en_ncs_noncompositional_conventionalized datasets/farahmand_ncs
```

### Loading
```python
import csv
with open("datasets/farahmand_ncs/instances_judgments/nonComp-judgments.csv") as f:
    for row in csv.reader(f):
        compound, *judgments = row
        is_noncomp = sum(int(j) for j in judgments) >= 3  # majority vote
```

### Notes
- Ground-truth compositionality labels for noun compounds
- 4 expert annotators per compound (binary: 1=non-compositional, 0=compositional)
- Ideal for validating whether Future Lens identifies non-compositional phrases

## Dataset 6: IdioTS (Idiomatic Language Test Suite)

### Overview
- **Source**: https://huggingface.co/datasets/fdelucaf/IdioTS
- **Size**: 195 examples (English and Spanish)
- **Format**: HuggingFace Dataset
- **Task**: Evaluating LLM idiom comprehension

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("fdelucaf/IdioTS")
ds.save_to_disk("datasets/idiots")
```

### Loading
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/idiots")
```
