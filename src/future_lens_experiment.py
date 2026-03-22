"""
Future Lens Experiment: Detecting Non-Compositional Phrases

Tests whether Future Lens prediction accuracy (predicting future tokens from
hidden states) is higher for non-compositional phrases than compositional ones,
providing a lower bound on what LLMs have memorized as single units.
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/workspaces/noncomp-phrases-nlp-claude/results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_model(model_name="gpt2-xl"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded. Layers: {model.config.n_layer}, Hidden: {model.config.n_embd}, Vocab: {model.config.vocab_size}")
    return model, tokenizer


class LinearProbe(nn.Module):
    """Linear probe mapping hidden_state[t] -> hidden_state[t+k]."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.linear(x)


def extract_hidden_states(model, tokenizer, texts, batch_size=8, max_length=128):
    """Extract hidden states from all layers for a batch of texts.

    Returns: dict mapping layer_idx -> tensor of shape [total_tokens, hidden_dim]
             along with token_ids and text boundaries.
    """
    all_hidden = {}  # layer -> list of tensors
    all_token_ids = []
    text_boundaries = []  # (start, end) for each text

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq, hidden)
        attention_mask = inputs.attention_mask

        for b in range(len(batch_texts)):
            mask = attention_mask[b].bool()
            seq_len = mask.sum().item()
            token_ids = inputs.input_ids[b][mask].cpu()

            start = sum(t[1] - t[0] for t in text_boundaries) if text_boundaries else 0
            text_boundaries.append((start, start + seq_len))
            all_token_ids.append(token_ids)

            for layer_idx in range(len(hidden_states)):
                if layer_idx not in all_hidden:
                    all_hidden[layer_idx] = []
                # Get hidden states for non-padded tokens
                hs = hidden_states[layer_idx][b][mask].cpu().float()
                all_hidden[layer_idx].append(hs)

    # Concatenate
    for layer_idx in all_hidden:
        all_hidden[layer_idx] = torch.cat(all_hidden[layer_idx], dim=0)
    all_token_ids = torch.cat(all_token_ids, dim=0)

    return all_hidden, all_token_ids, text_boundaries


def train_probes(model, tokenizer, train_texts, layers_to_probe, max_k=3,
                 batch_size=8, max_length=128):
    """Train linear probes for predicting future hidden states.

    For each layer L and offset k, trains a linear map:
        h[L, t] -> h[L, t+k]

    Returns: dict of (layer, k) -> trained LinearProbe
    """
    print(f"\nExtracting hidden states from {len(train_texts)} training texts...")
    hidden_states, token_ids, boundaries = extract_hidden_states(
        model, tokenizer, train_texts, batch_size=batch_size, max_length=max_length
    )

    hidden_dim = hidden_states[0].shape[1]
    probes = {}

    for layer in layers_to_probe:
        hs = hidden_states[layer]  # [total_tokens, hidden_dim]

        for k in range(1, max_k + 1):
            print(f"  Training probe: layer {layer}, k={k}...")

            # Create input-output pairs: h[t] -> h[t+k]
            # Only use pairs within the same text
            X_list, Y_list = [], []
            for start, end in boundaries:
                if end - start <= k:
                    continue
                seg = hs[start:end]
                X_list.append(seg[:-k])
                Y_list.append(seg[k:])

            if not X_list:
                continue

            X = torch.cat(X_list, dim=0)
            Y = torch.cat(Y_list, dim=0)

            # Train with closed-form least squares (faster than SGD for linear)
            # W = (X^T X)^{-1} X^T Y
            # Use regularized version for numerical stability
            probe = LinearProbe(hidden_dim)

            # Use mini-batch SGD for memory efficiency
            dataset = torch.utils.data.TensorDataset(X, Y)
            loader = DataLoader(dataset, batch_size=512, shuffle=True)
            optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

            probe.to(DEVICE)
            probe.train()

            for epoch in range(5):
                total_loss = 0
                n_batches = 0
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = probe(xb)
                    loss = nn.MSELoss()(pred, yb)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

                if epoch == 4:
                    print(f"    Final loss: {total_loss/n_batches:.6f}")

            probe.cpu()
            probe.eval()
            probes[(layer, k)] = probe

    return probes, hidden_dim


def evaluate_future_lens(model, tokenizer, probes, hidden_dim, phrases_with_contexts,
                         layers_to_probe, max_k=3, batch_size=4, max_length=128):
    """Evaluate Future Lens prediction accuracy on phrases in context.

    Args:
        phrases_with_contexts: list of dicts with keys:
            - 'text': full text containing the phrase
            - 'phrase': the target phrase
            - 'phrase_start_char': character offset where phrase starts
            - 'label': compositionality label
            - 'category': category name

    Returns: DataFrame with per-phrase Future Lens metrics.
    """
    results = []

    # Get the model's decoder components
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    for i in tqdm(range(0, len(phrases_with_contexts), batch_size), desc="Evaluating"):
        batch = phrases_with_contexts[i:i + batch_size]
        texts = [item['text'] for item in batch]

        inputs = tokenizer(
            texts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states

        for b, item in enumerate(batch):
            mask = inputs.attention_mask[b].bool()
            seq_token_ids = inputs.input_ids[b][mask]
            seq_list = seq_token_ids.tolist()

            # Find phrase position by encoding prefix up to phrase start
            prefix_text = item['text'][:item['phrase_start_char']]
            prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
            phrase_start = len(prefix_tokens)

            # Determine phrase length by encoding full text and prefix+phrase
            full_text_up_to_phrase_end = item['text'][:item['phrase_start_char'] + len(item['phrase'])]
            full_tokens = tokenizer.encode(full_text_up_to_phrase_end, add_special_tokens=False)
            phrase_len = len(full_tokens) - len(prefix_tokens)

            if phrase_len < 2 or phrase_start + phrase_len > len(seq_list):
                continue

            result = {
                'phrase': item['phrase'],
                'label': item['label'],
                'category': item['category'],
                'phrase_len_tokens': phrase_len,
                'phrase_start_pos': phrase_start,
            }

            # For each layer and k, compute prediction accuracy
            for layer in layers_to_probe:
                hs_layer = hidden_states[layer][b][mask].cpu().float()

                for k in range(1, max_k + 1):
                    if (layer, k) not in probes:
                        continue

                    probe = probes[(layer, k)].to(DEVICE)

                    # Predict from first token of phrase
                    if phrase_start + k < len(seq_list):
                        source_hs = hs_layer[phrase_start].unsqueeze(0).to(DEVICE)
                        predicted_hs = probe(source_hs)

                        # Decode to logits
                        with torch.no_grad():
                            logits = lm_head(ln_f(predicted_hs.half())).float()

                        probs = torch.softmax(logits, dim=-1).squeeze(0)
                        target_token = seq_list[phrase_start + k]

                        target_prob = probs[target_token].item()
                        top1_correct = (probs.argmax().item() == target_token)
                        top5_correct = (target_token in probs.topk(5).indices.tolist())
                        top10_correct = (target_token in probs.topk(10).indices.tolist())
                        surprisal = -np.log2(max(target_prob, 1e-10))

                        result[f'layer{layer}_k{k}_prob'] = target_prob
                        result[f'layer{layer}_k{k}_top1'] = int(top1_correct)
                        result[f'layer{layer}_k{k}_top5'] = int(top5_correct)
                        result[f'layer{layer}_k{k}_top10'] = int(top10_correct)
                        result[f'layer{layer}_k{k}_surprisal'] = surprisal

                    probe.cpu()

            # Also compute standard next-token prediction (baseline)
            # At phrase_start, what's the model's own prediction for next token?
            with torch.no_grad():
                logits_baseline = outputs.logits[b][phrase_start].float()
            probs_baseline = torch.softmax(logits_baseline, dim=-1).cpu()

            for k in range(1, max_k + 1):
                if phrase_start + k < len(seq_list):
                    target_token = seq_list[phrase_start + k]
                    result[f'baseline_k{k}_prob'] = probs_baseline[target_token].item()
                    result[f'baseline_k{k}_top1'] = int(probs_baseline.argmax().item() == target_token)

            results.append(result)

    return pd.DataFrame(results)


# ============== Data Preparation ==============

def load_farahmand_ncs():
    """Load Farahmand noun compound compositionality dataset."""
    ncs_path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/farahmand_ncs/instances_judgments/nonComp-judgments.csv")

    compounds = []
    with open(ncs_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                compound = parts[0].strip()
                judgments = []
                for j in parts[1:5]:
                    j = j.strip()
                    if j in ('0', '1'):
                        judgments.append(int(j))
                if len(judgments) == 4:
                    # Majority vote: non-compositional if >= 3 judges say so
                    non_comp_score = sum(judgments) / len(judgments)
                    is_non_comp = non_comp_score >= 0.75  # At least 3/4 judges
                    compounds.append({
                        'phrase': compound,
                        'non_comp_score': non_comp_score,
                        'is_non_comp': is_non_comp,
                        'judgments': judgments
                    })

    print(f"Loaded {len(compounds)} Farahmand compounds")
    non_comp = sum(1 for c in compounds if c['is_non_comp'])
    comp = len(compounds) - non_comp
    print(f"  Non-compositional (>=3/4 judges): {non_comp}")
    print(f"  Compositional: {comp}")

    return compounds


def load_idiomem():
    """Load IdioMem idiom dataset."""
    path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/idiomem.jsonl")
    idioms = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            idioms.append({
                'phrase': data['idiom'],
                'hard_to_guess': data.get('hard_to_guess', False),
                'source': data.get('source', 'unknown')
            })
    print(f"Loaded {len(idioms)} IdioMem idioms")
    return idioms


def prepare_phrases_in_context(compounds, idioms, tokenizer, max_phrase_tokens=6):
    """Prepare phrases embedded in simple context sentences for evaluation.

    Returns list of dicts with 'text', 'phrase', 'label', 'category'.
    """
    phrases = []

    # Farahmand compounds in context
    for c in compounds:
        phrase = c['phrase']
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if len(tokens) < 2 or len(tokens) > max_phrase_tokens:
            continue

        # Simple template context
        text = f"The {phrase} was mentioned in the news today."
        phrases.append({
            'text': text,
            'phrase': phrase,
            'label': 'non-compositional' if c['is_non_comp'] else 'compositional',
            'category': 'farahmand',
            'non_comp_score': c['non_comp_score'],
            'phrase_start_char': text.index(phrase),
        })

    # IdioMem idioms
    for idiom_data in idioms:
        phrase = idiom_data['phrase']
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if len(tokens) < 2 or len(tokens) > max_phrase_tokens + 4:  # idioms can be longer
            continue

        text = f"He said {phrase} and walked away."
        try:
            idx = text.index(phrase)
        except ValueError:
            continue

        phrases.append({
            'text': text,
            'phrase': phrase,
            'label': 'non-compositional',
            'category': 'idiomem',
            'non_comp_score': 1.0,
            'phrase_start_char': idx,
        })

    # Random compositional bigrams from common text as control
    common_bigrams = [
        "big house", "red car", "old man", "new book", "small dog",
        "long road", "dark room", "cold water", "fast train", "good food",
        "green grass", "blue sky", "white wall", "tall tree", "young woman",
        "hot day", "short story", "thick book", "bright light", "quiet place",
        "deep water", "wide river", "strong wind", "heavy rain", "sharp knife",
        "round table", "soft bed", "clean room", "fresh air", "warm sun",
        "hard work", "full moon", "high mountain", "low price", "dry land",
        "rough road", "smooth stone", "sweet taste", "loud noise", "clear view",
    ]
    for bigram in common_bigrams:
        tokens = tokenizer.encode(bigram, add_special_tokens=False)
        if len(tokens) < 2:
            continue
        text = f"The {bigram} was visible from a distance."
        try:
            idx = text.index(bigram)
        except ValueError:
            continue
        phrases.append({
            'text': text,
            'phrase': bigram,
            'label': 'compositional',
            'category': 'random_bigram',
            'non_comp_score': 0.0,
            'phrase_start_char': idx,
        })

    print(f"\nPrepared {len(phrases)} phrases for evaluation:")
    cats = {}
    for p in phrases:
        key = (p['category'], p['label'])
        cats[key] = cats.get(key, 0) + 1
    for key, count in sorted(cats.items()):
        print(f"  {key[0]} / {key[1]}: {count}")

    return phrases


def get_training_texts(tokenizer, n_texts=500, max_length=128):
    """Get training texts for probe training from Wikipedia."""
    wiki_path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/wikipedia_test_500.csv")

    if wiki_path.exists():
        df = pd.read_csv(wiki_path)
        texts = df['text'].dropna().tolist()
        # Split long texts into chunks
        chunks = []
        for text in texts:
            # Take first ~500 chars as a chunk
            if len(text) > 200:
                chunks.append(text[:500])
            if len(chunks) >= n_texts:
                break
        print(f"Loaded {len(chunks)} training text chunks from Wikipedia")
        return chunks

    # Fallback: use the Future Lens test data
    pile_path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/testing_data_teacher_1000.csv")
    if pile_path.exists():
        df = pd.read_csv(pile_path)
        texts = df['decoded_prefix'].dropna().tolist()[:n_texts]
        print(f"Loaded {len(texts)} training texts from Pile")
        return texts

    raise FileNotFoundError("No training data found")


def main():
    """Run the full Future Lens non-compositionality experiment."""
    start_time = time.time()

    print("=" * 70)
    print("Future Lens Non-Compositionality Detection Experiment")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Load model
    model_name = "gpt2-xl"
    model, tokenizer = load_model(model_name)
    hidden_dim = model.config.n_embd
    n_layers = model.config.n_layer

    # 2. Select layers to probe (focus on middle layers where Future Lens peaks)
    # GPT-2-xl has 48 layers. Future Lens peaks around middle layers.
    layers_to_probe = [0, 6, 12, 18, 24, 30, 36, 42, 47]
    print(f"\nProbing layers: {layers_to_probe}")

    # 3. Get training texts
    train_texts = get_training_texts(tokenizer, n_texts=400)

    # 4. Train probes
    max_k = 3
    probes, hidden_dim = train_probes(
        model, tokenizer, train_texts, layers_to_probe,
        max_k=max_k, batch_size=8, max_length=128
    )
    print(f"\nTrained {len(probes)} probes")

    # 5. Prepare evaluation data
    compounds = load_farahmand_ncs()
    idioms = load_idiomem()
    phrases = prepare_phrases_in_context(compounds, idioms, tokenizer)

    # 6. Evaluate
    print(f"\nEvaluating Future Lens on {len(phrases)} phrases...")
    results_df = evaluate_future_lens(
        model, tokenizer, probes, hidden_dim, phrases,
        layers_to_probe, max_k=max_k, batch_size=4
    )

    # 7. Save results
    results_path = RESULTS_DIR / "future_lens_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print(f"Total phrases evaluated: {len(results_df)}")

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s")

    # 8. Quick summary
    print("\n" + "=" * 50)
    print("QUICK SUMMARY")
    print("=" * 50)

    # Best layer analysis (use layer 24, middle of GPT-2-xl)
    best_layer = 24
    for k in range(1, max_k + 1):
        col = f'layer{best_layer}_k{k}_prob'
        if col in results_df.columns:
            for cat in results_df['category'].unique():
                for label in results_df['label'].unique():
                    mask = (results_df['category'] == cat) & (results_df['label'] == label)
                    if mask.sum() > 0:
                        mean_prob = results_df.loc[mask, col].mean()
                        print(f"  Layer {best_layer}, k={k}, {cat}/{label}: "
                              f"mean_prob={mean_prob:.4f} (n={mask.sum()})")

    return results_df


if __name__ == "__main__":
    results = main()
