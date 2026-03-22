"""
Future Lens v2: Non-Compositional Phrase Detection

Uses multiple signals to detect non-compositional phrases:
1. Model confidence: average next-token probability within phrases
2. Logit lens at middle layers: applying decoder to middle-layer hidden states
3. Future token information: whether middle-layer hidden states encode future tokens

Key improvement over v1: more robust phrase matching and richer metrics.
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from scipy import stats

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/workspaces/noncomp-phrases-nlp-claude/results")
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path("/workspaces/noncomp-phrases-nlp-claude/figures")
FIGURES_DIR.mkdir(exist_ok=True)


def load_model(model_name="gpt2-xl"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    n_layers = model.config.n_layer
    hidden_dim = model.config.n_embd
    vocab_size = model.config.vocab_size
    print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, Vocab: {vocab_size}")
    return model, tokenizer


# ======================== Data Loading ========================

def load_farahmand_ncs():
    """Load Farahmand noun compound compositionality dataset."""
    ncs_path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/farahmand_ncs/instances_judgments/nonComp-judgments.csv")
    compounds = []
    with open(ncs_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                compound = parts[0].strip()
                judgments = [int(j.strip()) for j in parts[1:5] if j.strip() in ('0', '1')]
                if len(judgments) == 4:
                    non_comp_score = sum(judgments) / 4.0
                    compounds.append({
                        'phrase': compound,
                        'non_comp_score': non_comp_score,
                        'is_non_comp': non_comp_score >= 0.75,
                        'judgments': judgments
                    })
    print(f"Loaded {len(compounds)} Farahmand compounds")
    nc = sum(1 for c in compounds if c['is_non_comp'])
    print(f"  Non-compositional (>=3/4): {nc}, Compositional: {len(compounds)-nc}")
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


def prepare_evaluation_data(compounds, idioms, tokenizer):
    """Prepare phrases for evaluation.

    For each phrase, we create a context and record the phrase boundaries.
    Uses character-level offsets for reliable phrase location.
    """
    phrases = []

    # Template contexts - the phrase starts right after "The "
    def make_entry(phrase, label, category, non_comp_score=None):
        # Use a minimal context so tokenization is predictable
        context = f"The {phrase} is"
        # Tokenize the full context
        full_ids = tokenizer.encode(context, add_special_tokens=False)
        # Tokenize just the prefix before the phrase
        prefix = "The "
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        # Tokenize prefix + phrase
        prefix_phrase = f"The {phrase}"
        prefix_phrase_ids = tokenizer.encode(prefix_phrase, add_special_tokens=False)

        phrase_start_idx = len(prefix_ids)
        phrase_end_idx = len(prefix_phrase_ids)
        phrase_len = phrase_end_idx - phrase_start_idx

        if phrase_len < 2:
            return None

        return {
            'text': context,
            'phrase': phrase,
            'label': label,
            'category': category,
            'non_comp_score': non_comp_score if non_comp_score is not None else (1.0 if label == 'non-compositional' else 0.0),
            'token_ids': full_ids,
            'phrase_start_idx': phrase_start_idx,
            'phrase_end_idx': phrase_end_idx,
            'phrase_len_tokens': phrase_len,
        }

    # Farahmand compounds
    for c in compounds:
        entry = make_entry(
            c['phrase'],
            'non-compositional' if c['is_non_comp'] else 'compositional',
            'farahmand',
            c['non_comp_score']
        )
        if entry:
            phrases.append(entry)

    # IdioMem idioms (all non-compositional)
    for idiom in idioms:
        entry = make_entry(idiom['phrase'], 'non-compositional', 'idiomem')
        if entry:
            phrases.append(entry)

    # Control: compositional bigrams
    compositional_phrases = [
        "big house", "red car", "old man", "new book", "small dog",
        "long road", "dark room", "cold water", "fast train", "good food",
        "green grass", "blue sky", "white wall", "tall tree", "young woman",
        "hot day", "short story", "thick book", "bright light", "quiet place",
        "deep water", "wide river", "strong wind", "heavy rain", "sharp knife",
        "round table", "soft bed", "clean room", "fresh air", "warm sun",
        "hard work", "full moon", "high mountain", "low price", "dry land",
        "rough road", "smooth stone", "sweet taste", "loud noise", "clear view",
        "large building", "great idea", "first time", "last chance", "real problem",
        "whole family", "next step", "right hand", "left foot", "other side",
        "main street", "little girl", "local store", "open door", "broken window",
        "empty bottle", "heavy load", "thin paper", "flat surface", "wet floor",
    ]
    for phrase in compositional_phrases:
        entry = make_entry(phrase, 'compositional', 'control_bigram')
        if entry:
            phrases.append(entry)

    # Summary
    from collections import Counter
    cat_counts = Counter((p['category'], p['label']) for p in phrases)
    print(f"\nPrepared {len(phrases)} phrases:")
    for (cat, label), count in sorted(cat_counts.items()):
        print(f"  {cat} / {label}: {count}")

    return phrases


# ======================== Core Analysis ========================

def analyze_phrase(model, tokenizer, text, phrase_start_idx, phrase_end_idx, token_ids,
                   layers_to_check):
    """Analyze a single phrase for Future Lens signals.

    Returns dict with metrics measuring how well middle-layer hidden states
    predict future tokens within the phrase.
    """
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # [seq_len, vocab]
    hidden_states = outputs.hidden_states  # tuple of [1, seq_len, hidden]

    # Get model's decoder components
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    actual_ids = inputs.input_ids[0]
    seq_len = actual_ids.shape[0]

    results = {}

    # 1. Model confidence: average next-token probability within the phrase
    # At position t, the model predicts token at t+1
    phrase_probs = []
    for t in range(max(0, phrase_start_idx - 1), min(phrase_end_idx - 1, seq_len - 1)):
        target_token = actual_ids[t + 1].item()
        prob = torch.softmax(logits[t].float(), dim=-1)[target_token].item()
        phrase_probs.append(prob)

    if phrase_probs:
        results['avg_next_token_prob'] = np.mean(phrase_probs)
        results['min_next_token_prob'] = np.min(phrase_probs)
        results['max_next_token_prob'] = np.max(phrase_probs)
        results['avg_surprisal'] = np.mean([-np.log2(max(p, 1e-10)) for p in phrase_probs])

    # 2. Logit lens at middle layers: apply decoder to middle-layer hidden states
    # At position t, apply decoder to hidden_state[layer][t] and check if it predicts
    # the correct next token (t+1)
    for layer in layers_to_check:
        hs = hidden_states[layer][0]  # [seq_len, hidden]

        layer_probs = []
        for t in range(max(0, phrase_start_idx - 1), min(phrase_end_idx - 1, seq_len - 1)):
            target_token = actual_ids[t + 1].item()
            with torch.no_grad():
                layer_logits = lm_head(ln_f(hs[t:t+1])).float()
            prob = torch.softmax(layer_logits, dim=-1)[0, target_token].item()
            layer_probs.append(prob)

        if layer_probs:
            results[f'logit_lens_L{layer}_avg_prob'] = np.mean(layer_probs)
            results[f'logit_lens_L{layer}_avg_surprisal'] = np.mean(
                [-np.log2(max(p, 1e-10)) for p in layer_probs]
            )

    # 3. Future token prediction from first token's hidden state
    # At phrase_start_idx, check if hidden states predict tokens at start+1, start+2, etc.
    for layer in layers_to_check:
        hs_first = hidden_states[layer][0, phrase_start_idx]  # [hidden]

        for k in range(1, min(4, phrase_end_idx - phrase_start_idx)):
            target_pos = phrase_start_idx + k
            if target_pos >= seq_len:
                break
            target_token = actual_ids[target_pos].item()

            # Apply decoder to first token's hidden state
            with torch.no_grad():
                pred_logits = lm_head(ln_f(hs_first.unsqueeze(0))).float()
            prob = torch.softmax(pred_logits, dim=-1)[0, target_token].item()
            top1 = int(pred_logits.argmax(-1).item() == target_token)
            top10 = int(target_token in pred_logits.topk(10, dim=-1).indices[0].tolist())

            results[f'future_L{layer}_k{k}_prob'] = prob
            results[f'future_L{layer}_k{k}_top1'] = top1
            results[f'future_L{layer}_k{k}_top10'] = top10

    # 4. Cross-position prediction: for each position in phrase, predict all future positions
    # This captures how much "lookahead" is encoded at each position
    last_layer = max(layers_to_check)
    lookahead_probs = []
    for t in range(phrase_start_idx, min(phrase_end_idx - 1, seq_len - 1)):
        for k in range(2, min(4, phrase_end_idx - t)):
            target_pos = t + k
            if target_pos >= seq_len:
                break
            target_token = actual_ids[target_pos].item()
            with torch.no_grad():
                pred_logits = lm_head(ln_f(hidden_states[last_layer][0, t:t+1])).float()
            prob = torch.softmax(pred_logits, dim=-1)[0, target_token].item()
            lookahead_probs.append(prob)

    if lookahead_probs:
        results['avg_lookahead_prob'] = np.mean(lookahead_probs)

    return results


def run_evaluation(model, tokenizer, phrases, layers_to_check):
    """Run evaluation on all phrases."""
    all_results = []

    for item in tqdm(phrases, desc="Evaluating phrases"):
        metrics = analyze_phrase(
            model, tokenizer,
            item['text'],
            item['phrase_start_idx'],
            item['phrase_end_idx'],
            item['token_ids'],
            layers_to_check
        )

        row = {
            'phrase': item['phrase'],
            'label': item['label'],
            'category': item['category'],
            'non_comp_score': item['non_comp_score'],
            'phrase_len_tokens': item['phrase_len_tokens'],
        }
        row.update(metrics)
        all_results.append(row)

    return pd.DataFrame(all_results)


# ======================== Analysis ========================

def analyze_results(df):
    """Perform statistical analysis of results."""
    print("\n" + "=" * 70)
    print("ANALYSIS OF RESULTS")
    print("=" * 70)

    # Focus on key metrics
    key_metrics = [
        'avg_next_token_prob', 'avg_surprisal',
        'avg_lookahead_prob',
    ]

    # Add logit lens and future prediction metrics
    for col in df.columns:
        if col.startswith('logit_lens_') and col.endswith('_avg_prob'):
            key_metrics.append(col)
        if col.startswith('future_') and col.endswith('_prob') and '_k1_' in col:
            key_metrics.append(col)

    # 1. Farahmand: compositional vs non-compositional
    print("\n--- Farahmand Compounds: Compositional vs Non-Compositional ---")
    far = df[df['category'] == 'farahmand']
    far_comp = far[far['label'] == 'compositional']
    far_noncomp = far[far['label'] == 'non-compositional']

    print(f"  N compositional: {len(far_comp)}")
    print(f"  N non-compositional: {len(far_noncomp)}")

    stat_results = []
    for metric in key_metrics:
        if metric not in df.columns:
            continue
        comp_vals = far_comp[metric].dropna()
        noncomp_vals = far_noncomp[metric].dropna()
        if len(comp_vals) < 5 or len(noncomp_vals) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(noncomp_vals, comp_vals, alternative='greater')
        cohens_d = (noncomp_vals.mean() - comp_vals.mean()) / np.sqrt(
            (comp_vals.std()**2 + noncomp_vals.std()**2) / 2
        ) if comp_vals.std() + noncomp_vals.std() > 0 else 0

        print(f"\n  {metric}:")
        print(f"    Comp: {comp_vals.mean():.6f} ± {comp_vals.std():.6f}")
        print(f"    NonComp: {noncomp_vals.mean():.6f} ± {noncomp_vals.std():.6f}")
        print(f"    Mann-Whitney U p={p_val:.4f}, Cohen's d={cohens_d:.3f}")

        stat_results.append({
            'metric': metric,
            'comparison': 'farahmand_noncomp_vs_comp',
            'comp_mean': comp_vals.mean(),
            'noncomp_mean': noncomp_vals.mean(),
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_comp': len(comp_vals),
            'n_noncomp': len(noncomp_vals),
        })

    # 2. IdioMem vs control bigrams
    print("\n--- IdioMem Idioms vs Control Bigrams ---")
    idiom_df = df[df['category'] == 'idiomem']
    control_df = df[df['category'] == 'control_bigram']

    print(f"  N idioms: {len(idiom_df)}")
    print(f"  N control: {len(control_df)}")

    for metric in key_metrics:
        if metric not in df.columns:
            continue
        idiom_vals = idiom_df[metric].dropna()
        control_vals = control_df[metric].dropna()
        if len(idiom_vals) < 5 or len(control_vals) < 5:
            continue

        u_stat, p_val = stats.mannwhitneyu(idiom_vals, control_vals, alternative='greater')
        cohens_d = (idiom_vals.mean() - control_vals.mean()) / np.sqrt(
            (control_vals.std()**2 + idiom_vals.std()**2) / 2
        ) if control_vals.std() + idiom_vals.std() > 0 else 0

        print(f"\n  {metric}:")
        print(f"    Control: {control_vals.mean():.6f} ± {control_vals.std():.6f}")
        print(f"    Idiom: {idiom_vals.mean():.6f} ± {idiom_vals.std():.6f}")
        print(f"    Mann-Whitney U p={p_val:.4f}, Cohen's d={cohens_d:.3f}")

        stat_results.append({
            'metric': metric,
            'comparison': 'idiomem_vs_control',
            'comp_mean': control_vals.mean(),
            'noncomp_mean': idiom_vals.mean(),
            'p_value': p_val,
            'cohens_d': cohens_d,
            'n_comp': len(control_vals),
            'n_noncomp': len(idiom_vals),
        })

    # 3. Correlation with compositionality score (Farahmand)
    print("\n--- Correlation: Future Lens metrics vs Compositionality Score ---")
    for metric in key_metrics:
        if metric not in far.columns:
            continue
        valid = far[[metric, 'non_comp_score']].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid['non_comp_score'], valid[metric])
        if p < 0.1:  # Report marginally significant
            print(f"  {metric}: Spearman r={r:.3f}, p={p:.4f}")
            stat_results.append({
                'metric': metric,
                'comparison': 'spearman_with_noncomp_score',
                'spearman_r': r,
                'p_value': p,
            })

    # 4. ROC/AUC for binary classification
    print("\n--- ROC AUC: Binary Non-Compositionality Classification (Farahmand) ---")
    from sklearn.metrics import roc_auc_score
    for metric in key_metrics:
        if metric not in far.columns:
            continue
        valid = far[[metric, 'is_non_comp']].dropna()
        if len(valid) < 20:
            continue
        try:
            # For surprisal, higher = more compositional, so negate
            scores = valid[metric].values
            if 'surprisal' in metric:
                scores = -scores
            auc = roc_auc_score(valid['is_non_comp'].astype(int), scores)
            print(f"  {metric}: AUC={auc:.3f}")
            stat_results.append({
                'metric': metric,
                'comparison': 'roc_auc_farahmand',
                'auc': auc,
            })
        except Exception:
            pass

    return pd.DataFrame(stat_results)


def create_visualizations(df, stat_df):
    """Create publication-quality visualizations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})

    # 1. Average next-token probability by category
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Farahmand compounds
    far = df[df['category'] == 'farahmand']
    ax = axes[0]
    for label, color in [('compositional', '#2196F3'), ('non-compositional', '#F44336')]:
        subset = far[far['label'] == label]['avg_next_token_prob'].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel('Average Next-Token Probability')
    ax.set_ylabel('Density')
    ax.set_title('Farahmand Compounds')
    ax.legend()

    # Panel B: Idioms vs control
    ax = axes[1]
    for cat, label, color in [('control_bigram', 'Control Bigrams', '#2196F3'),
                               ('idiomem', 'Idioms', '#F44336')]:
        subset = df[df['category'] == cat]['avg_next_token_prob'].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_xlabel('Average Next-Token Probability')
    ax.set_title('Idioms vs Control Bigrams')
    ax.legend()

    # Panel C: Average surprisal comparison
    ax = axes[2]
    categories = ['control_bigram', 'farahmand', 'idiomem']
    labels_map = {'control_bigram': 'Control\nBigrams', 'farahmand': 'Farahmand\nCompounds', 'idiomem': 'Idioms'}
    means = []
    stds = []
    for cat in categories:
        vals = df[df['category'] == cat]['avg_surprisal'].dropna()
        means.append(vals.mean())
        stds.append(vals.std() / np.sqrt(len(vals)))

    bars = ax.bar([labels_map[c] for c in categories], means, yerr=stds,
                  color=['#2196F3', '#9C27B0', '#F44336'], alpha=0.7, capsize=5)
    ax.set_ylabel('Average Surprisal (bits)')
    ax.set_title('Phrase Predictability')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'phrase_predictability.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'phrase_predictability.png'}")

    # 2. Future Lens: Logit lens accuracy across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Logit lens avg prob across layers for different categories
    ax = axes[0]
    layers = []
    for col in sorted(df.columns):
        if col.startswith('logit_lens_L') and col.endswith('_avg_prob'):
            layer = int(col.split('_L')[1].split('_')[0])
            layers.append(layer)

    for cat, label, color, marker in [
        ('control_bigram', 'Control Bigrams', '#2196F3', 'o'),
        ('farahmand', 'Farahmand (all)', '#9C27B0', 's'),
        ('idiomem', 'Idioms', '#F44336', '^'),
    ]:
        subset = df[df['category'] == cat]
        layer_means = []
        layer_sems = []
        for layer in sorted(layers):
            col = f'logit_lens_L{layer}_avg_prob'
            vals = subset[col].dropna()
            layer_means.append(vals.mean())
            layer_sems.append(vals.std() / np.sqrt(max(len(vals), 1)))

        ax.errorbar(sorted(layers), layer_means, yerr=layer_sems,
                    label=label, marker=marker, color=color, capsize=3)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Average Next-Token Probability (Logit Lens)')
    ax.set_title('Logit Lens: Next-Token Prediction by Layer')
    ax.legend()

    # Panel B: Future prediction (k>1) from first token
    ax = axes[1]
    for cat, label, color, marker in [
        ('control_bigram', 'Control Bigrams', '#2196F3', 'o'),
        ('farahmand', 'Farahmand (all)', '#9C27B0', 's'),
        ('idiomem', 'Idioms', '#F44336', '^'),
    ]:
        subset = df[df['category'] == cat]
        layer_means = []
        valid_layers = []
        for layer in sorted(layers):
            col = f'future_L{layer}_k2_prob'
            if col in subset.columns:
                vals = subset[col].dropna()
                if len(vals) > 0:
                    layer_means.append(vals.mean())
                    valid_layers.append(layer)

        if valid_layers:
            ax.plot(valid_layers, layer_means, label=label, marker=marker, color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Probability of Token at t+2')
    ax.set_title('Future Lens: Predict Token 2 Ahead from First Token')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'future_lens_layers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'future_lens_layers.png'}")

    # 3. Compositionality score vs Future Lens metrics (scatter)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    far = df[df['category'] == 'farahmand']

    ax = axes[0]
    valid = far[['non_comp_score', 'avg_next_token_prob']].dropna()
    ax.scatter(valid['non_comp_score'], valid['avg_next_token_prob'],
               alpha=0.3, s=20, color='#673AB7')
    r, p = stats.spearmanr(valid['non_comp_score'], valid['avg_next_token_prob'])
    ax.set_xlabel('Non-Compositionality Score (0=comp, 1=non-comp)')
    ax.set_ylabel('Average Next-Token Probability')
    ax.set_title(f'Compositionality vs Predictability\n(Spearman r={r:.3f}, p={p:.4f})')

    ax = axes[1]
    valid = far[['non_comp_score', 'avg_surprisal']].dropna()
    ax.scatter(valid['non_comp_score'], valid['avg_surprisal'],
               alpha=0.3, s=20, color='#673AB7')
    r, p = stats.spearmanr(valid['non_comp_score'], valid['avg_surprisal'])
    ax.set_xlabel('Non-Compositionality Score')
    ax.set_ylabel('Average Surprisal (bits)')
    ax.set_title(f'Compositionality vs Surprisal\n(Spearman r={r:.3f}, p={p:.4f})')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'compositionality_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'compositionality_correlation.png'}")

    # 4. Farahmand non-comp vs comp: effect across layers (key result)
    fig, ax = plt.subplots(figsize=(10, 5))
    far_comp = far[far['label'] == 'compositional']
    far_noncomp = far[far['label'] == 'non-compositional']

    diff_means = []
    diff_layers = []
    for layer in sorted(layers):
        col = f'logit_lens_L{layer}_avg_prob'
        if col not in far.columns:
            continue
        comp_mean = far_comp[col].dropna().mean()
        noncomp_mean = far_noncomp[col].dropna().mean()
        diff_means.append(noncomp_mean - comp_mean)
        diff_layers.append(layer)

    ax.bar(range(len(diff_layers)), diff_means, tick_label=[str(l) for l in diff_layers],
           color=['#F44336' if d > 0 else '#2196F3' for d in diff_means], alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ Logit Lens Prob (NonComp - Comp)')
    ax.set_title('Non-Compositional Advantage in Logit Lens Prediction')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'noncomp_advantage_layers.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'noncomp_advantage_layers.png'}")

    # 5. Lookahead probability comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    data_for_box = []
    labels_for_box = []
    for cat, label in [('control_bigram', 'Control\nBigrams'),
                        ('farahmand', 'Farahmand\nCompounds'),
                        ('idiomem', 'Idioms')]:
        vals = df[df['category'] == cat]['avg_lookahead_prob'].dropna().values
        data_for_box.append(vals)
        labels_for_box.append(label)

    bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    colors = ['#2196F3', '#9C27B0', '#F44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Average Lookahead Probability (k≥2)')
    ax.set_title('Future Token Predictability by Phrase Type')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'lookahead_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'lookahead_boxplot.png'}")


def main():
    start_time = time.time()

    print("=" * 70)
    print("Future Lens v2: Non-Compositional Phrase Detection")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model, tokenizer = load_model("gpt2-xl")
    n_layers = model.config.n_layer  # 48

    # Select layers to analyze
    layers_to_check = [0, 6, 12, 18, 24, 30, 36, 42, 47]

    # Load data
    compounds = load_farahmand_ncs()
    idioms = load_idiomem()
    phrases = prepare_evaluation_data(compounds, idioms, tokenizer)

    # Run evaluation
    print(f"\nRunning evaluation on {len(phrases)} phrases...")
    results_df = run_evaluation(model, tokenizer, phrases, layers_to_check)

    # Save raw results
    results_df.to_csv(RESULTS_DIR / "future_lens_v2_results.csv", index=False)
    print(f"Saved results: {len(results_df)} rows")

    # Statistical analysis
    stat_df = analyze_results(results_df)
    stat_df.to_csv(RESULTS_DIR / "statistical_results.csv", index=False)

    # Visualizations
    print("\nCreating visualizations...")
    create_visualizations(results_df, stat_df)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return results_df, stat_df


if __name__ == "__main__":
    results_df, stat_df = main()
