"""
Future Lens v3: Non-Compositional Phrase Detection

Uses offset_mapping for reliable phrase boundary detection.
Measures multiple signals to test if Future Lens can lower-bound
the set of non-compositional phrases memorized by LLMs.
"""

import json
import random
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

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
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    print(f"  {model.config.n_layer} layers, {model.config.n_embd} hidden, {model.config.vocab_size} vocab")
    return model, tokenizer


def find_phrase_tokens(tokenizer, text, phrase, phrase_char_start):
    """Find token indices corresponding to phrase in text using offset_mapping."""
    enc = tokenizer(text, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    token_ids = enc['input_ids']

    phrase_char_end = phrase_char_start + len(phrase)

    # Find tokens that overlap with the phrase character span
    phrase_token_start = None
    phrase_token_end = None

    for i, (start, end) in enumerate(offsets):
        if start < phrase_char_end and end > phrase_char_start:
            if phrase_token_start is None:
                phrase_token_start = i
            phrase_token_end = i + 1

    if phrase_token_start is None:
        return None, None, token_ids

    return phrase_token_start, phrase_token_end, token_ids


def load_farahmand_ncs():
    path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/farahmand_ncs/instances_judgments/nonComp-judgments.csv")
    compounds = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 5:
                compound = parts[0].strip()
                judgments = [int(j.strip()) for j in parts[1:5] if j.strip() in ('0', '1')]
                if len(judgments) == 4:
                    score = sum(judgments) / 4.0
                    compounds.append({
                        'phrase': compound,
                        'non_comp_score': score,
                        'is_non_comp': score >= 0.75,
                    })
    nc = sum(1 for c in compounds if c['is_non_comp'])
    print(f"Farahmand: {len(compounds)} compounds ({nc} non-comp, {len(compounds)-nc} comp)")
    return compounds


def load_idiomem():
    path = Path("/workspaces/noncomp-phrases-nlp-claude/datasets/idiomem.jsonl")
    idioms = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            idioms.append({'phrase': data['idiom']})
    print(f"IdioMem: {len(idioms)} idioms")
    return idioms


def prepare_data(compounds, idioms, tokenizer):
    """Prepare all phrases with reliable token boundaries."""
    entries = []

    def add_phrase(phrase, label, category, non_comp_score=None):
        text = f"The {phrase} is important"
        char_start = 4  # "The " is 4 chars
        start, end, token_ids = find_phrase_tokens(tokenizer, text, phrase, char_start)
        if start is None or end is None:
            return
        n_tokens = end - start
        if n_tokens < 2:
            return
        entries.append({
            'text': text,
            'phrase': phrase,
            'label': label,
            'category': category,
            'non_comp_score': non_comp_score if non_comp_score is not None else (1.0 if label == 'non-compositional' else 0.0),
            'is_non_comp': label == 'non-compositional',
            'phrase_token_start': start,
            'phrase_token_end': end,
            'n_phrase_tokens': n_tokens,
        })

    for c in compounds:
        add_phrase(c['phrase'], 'non-compositional' if c['is_non_comp'] else 'compositional',
                   'farahmand', c['non_comp_score'])

    for idiom in idioms:
        add_phrase(idiom['phrase'], 'non-compositional', 'idiomem')

    # Control: compositional adjective-noun pairs
    controls = [
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
    ]
    for phrase in controls:
        add_phrase(phrase, 'compositional', 'control')

    from collections import Counter
    counts = Counter((e['category'], e['label']) for e in entries)
    print(f"\nPrepared {len(entries)} phrases:")
    for (cat, lab), n in sorted(counts.items()):
        print(f"  {cat}/{lab}: {n}")

    return entries


def analyze_single(model, tokenizer, entry, layers):
    """Analyze one phrase, returning a dict of metrics."""
    text = entry['text']
    ps = entry['phrase_token_start']  # phrase start token index
    pe = entry['phrase_token_end']    # phrase end token index

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits[0].float()       # [seq, vocab]
    hs = out.hidden_states               # tuple of [1, seq, hidden]
    ids = inputs.input_ids[0]
    seq_len = ids.shape[0]
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    result = {}

    # 1. Model's own next-token prediction within phrase
    # At position t, model predicts t+1
    ntp_probs = []
    for t in range(ps, min(pe - 1, seq_len - 1)):
        target = ids[t + 1].item()
        prob = torch.softmax(logits[t], dim=-1)[target].item()
        ntp_probs.append(prob)

    if ntp_probs:
        result['ntp_avg_prob'] = np.mean(ntp_probs)
        result['ntp_avg_surprisal'] = np.mean([-np.log2(max(p, 1e-10)) for p in ntp_probs])
        # Also: probability of 2nd token given position just before phrase
        if ps > 0:
            target = ids[ps].item()
            prob_before = torch.softmax(logits[ps - 1], dim=-1)[target].item()
            result['ntp_entry_prob'] = prob_before

    # 2. Logit lens at each layer (next-token prediction from intermediate representations)
    for layer in layers:
        h = hs[layer][0]  # [seq, hidden]
        probs = []
        for t in range(ps, min(pe - 1, seq_len - 1)):
            target = ids[t + 1].item()
            with torch.no_grad():
                layer_logits = lm_head(ln_f(h[t:t+1])).float()
            prob = torch.softmax(layer_logits, dim=-1)[0, target].item()
            probs.append(prob)
        if probs:
            result[f'll_L{layer}_avg'] = np.mean(probs)

    # 3. Future Lens: from first phrase token, predict k tokens ahead
    for layer in layers:
        h_first = hs[layer][0, ps]  # hidden state at first phrase token
        with torch.no_grad():
            first_logits = lm_head(ln_f(h_first.unsqueeze(0))).float()
        first_probs = torch.softmax(first_logits, dim=-1)[0]

        for k in range(1, min(5, pe - ps)):
            target = ids[ps + k].item()
            prob = first_probs[target].item()
            top1 = int(first_logits.argmax(-1).item() == target)
            top5 = int(target in first_logits.topk(5, dim=-1).indices[0].tolist())
            result[f'fl_L{layer}_k{k}_prob'] = prob
            result[f'fl_L{layer}_k{k}_top1'] = top1
            result[f'fl_L{layer}_k{k}_top5'] = top5

    # 4. Aggregate Future Lens: average probability of all future tokens from first token
    best_layer = layers[len(layers)//2]  # Middle layer
    h_first = hs[best_layer][0, ps]
    with torch.no_grad():
        first_logits = lm_head(ln_f(h_first.unsqueeze(0))).float()
    first_probs = torch.softmax(first_logits, dim=-1)[0]

    future_probs = []
    for k in range(1, pe - ps):
        target = ids[ps + k].item()
        future_probs.append(first_probs[target].item())
    if future_probs:
        result['fl_mid_avg_prob'] = np.mean(future_probs)
        result['fl_mid_avg_surprisal'] = np.mean(
            [-np.log2(max(p, 1e-10)) for p in future_probs]
        )

    # 5. Cross-position lookahead: from each phrase position, predict k=2,3 ahead
    lookahead_probs = []
    for t in range(ps, min(pe - 2, seq_len - 2)):
        for k in range(2, min(4, pe - t)):
            target = ids[t + k].item()
            with torch.no_grad():
                la_logits = lm_head(ln_f(hs[best_layer][0, t:t+1])).float()
            prob = torch.softmax(la_logits, dim=-1)[0, target].item()
            lookahead_probs.append(prob)
    if lookahead_probs:
        result['lookahead_avg'] = np.mean(lookahead_probs)

    return result


def run_experiment(model, tokenizer, entries, layers):
    rows = []
    for entry in tqdm(entries, desc="Evaluating"):
        metrics = analyze_single(model, tokenizer, entry, layers)
        row = {
            'phrase': entry['phrase'],
            'label': entry['label'],
            'category': entry['category'],
            'non_comp_score': entry['non_comp_score'],
            'is_non_comp': entry['is_non_comp'],
            'n_tokens': entry['n_phrase_tokens'],
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_analysis(df):
    """Run full statistical analysis."""
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS")
    print("=" * 70)

    key_metrics = ['ntp_avg_prob', 'ntp_avg_surprisal', 'fl_mid_avg_prob',
                   'fl_mid_avg_surprisal', 'lookahead_avg']

    # Add layer-specific metrics
    for col in df.columns:
        if col.startswith('ll_L') and col.endswith('_avg'):
            key_metrics.append(col)
        if col.startswith('fl_L') and '_k1_prob' in col:
            key_metrics.append(col)

    stat_rows = []

    # --- Test 1: Farahmand comp vs non-comp ---
    print("\n--- Farahmand: Compositional vs Non-Compositional ---")
    far = df[df['category'] == 'farahmand']
    comp = far[far['label'] == 'compositional']
    ncomp = far[far['label'] == 'non-compositional']
    print(f"  N comp: {len(comp)}, N non-comp: {len(ncomp)}")

    for m in key_metrics:
        if m not in df.columns:
            continue
        cv = comp[m].dropna()
        nv = ncomp[m].dropna()
        if len(cv) < 3 or len(nv) < 3:
            continue
        u, p = stats.mannwhitneyu(nv, cv, alternative='two-sided')
        pooled_std = np.sqrt((cv.var() + nv.var()) / 2) if cv.var() + nv.var() > 0 else 1
        d = (nv.mean() - cv.mean()) / pooled_std
        print(f"  {m}: comp={cv.mean():.4f}±{cv.std():.4f}, ncomp={nv.mean():.4f}±{nv.std():.4f}, p={p:.4f}, d={d:.3f}")
        stat_rows.append({'test': 'farahmand', 'metric': m,
                          'comp_mean': cv.mean(), 'ncomp_mean': nv.mean(),
                          'p': p, 'd': d, 'n_comp': len(cv), 'n_ncomp': len(nv)})

    # --- Test 2: Idioms vs control bigrams ---
    print("\n--- IdioMem vs Control Bigrams ---")
    idiom = df[df['category'] == 'idiomem']
    ctrl = df[df['category'] == 'control']
    print(f"  N idioms: {len(idiom)}, N control: {len(ctrl)}")

    for m in key_metrics:
        if m not in df.columns:
            continue
        iv = idiom[m].dropna()
        cv2 = ctrl[m].dropna()
        if len(iv) < 3 or len(cv2) < 3:
            continue
        u, p = stats.mannwhitneyu(iv, cv2, alternative='two-sided')
        pooled_std = np.sqrt((cv2.var() + iv.var()) / 2) if cv2.var() + iv.var() > 0 else 1
        d = (iv.mean() - cv2.mean()) / pooled_std
        print(f"  {m}: ctrl={cv2.mean():.4f}±{cv2.std():.4f}, idiom={iv.mean():.4f}±{iv.std():.4f}, p={p:.4f}, d={d:.3f}")
        stat_rows.append({'test': 'idiom_vs_ctrl', 'metric': m,
                          'comp_mean': cv2.mean(), 'ncomp_mean': iv.mean(),
                          'p': p, 'd': d, 'n_comp': len(cv2), 'n_ncomp': len(iv)})

    # --- Test 3: Correlation with compositionality score ---
    print("\n--- Spearman Correlation with Non-Compositionality Score (Farahmand) ---")
    for m in key_metrics:
        if m not in far.columns:
            continue
        valid = far[[m, 'non_comp_score']].dropna()
        if len(valid) < 10:
            continue
        r, p = stats.spearmanr(valid['non_comp_score'], valid[m])
        if abs(r) > 0.05:
            print(f"  {m}: r={r:.4f}, p={p:.4f}")
            stat_rows.append({'test': 'spearman', 'metric': m, 'r': r, 'p': p})

    # --- Test 4: ROC AUC ---
    print("\n--- ROC AUC for Binary Classification (Farahmand) ---")
    for m in key_metrics:
        if m not in far.columns:
            continue
        valid = far[[m, 'is_non_comp']].dropna()
        if len(valid) < 20 or valid['is_non_comp'].nunique() < 2:
            continue
        scores = valid[m].values
        if 'surprisal' in m:
            scores = -scores  # Lower surprisal = more predictable = more non-comp
        try:
            auc = roc_auc_score(valid['is_non_comp'].astype(int), scores)
            print(f"  {m}: AUC = {auc:.3f}")
            stat_rows.append({'test': 'auc', 'metric': m, 'auc': auc})
        except Exception:
            pass

    # --- Combined classification: Farahmand + control vs idioms ---
    print("\n--- ROC AUC: All Compositional vs All Non-Compositional ---")
    all_comp = df[(df['label'] == 'compositional')]
    all_ncomp = df[(df['label'] == 'non-compositional')]
    combined = pd.concat([all_comp, all_ncomp])
    for m in ['ntp_avg_prob', 'fl_mid_avg_prob', 'lookahead_avg']:
        if m not in combined.columns:
            continue
        valid = combined[[m, 'is_non_comp']].dropna()
        if len(valid) < 20 and valid['is_non_comp'].nunique() >= 2:
            continue
        scores = valid[m].values
        try:
            auc = roc_auc_score(valid['is_non_comp'].astype(int), scores)
            print(f"  {m}: AUC = {auc:.3f}")
            stat_rows.append({'test': 'auc_combined', 'metric': m, 'auc': auc})
        except Exception:
            pass

    return pd.DataFrame(stat_rows)


def create_plots(df):
    """Create publication-quality plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

    # 1. Box plots of key metrics by category
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metric_titles = [
        ('ntp_avg_prob', 'Next-Token Probability\n(within phrase)', axes[0]),
        ('fl_mid_avg_prob', 'Future Lens Probability\n(from 1st token, middle layer)', axes[1]),
        ('ntp_avg_surprisal', 'Surprisal (bits)\n(within phrase)', axes[2]),
    ]

    cat_order = ['control', 'farahmand', 'idiomem']
    cat_labels = {'control': 'Control\nBigrams', 'farahmand': 'Farahmand\nCompounds', 'idiomem': 'IdioMem\nIdioms'}
    colors = {'control': '#2196F3', 'farahmand': '#9C27B0', 'idiomem': '#F44336'}

    for metric, title, ax in metric_titles:
        if metric not in df.columns:
            continue
        data = []
        labels = []
        cols = []
        for cat in cat_order:
            vals = df[df['category'] == cat][metric].dropna().values
            if len(vals) > 0:
                data.append(vals)
                labels.append(cat_labels.get(cat, cat))
                cols.append(colors.get(cat, '#999'))

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)
            for patch, c in zip(bp['boxes'], cols):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_title(title)

    plt.suptitle('Phrase Predictability by Category', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'metrics_by_category.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics_by_category.png")

    # 2. Logit lens profile across layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 2a: Next-token prediction by layer
    ax = axes[0]
    layer_cols = sorted([c for c in df.columns if c.startswith('ll_L') and c.endswith('_avg')])
    layer_nums = [int(c.split('_L')[1].split('_')[0]) for c in layer_cols]

    for cat, label, color, marker in [
        ('control', 'Control Bigrams', '#2196F3', 'o'),
        ('farahmand', 'Farahmand (all)', '#9C27B0', 's'),
        ('idiomem', 'IdioMem Idioms', '#F44336', '^'),
    ]:
        subset = df[df['category'] == cat]
        means = [subset[c].dropna().mean() for c in layer_cols]
        sems = [subset[c].dropna().std() / np.sqrt(max(len(subset[c].dropna()), 1)) for c in layer_cols]
        ax.errorbar(layer_nums, means, yerr=sems, label=label, marker=marker,
                    color=color, capsize=3, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Avg Next-Token Prob (Logit Lens)')
    ax.set_title('Logit Lens: Within-Phrase Prediction by Layer')
    ax.legend(fontsize=9)

    # 2b: Future Lens k=1 from first token
    ax = axes[1]
    fl_k1_cols = sorted([c for c in df.columns if c.startswith('fl_L') and '_k1_prob' in c])
    fl_layers = [int(c.split('_L')[1].split('_')[0]) for c in fl_k1_cols]

    for cat, label, color, marker in [
        ('control', 'Control Bigrams', '#2196F3', 'o'),
        ('farahmand', 'Farahmand (all)', '#9C27B0', 's'),
        ('idiomem', 'IdioMem Idioms', '#F44336', '^'),
    ]:
        subset = df[df['category'] == cat]
        means = [subset[c].dropna().mean() for c in fl_k1_cols]
        ax.plot(fl_layers, means, label=label, marker=marker, color=color, linewidth=1.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Prob of 2nd Token from 1st (k=1)')
    ax.set_title('Future Lens: Predict Next Token from First Token')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'layer_profiles.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved layer_profiles.png")

    # 3. Farahmand compositionality score vs metrics scatter
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    far = df[df['category'] == 'farahmand']

    for ax, (metric, title) in zip(axes, [
        ('ntp_avg_prob', 'Next-Token Prob'),
        ('fl_mid_avg_prob', 'Future Lens Prob (mid layer)'),
        ('lookahead_avg', 'Lookahead Prob'),
    ]):
        if metric not in far.columns:
            continue
        valid = far[[metric, 'non_comp_score']].dropna()
        if len(valid) < 5:
            continue
        ax.scatter(valid['non_comp_score'], valid[metric], alpha=0.4, s=30, color='#673AB7')
        r, p = stats.spearmanr(valid['non_comp_score'], valid[metric])
        ax.set_xlabel('Non-Compositionality Score')
        ax.set_ylabel(title)
        ax.set_title(f'r={r:.3f}, p={p:.4f}')

    plt.suptitle('Compositionality Score vs Prediction Metrics (Farahmand)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'compositionality_scatter.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved compositionality_scatter.png")

    # 4. Non-comp advantage across layers (Farahmand)
    fig, ax = plt.subplots(figsize=(10, 5))
    comp = far[far['label'] == 'compositional']
    ncomp = far[far['label'] == 'non-compositional']

    if len(comp) >= 3 and len(ncomp) >= 3:
        diffs = []
        for c in layer_cols:
            cm = comp[c].dropna().mean()
            nm = ncomp[c].dropna().mean()
            diffs.append(nm - cm)

        colors_bar = ['#F44336' if d > 0 else '#2196F3' for d in diffs]
        ax.bar(range(len(layer_nums)), diffs, tick_label=[str(l) for l in layer_nums],
               color=colors_bar, alpha=0.7)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Δ Logit Lens Prob (NonComp − Comp)')
        ax.set_title('Non-Compositional Advantage in Logit Lens (Farahmand)')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'noncomp_advantage.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved noncomp_advantage.png")

    # 5. Distribution comparison: idioms vs control
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(axes,
        ['ntp_avg_prob', 'fl_mid_avg_prob'],
        ['Next-Token Probability', 'Future Lens Probability (mid layer)']):
        if metric not in df.columns:
            continue
        for cat, label, color in [('control', 'Control Bigrams', '#2196F3'),
                                   ('idiomem', 'Idioms', '#F44336')]:
            vals = df[df['category'] == cat][metric].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=25, alpha=0.5, label=f'{label} (n={len(vals)})',
                        color=color, density=True)
        ax.set_xlabel(title)
        ax.set_ylabel('Density')
        ax.legend()

    plt.suptitle('Distribution of Prediction Metrics', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'distribution_comparison.png', bbox_inches='tight')
    plt.close()
    print(f"  Saved distribution_comparison.png")


def main():
    t0 = time.time()
    print("=" * 70)
    print("Future Lens v3: Non-Compositional Phrase Detection")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Seed: {SEED}")

    model, tokenizer = load_model("gpt2-xl")
    n_layers = model.config.n_layer

    # Layers to probe: sample across depth
    layers = [0, 6, 12, 18, 24, 30, 36, 42, 47]

    compounds = load_farahmand_ncs()
    idioms = load_idiomem()
    entries = prepare_data(compounds, idioms, tokenizer)

    print(f"\nRunning evaluation on {len(entries)} phrases...")
    df = run_experiment(model, tokenizer, entries, layers)
    df.to_csv(RESULTS_DIR / 'future_lens_v3_results.csv', index=False)
    print(f"Results saved: {len(df)} rows")

    # Analysis
    stat_df = statistical_analysis(df)
    stat_df.to_csv(RESULTS_DIR / 'statistical_results_v3.csv', index=False)

    # Plots
    print("\nCreating visualizations...")
    create_plots(df)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Final summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    far = df[df['category'] == 'farahmand']
    ctrl = df[df['category'] == 'control']
    idiom = df[df['category'] == 'idiomem']

    for metric, name in [('ntp_avg_prob', 'Next-Token Prob'),
                          ('fl_mid_avg_prob', 'Future Lens (mid layer)'),
                          ('lookahead_avg', 'Lookahead')]:
        if metric not in df.columns:
            continue
        print(f"\n{name}:")
        for cat, sub, lbl in [('Control', ctrl, 'control'),
                                ('Farahmand comp', far[far['label']=='compositional'], 'comp'),
                                ('Farahmand non-comp', far[far['label']=='non-compositional'], 'ncomp'),
                                ('Idioms', idiom, 'idiom')]:
            vals = sub[metric].dropna()
            if len(vals) > 0:
                print(f"  {cat}: {vals.mean():.4f} ± {vals.std():.4f} (n={len(vals)})")

    return df


if __name__ == "__main__":
    main()
