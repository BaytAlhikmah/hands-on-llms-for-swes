"""
Session 3 scratch: Character embeddings — the bridge from V×V to dense vectors.

Session 2 ended with: "What if instead of a V×V matrix, we represented each
character as a small vector?" This is that.

1. Recap: load data, build the counting bigram (our ground truth)
2. Full-rank model: V×V matrix (d=V) — same as Session 2
3. Factored model: V×d embedding + d×V output (d < V) — the new idea
4. Compare probabilities: does the compressed model recover the same distribution?
5. Visualize: plot the 2D character embeddings — what did the model learn?
6. The punchline: same trick works for words, and that's how real LLMs start
"""

import random
import math
import urllib.request

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

random.seed(42)
torch.manual_seed(42)


# ── helpers ──────────────────────────────────────────────────────

def softmax(logits):
    """Row-wise softmax: logits is (*, V). Pure tensor ops."""
    maxes = logits.max(dim=-1, keepdim=True).values
    exp = (logits - maxes).exp()
    return exp / exp.sum(dim=-1, keepdim=True)


# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA (same as Session 2)
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("1. LOAD DATA")
print("=" * 60)

url = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
response = urllib.request.urlopen(url)
names = [n.strip().lower() for n in response.read().decode('utf-8').strip().split('\n') if n.strip()]

chars = sorted(set("".join(names)))
stoi = {'.': 0}
for i, c in enumerate(chars):
    stoi[c] = i + 1
itos = {i: c for c, i in stoi.items()}
V = len(stoi)

# Training pairs
xs, ys = [], []
for name in names:
    chs = ['.'] + list(name) + ['.']
    for i in range(len(chs) - 1):
        xs.append(stoi[chs[i]])
        ys.append(stoi[chs[i + 1]])

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(f"Names: {len(names):,}")
print(f"Vocab: {V} characters")
print(f"Training bigrams: {len(xs):,}")

# Counting bigram (ground truth)
counts = [[0] * V for _ in range(V)]
for name in names:
    chs = ['.'] + list(name) + ['.']
    for i in range(len(chs) - 1):
        counts[stoi[chs[i]]][stoi[chs[i + 1]]] += 1

P_counts = torch.zeros(V, V)
for i in range(V):
    row_total = sum(counts[i])
    if row_total > 0:
        for j in range(V):
            P_counts[i][j] = counts[i][j] / row_total

# Counting model loss
nll = 0.0
for i in range(len(xs)):
    p = P_counts[xs[i].item(), ys[i].item()].item()
    nll += -math.log(p) if p > 0 else 20.0
counting_loss = nll / len(xs)
print(f"Counting bigram loss: {counting_loss:.4f}")


# ══════════════════════════════════════════════════════════════════
# 2. FULL-RANK MODEL: V×V (Session 2 recap)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. FULL-RANK MODEL: V×V MATRIX (d = V = {})".format(V))
print("=" * 60)

print(f"""
This is what we did in Session 2: one big matrix W of shape {V}×{V}.
Row i gives the logits for "what comes after character i".
Parameters: {V}×{V} = {V * V}
""")

torch.manual_seed(42)
W_full = torch.randn((V, V), requires_grad=True)

lr = 50.0
for step in range(300):
    logits = W_full[xs]
    loss = F.cross_entropy(logits, ys)
    W_full.grad = None
    loss.backward()
    with torch.no_grad():
        W_full -= lr * W_full.grad
    if step % 100 == 0 or step == 299:
        print(f"  step {step:>3d}  loss = {loss.item():.4f}")

P_full = softmax(W_full.detach())


# ══════════════════════════════════════════════════════════════════
# 3. FACTORED MODEL: V×d + d×V (THE NEW IDEA)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. FACTORED MODEL: EMBEDDINGS")
print("=" * 60)

print(f"""
Instead of one V×V matrix, we split it into two smaller matrices:

  C : V × d   (the "embedding" — each character becomes a d-dim vector)
  W : d × V   (the "output"   — turns the vector back into scores)

To get logits for character i:
  1. Look up row i of C  →  a vector of d numbers
  2. Multiply by W       →  a vector of V scores (logits)

This is: logits = C @ W  (a V×V matrix, but built from two small ones)

If d = V, this can represent any V×V matrix.
If d < V, the model is forced to compress: similar characters
must end up with similar embedding vectors.

Key insight: we build the FULL logit matrix (C @ W), then index into it.
Autograd sees both C and W as leaf tensors and can compute gradients for both.
""")


def train_factored(d, steps=2000, lr=10.0, seed=42):
    """Train a factored bigram: C (V×d) @ W (d×V)."""
    torch.manual_seed(seed)
    # Small init — two matrices multiplied together means gradients
    # flow through a product, so we keep values small to start
    C = (torch.randn((V, d)) * 0.01).requires_grad_(True)
    W = (torch.randn((d, V)) * 0.01).requires_grad_(True)

    n_params = V * d + d * V
    print(f"  d = {d:>2d}  →  params: {V}×{d} + {d}×{V} = {n_params}"
          f"  (vs {V * V} for full V×V)")

    losses = []
    for step in range(steps):
        # Build the full V×V logit matrix, then index into it
        logit_matrix = C @ W              # (V, V)
        logits = logit_matrix[xs]         # (N, V) — look up row per input
        loss = F.cross_entropy(logits, ys)

        C.grad = None
        W.grad = None
        loss.backward()
        with torch.no_grad():
            C -= lr * C.grad
            W -= lr * W.grad

        losses.append(loss.item())
        if step % 200 == 0 or step == steps - 1:
            print(f"    step {step:>3d}  loss = {loss.item():.4f}")

    P = softmax((C @ W).detach())
    return C.detach(), W.detach(), P, losses


# Train at several dimensions to show the compression tradeoff
results = {}
for d in [2, 5, 10, 15, 27]:
    print(f"\n--- d = {d} ---")
    C, W_out, P, losses = train_factored(d)
    results[d] = {'C': C, 'W': W_out, 'P': P, 'losses': losses}


# ══════════════════════════════════════════════════════════════════
# 4. COMPARE PROBABILITIES
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("4. COMPARE: DO THEY RECOVER THE SAME DISTRIBUTION?")
print("=" * 60)

print(f"\n  {'model':>20s}  {'params':>8s}  {'loss':>8s}  {'max diff':>10s}  {'mean diff':>10s}")
print(f"  {'-' * 62}")

# Counting baseline
print(f"  {'counting':>20s}  {'—':>8s}  {counting_loss:>8.4f}  {'—':>10s}  {'—':>10s}")

# Full V×V
diff = (P_counts - P_full).abs()
full_loss = F.cross_entropy(W_full.detach()[xs], ys).item()
print(f"  {'full V×V (d=27)':>20s}  {V * V:>8d}  {full_loss:>8.4f}"
      f"  {diff.max().item():>10.6f}  {diff.mean().item():>10.6f}")

# Factored models
for d in [2, 5, 10, 15, 27]:
    P = results[d]['P']
    diff = (P_counts - P).abs()
    final_loss = results[d]['losses'][-1]
    params = 2 * V * d
    print(f"  {'factored d=' + str(d):>20s}  {params:>8d}  {final_loss:>8.4f}"
          f"  {diff.max().item():>10.6f}  {diff.mean().item():>10.6f}")

print(f"""
At d=27, the factored model has the same capacity as the V×V matrix
and recovers the exact same probabilities. As d shrinks, the model
is forced to compress — but even d=10 gets very close.

d=2 is too small to represent all the patterns, but it's perfect
for visualization: each character IS a 2D point we can plot.
""")


# ══════════════════════════════════════════════════════════════════
# 5. DETAILED PROBABILITY COMPARISON (d=2 vs counting)
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("5. PROBABILITY COMPARISON: d=2 vs COUNTING")
print("=" * 60)

P_d2 = results[2]['P']

# P(next | '.') — what starts a name?
print("\nP(first char | '.') — top 10:")
print(f"  {'char':>6s}  {'counting':>10s}  {'d=2':>10s}  {'diff':>10s}")
print(f"  {'-' * 42}")
start_id = stoi['.']
c_row = P_counts[start_id]
p_row = P_d2[start_id]
top_idx = c_row.argsort(descending=True)[:10]
for idx in top_idx:
    ch = itos[idx.item()]
    pc = c_row[idx].item()
    pp = p_row[idx].item()
    print(f"  {repr(ch):>6s}  {pc:>10.4f}  {pp:>10.4f}  {abs(pc - pp):>10.6f}")

# P(next | 'm')
print("\nP(next | 'm') — top 10:")
print(f"  {'char':>6s}  {'counting':>10s}  {'d=2':>10s}  {'diff':>10s}")
print(f"  {'-' * 42}")
m_id = stoi['m']
c_row = P_counts[m_id]
p_row = P_d2[m_id]
top_idx = c_row.argsort(descending=True)[:10]
for idx in top_idx:
    ch = itos[idx.item()]
    pc = c_row[idx].item()
    pp = p_row[idx].item()
    print(f"  {repr(ch):>6s}  {pc:>10.4f}  {pp:>10.4f}  {abs(pc - pp):>10.6f}")

# P(next | 'q') — interesting because q almost always precedes u
print("\nP(next | 'q') — all non-zero:")
print(f"  {'char':>6s}  {'counting':>10s}  {'d=2':>10s}")
print(f"  {'-' * 30}")
q_id = stoi['q']
c_row = P_counts[q_id]
p_row = P_d2[q_id]
top_idx = c_row.argsort(descending=True)
for idx in top_idx:
    pc = c_row[idx].item()
    if pc < 0.001 and p_row[idx].item() < 0.001:
        continue
    ch = itos[idx.item()]
    print(f"  {repr(ch):>6s}  {pc:>10.4f}  {p_row[idx].item():>10.4f}")


# ══════════════════════════════════════════════════════════════════
# 6. VISUALIZE THE 2D EMBEDDINGS
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("6. VISUALIZE CHARACTER EMBEDDINGS (d=2)")
print("=" * 60)

C_2d = results[2]['C'].numpy()

# Define character groups for coloring
vowels = set('aeiou')
common_consonants = set('tnrsld')  # very common in English names
rare_consonants = set('qxz')
special = set('.')

def get_group(ch):
    if ch in special:
        return 'start/end (.)'
    elif ch in vowels:
        return 'vowels'
    elif ch in rare_consonants:
        return 'rare (q,x,z)'
    elif ch in common_consonants:
        return 'common consonants'
    else:
        return 'other consonants'

group_colors = {
    'start/end (.)': '#e74c3c',
    'vowels': '#3498db',
    'common consonants': '#2ecc71',
    'other consonants': '#95a5a6',
    'rare (q,x,z)': '#e67e22',
}

fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

# Plot each character
for i in range(V):
    ch = itos[i]
    group = get_group(ch)
    color = group_colors[group]
    ax.scatter(C_2d[i, 0], C_2d[i, 1], c=color, s=120, zorder=3, edgecolors='white', linewidth=0.5)
    ax.annotate(ch, (C_2d[i, 0], C_2d[i, 1]),
                fontsize=11, fontweight='bold', ha='center', va='center',
                color='white' if ch != '.' else 'white', zorder=4)

# Legend
for group, color in group_colors.items():
    ax.scatter([], [], c=color, s=80, label=group)
ax.legend(loc='best', fontsize=9, framealpha=0.9)

ax.set_xlabel("Embedding dimension 1", fontsize=11)
ax.set_ylabel("Embedding dimension 2", fontsize=11)
ax.set_title("Character Embeddings (d=2) — Learned from Bigram Data\n"
             "Characters that appear in similar contexts end up nearby",
             fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/Users/adel/Desktop/hands-on-llms-for-swes/scratch/session3/char_embeddings_2d.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: char_embeddings_2d.png")


# ══════════════════════════════════════════════════════════════════
# 7. LOSS CURVES: HOW MUCH DOES DIMENSION MATTER?
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("7. LOSS CURVES BY EMBEDDING DIMENSION")
print("=" * 60)

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

colors = {2: '#e74c3c', 5: '#e67e22', 10: '#3498db', 15: '#2ecc71', 27: '#9b59b6'}
for d in [2, 5, 10, 15, 27]:
    losses = results[d]['losses']
    ax.plot(losses, label=f'd={d} ({2 * V * d} params)', color=colors[d], linewidth=1.5)

ax.axhline(y=counting_loss, color='black', linestyle='--', linewidth=1, label=f'counting ({counting_loss:.4f})')
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Loss (cross-entropy)", fontsize=11)
ax.set_title("Embedding Dimension vs Loss\n"
             "Higher d = more capacity = closer to counting baseline", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=2.0)
plt.tight_layout()
plt.savefig("/Users/adel/Desktop/hands-on-llms-for-swes/scratch/session3/loss_by_dimension.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Saved: loss_by_dimension.png")


# ══════════════════════════════════════════════════════════════════
# 8. THE PUNCHLINE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("8. THE PUNCHLINE")
print("=" * 60)

print(f"""
What we just did:
  - Took the V×V bigram matrix from Session 2
  - Factored it into C (V×d) and W (d×V)
  - Each character is now a vector of d numbers — an EMBEDDING
  - Characters that behave similarly get similar vectors
  - Even d=2 captures most of the structure (loss {results[2]['losses'][-1]:.4f} vs {counting_loss:.4f})

The parameter savings:
  V×V full matrix:  {V}×{V} = {V * V:>6d} parameters
  Factored d=10:    {V}×10 + 10×{V} = {2 * V * 10:>6d} parameters  ({2 * V * 10 / (V * V) * 100:.0f}% of full)
  Factored d=2:     {V}×2  + 2×{V}  = {2 * V * 2:>6d} parameters  ({2 * V * 2 / (V * V) * 100:.0f}% of full)

Now scale this up to words:
  V = 50,000 words (GPT-2 vocab)
  Full V×V:     50000² = {50000 ** 2:,} parameters (10 GB)
  Factored d=768: 50000×768×2 = {50000 * 768 * 2:,} parameters ({50000 * 768 * 2 * 4 / 1e6:.0f} MB)

That's the same trick. Every language model starts with an embedding table.
The embedding table in GPT-2 IS this C matrix, just bigger.

Next: what if we look at MORE than one previous character?
Concatenate multiple embeddings → add a hidden layer → that's Bengio (2003).
""")
