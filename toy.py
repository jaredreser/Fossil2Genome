#!/usr/bin/env python3
"""
paleo-genome-toy: Single-file forward + inverse demo on synthetic data.

Forward: genome tokens (A,C,G,T) -> morphology embedding (tiny conv/MLP)
Inverse: optimize per-position token probabilities to match target morphology

Run:
  pip install torch numpy
  python toy.py
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot_tokens(tokens, vocab=4):
    """
    tokens: (B, L) long ints in [0..vocab-1]
    returns: (B, vocab, L) float32 one-hot
    """
    B, L = tokens.shape
    x = torch.zeros((B, vocab, L), dtype=torch.float32, device=tokens.device)
    # index shape must match x across non-scatter dims -> (B, 1, L)
    x.scatter_(1, tokens.unsqueeze(1), 1.0)
    return x


def edit_distance(a, b):
    """Levenshtein distance between two 1D int arrays (CPU)."""
    a = np.asarray(a, dtype=np.int32)
    b = np.asarray(b, dtype=np.int32)
    L1, L2 = len(a), len(b)
    dp = np.zeros((L1 + 1, L2 + 1), dtype=np.int32)
    for i in range(L1 + 1):
        dp[i, 0] = i
    for j in range(L2 + 1):
        dp[0, j] = j
    for i in range(1, L1 + 1):
        for j in range(1, L2 + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[L1, L2])


# -----------------------------
# Synthetic ground-truth generator
# -----------------------------

def make_synthetic_dataset(n_train=2000, n_val=200, L=128, morph_dim=32, seed=123):
    """
    Create random genomes (0..3) and a fixed "true" linear mapping from
    flattened one-hot genomes to morphology (plus tiny noise). This defines the
    target function the forward model should learn.
    """
    rng = np.random.default_rng(seed)
    train_gen = rng.integers(0, 4, size=(n_train, L), dtype=np.int64)
    val_gen   = rng.integers(0, 4, size=(n_val, L), dtype=np.int64)

    # True random projection from flattened one-hot (L*4) -> morph_dim
    W = rng.normal(0, 1 / np.sqrt(L * 4), size=(L * 4, morph_dim)).astype(np.float32)
    b = rng.normal(0, 0.01, size=(morph_dim,)).astype(np.float32)

    def oh_flat(G):
        B = G.shape[0]
        out = np.zeros((B, L * 4), dtype=np.float32)
        for i in range(B):
            # Put a 1 at the index for each position’s chosen base (A/C/G/T)
            out[i, (np.arange(L) * 4 + G[i])] = 1.0
        return out

    train_morph = oh_flat(train_gen) @ W + b + rng.normal(0, 0.01, size=(n_train, morph_dim)).astype(np.float32)
    val_morph   = oh_flat(val_gen)   @ W + b + rng.normal(0, 0.01, size=(n_val, morph_dim)).astype(np.float32)

    return {
        "train_gen": train_gen,
        "val_gen": val_gen,
        "train_morph": train_morph,
        "val_morph": val_morph,
        "L": L,
        "morph_dim": morph_dim,
    }


# -----------------------------
# Forward model (differentiable w.r.t. soft tokens)
# -----------------------------

class ForwardGenomeToMorph(nn.Module):
    """
    Input: (B, Vocab=4, L) soft one-hot (probabilities per position)
    Layers: Conv1d over channels=4, global pooling, MLP to morph_dim.
    """
    def __init__(self, vocab=4, conv_dim=64, morph_dim=32, k=9):
        super().__init__()
        self.conv = nn.Conv1d(vocab, conv_dim, kernel_size=k, padding=k // 2)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(conv_dim, 128),
            nn.GELU(),
            nn.Linear(128, morph_dim),
        )

    def forward(self, x):  # x: (B, 4, L)
        h = self.act(self.conv(x))
        h = self.pool(h).squeeze(-1)  # (B, conv_dim)
        m = self.head(h)              # (B, morph_dim)
        return m


# -----------------------------
# Training and inversion
# -----------------------------

def train_forward(model, train_gen, train_morph, val_gen, val_morph, epochs=6, bs=128, lr=3e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(train_gen).long().to(device)
    Ytr = torch.from_numpy(train_morph).float().to(device)
    Xva = torch.from_numpy(val_gen).long().to(device)
    Yva = torch.from_numpy(val_morph).float().to(device)

    def batches(X, Y, bsz):
        N = X.shape[0]
        idx = torch.randperm(N, device=X.device)
        for i in range(0, N, bsz):
            j = idx[i:i + bsz]
            yield X[j], Y[j]

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for Xt, Yt in batches(Xtr, Ytr, bs):
            x = one_hot_tokens(Xt)  # (B, 4, L)
            pred = model(x)
            loss = loss_fn(pred, Yt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(Xt); n += len(Xt)
        tr_loss = tot / n

        model.eval()
        with torch.no_grad():
            x = one_hot_tokens(Xva)
            pred = model(x)
            va_loss = float(loss_fn(pred, Yva))

        print(f"[epoch {ep}] train {tr_loss:.4f} | val {va_loss:.4f}")
    return model


def invert_morphology(model, target_morph, L, steps=400, lr=0.6, entropy_w=1e-3):
    """
    Optimize per-position logits -> softmax probabilities (L,4),
    pass through forward model, match target morphology.
    Returns argmax-discretized genome tokens (L,).
    """
    device = next(model.parameters()).device
    target = torch.as_tensor(target_morph, dtype=torch.float32, device=device).unsqueeze(0)  # (1, M)

    logits = torch.zeros((L, 4), dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    loss_fn = nn.MSELoss()

    for t in range(1, steps + 1):
        P = F.softmax(logits, dim=-1)             # (L, 4)
        x = P.transpose(0, 1).unsqueeze(0)        # (1, 4, L)
        pred = model(x)                            # (1, M)
        fit = loss_fn(pred, target)

        # Encourage low-entropy (near one-hot) distributions per position
        ent = -(P * (P.clamp_min(1e-8)).log()).sum(dim=-1).mean()
        loss = fit + entropy_w * ent

        opt.zero_grad(); loss.backward(); opt.step()

        if t % 50 == 0 or t <= 5:
            print(f"[invert step {t:3d}] loss {float(loss):.4f} (fit {float(fit):.4f}, ent {float(ent):.4f})")

    with torch.no_grad():
        P = F.softmax(logits, dim=-1)             # (L, 4)
        tokens = torch.argmax(P, dim=-1).detach().cpu().numpy()
    return tokens


# -----------------------------
# Main demo
# -----------------------------

def main():
    set_seed(123)
    # hyperparams
    L = 128
    morph_dim = 32
    data = make_synthetic_dataset(n_train=2000, n_val=200, L=L, morph_dim=morph_dim, seed=123)
    print("Synthetic data shapes:",
          {k: (np.array(v).shape if hasattr(v, "shape") else type(v)) for k, v in data.items()})

    model = ForwardGenomeToMorph(vocab=4, conv_dim=64, morph_dim=morph_dim, k=9)
    model = train_forward(
        model,
        data["train_gen"], data["train_morph"],
        data["val_gen"],   data["val_morph"],
        epochs=6, bs=128, lr=3e-3
    )

    # Choose one validation example as the "fossil morphology" to invert
    idx = 7
    true_gen = data["val_gen"][idx]
    target_morph = data["val_morph"][idx]

    print("\nStarting inversion...")
    cand = invert_morphology(model, target_morph, L=L, steps=400, lr=0.6, entropy_w=1e-3)

    # Evaluate
    ed = edit_distance(true_gen, cand)
    match = 1.0 - ed / len(true_gen)
    print(f"\nRESULTS: edit distance = {ed}  |  match fraction ≈ {match:.3f}")
    print("True   (first 64):", true_gen[:64])
    print("Cand.  (first 64):", cand[:64])


if __name__ == "__main__":
    main()
