"""Scaled dot-product attention, multi-head attention, a transformer block, and a
KV cache -- written from scratch and checked against PyTorch's built-ins.

Run:  python code/04_attention_from_scratch.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ------------------------------------------------ 1. attention in NumPy (the formula)
print("=== 1. scaled dot-product attention in NumPy ===")


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def attention_np(Q, K, V, causal=False):
    """Q: (n, d), K: (m, d), V: (m, dv). Each query mixes values by softmaxed similarity to keys."""
    d = Q.shape[-1]
    scores = Q @ K.T / math.sqrt(d)               # (n, m); /sqrt(d) keeps variance ~1 -> softmax not saturated
    if causal:                                     # token i may only look at tokens <= i
        scores = np.where(np.tril(np.ones_like(scores)) == 1, scores, -np.inf)
    weights = softmax(scores)                      # rows sum to 1
    return weights @ V, weights


rng = np.random.default_rng(0)
n, d = 4, 8
Q, K, V = rng.normal(size=(n, d)), rng.normal(size=(n, d)), rng.normal(size=(n, d))
out, w = attention_np(Q, K, V, causal=True)
print("causal attention weights (lower-triangular, rows sum to 1):")
print(np.round(w, 2))
assert np.allclose(w.sum(1), 1) and np.allclose(np.triu(w, 1), 0)

# why sqrt(d): without it, logits have variance d and softmax becomes near one-hot
big_d = 512
Qb, Kb = rng.normal(size=(1, big_d)), rng.normal(size=(50, big_d))
print(f"logit std without scaling: {(Qb @ Kb.T).std():.1f}   with /sqrt(d): {(Qb @ Kb.T / math.sqrt(big_d)).std():.2f}")

# ------------------------------------------------ 2. check against torch's fused kernel
print("\n=== 2. matches F.scaled_dot_product_attention (FlashAttention entry point) ===")
Qt, Kt, Vt = (torch.tensor(a, dtype=torch.float32)[None, None] for a in (Q, K, V))   # (B, heads, n, d)
ref = F.scaled_dot_product_attention(Qt, Kt, Vt, is_causal=True)
assert torch.allclose(ref[0, 0], torch.tensor(out, dtype=torch.float32), atol=1e-5)
print("ok")


# ------------------------------------------------ 3. multi-head attention + transformer block
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)      # one fused projection for Q, K, V (3d^2 params)
        self.out = nn.Linear(d_model, d_model)          # output projection (d^2 params)

    def forward(self, x, kv_cache=None):
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, n, d) -> (B, heads, n, dk): each head attends in its own subspace
        q, k, v = (t.view(B, -1, self.h, self.dk).transpose(1, 2) for t in (q, k, v))
        if kv_cache is not None:                        # inference: append to cached keys/values
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=2)
                v = torch.cat([kv_cache["v"], v], dim=2)
            kv_cache["k"], kv_cache["v"] = k, v
        causal = kv_cache is None or q.shape[2] == k.shape[2]   # a single new token sees all cached keys
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        y = y.transpose(1, 2).reshape(B, n, d)          # concat heads
        return self.out(y)


class Block(nn.Module):
    """Pre-LN transformer block: x + Attn(LN(x)); x + FFN(LN(x))."""

    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
                                 nn.Linear(ffn_mult * d_model, d_model))   # 8d^2 params

    def forward(self, x, kv_cache=None):
        x = x + self.attn(self.ln1(x), kv_cache)        # residual: gradient highway
        x = x + self.ffn(self.ln2(x))
        return x


print("\n=== 3. parameter count of a block ~ 12 d^2 ===")
d_model, n_heads = 256, 8
block = Block(d_model, n_heads)
n_params = sum(p.numel() for p in block.parameters())
print(f"d={d_model}: params {n_params:,}  vs 12*d^2 = {12 * d_model**2:,}  (rest is biases + LayerNorm)")
assert abs(n_params - 12 * d_model**2) / (12 * d_model**2) < 0.02

# where the FLOPs go for a sequence of length n: attention scores O(n^2 d) vs linear layers O(n d^2)
for n_tokens in (128, 2048, 16384):
    attn_flops = 2 * 2 * n_tokens**2 * d_model          # QK^T and PV
    linear_flops = 2 * n_tokens * 12 * d_model**2       # all projections + FFN
    print(f"n={n_tokens:5d}: attention-score FLOPs / linear FLOPs = {attn_flops / linear_flops:.2f}")

# ------------------------------------------------ 4. KV cache: incremental decoding == full recompute
print("\n=== 4. KV cache gives the same result as recomputing the whole sequence ===")
block.eval()
x = torch.randn(1, 6, d_model)
with torch.no_grad():
    full = block(x)                                     # process all 6 tokens at once (prefill-style)
    cache, outs = {}, []
    for t in range(6):                                  # decode-style: one token at a time
        outs.append(block(x[:, t:t + 1], cache))
    incremental = torch.cat(outs, dim=1)
assert torch.allclose(full, incremental, atol=1e-5)
print(f"cached K shape: {tuple(cache['k'].shape)}  (B, heads, tokens, head_dim)")
per_token_bytes = 2 * cache["k"][0, :, 0].numel() * 2   # K and V, bf16 -> 2 bytes
print(f"KV cache per token for this 1-layer toy in bf16: {per_token_bytes} bytes; "
      f"LLaMA-2-7B (32 layers, 32 heads x 128): {2 * 32 * 32 * 128 * 2 / 2**20:.2f} MB")
print("Done.")
