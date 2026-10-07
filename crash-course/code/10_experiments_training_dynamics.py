"""Training-dynamics experiments where the textbook answer is wrong or incomplete.

Each section prints an observation you can quote as something you measured yourself.
Pure CPU, ~20 seconds total.

Run:  python code/10_experiments_training_dynamics.py
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)


def section(t):
    print(f"\n=== {t} ===")


# ---------------------------------------------------------------------------- A
section("A. bf16 silently drops small updates (why fp32 master weights exist)")
# bf16 keeps 8 significant bits -> relative precision ~0.4%. Any update smaller than
# ~half an ulp of the weight rounds to *no change at all*. It is not 'a bit of noise'.
w = torch.randn(1_000_000) * 0.02                     # typical init scale of a transformer weight
wb = w.bfloat16()
for lr in (1e-3, 1e-4, 1e-5, 1e-6):
    upd = lr * torch.sign(torch.randn_like(w))       # Adam steps have magnitude ~lr per parameter
    changed = ((wb + upd.bfloat16()) != wb).float().mean().item()
    print(f"  weights ~N(0,0.02), Adam-sized update {lr:.0e}: {changed:5.1%} of bf16 weights actually change "
          f"(fp32: {((w + upd) != w).float().mean():.0%})")
print("  -> at the end of a cosine schedule (lr ~1e-5..1e-6) pure-bf16 weights stop learning; keep an fp32 master copy,\n"
      "     or use stochastic rounding / Kahan summation in the optimizer.")

print(f"  Adam eps=1e-8 stored in fp16 becomes {torch.tensor(1e-8, dtype=torch.float16).item()} "
      f"(fp16 min subnormal ~6e-8) -> division by sqrt(v)+0 for zero-gradient params = NaN/inf; bf16 keeps it: "
      f"{torch.tensor(1e-8, dtype=torch.bfloat16).item():.1e}")
print(f"  exp(12) in fp16 = {torch.exp(torch.tensor(12.0, dtype=torch.float16)).item()} -> a softmax without the max-subtraction "
      f"overflows at logit 11.09; 300^2 in fp16 = {(torch.tensor(300.0, dtype=torch.float16) ** 2).item()}")
s = np.float16(0)
for _ in range(5000):
    s = np.float16(s + np.float16(1))                 # accumulate 5000 ones in fp16
print(f"  summing 5000 ones in fp16 gives {float(s):.0f}: once the sum hits 2048 the ulp is 2 and +1 rounds away.\n"
      "  -> reductions (loss sums, LayerNorm variance, softmax denominators) must run in fp32: that is what autocast does.")


# ---------------------------------------------------------------------------- B
section("B. with normalization, weight decay sets the effective learning rate; it is not a regularizer")


class NormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Linear(32, 64, bias=False)
        self.ln = nn.LayerNorm(64, elementwise_affine=False)   # LN(aWx) == LN(Wx): the function ignores |W|
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        return self.out(F.relu(self.ln(self.W(x))))


X = torch.randn(4096, 32)
y = (X[:, :4].sum(1) + 1.5 * torch.randn(4096) > 0).long()   # label noise: the loss cannot reach 0, gradients persist
net = NormNet()
with torch.no_grad():
    out1 = net(X)
    net.W.weight.mul_(3.0)                          # scale W by 3
    out3 = net(X)
    net.W.weight.div_(3.0)
print(f"  f(3W) == f(W): {torch.allclose(out1, out3, atol=1e-4)}   (scale invariance through LayerNorm)")


def grad_W(scale):
    net.zero_grad()
    with torch.no_grad():
        net.W.weight.mul_(scale)
    F.cross_entropy(net(X), y).backward()
    g = net.W.weight.grad.clone()
    with torch.no_grad():
        net.W.weight.div_(scale)
    return g


g1, g3 = grad_W(1.0), grad_W(3.0)
print(f"  |grad at 3W| / |grad at W| = {g3.norm() / g1.norm():.3f}  (gradient shrinks as 1/|W|)")
print(f"  cos(grad, W) = {F.cosine_similarity(g1.flatten(), net.W.weight.flatten(), dim=0):.4f}  (gradient is orthogonal to W)")
print("  => each SGD step can only *increase* |W| (|W+g|^2 = |W|^2 + |g|^2), which shrinks the effective step lr/|W|^2.")


def train(wd, steps=1500, lr=0.5, batch=128):
    torch.manual_seed(0)
    m = NormNet()
    opt = torch.optim.SGD([{"params": m.W.parameters(), "weight_decay": wd},
                           {"params": m.out.parameters(), "weight_decay": 0.0}], lr=lr)
    norms = []
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,))
        opt.zero_grad()
        loss = F.cross_entropy(m(X[idx]), y[idx])
        loss.backward()
        norms.append(m.W.weight.norm().item())
        opt.step()
    with torch.no_grad():
        full = F.cross_entropy(m(X), y).item()
    return np.array(norms), full


for wd in (0.0, 1e-2, 5e-2):
    norms, final = train(wd)
    eff = norms[0] ** 2 / norms ** 2                   # effective LR relative to step 0
    print(f"  weight_decay={wd:<5}: |W| {norms[0]:.1f} -> {norms[300]:.1f} -> {norms[800]:.1f} -> {norms[-1]:.1f};  "
          f"effective lr at the end = {eff[-1]:.2f}x initial;  train loss {final:.3f}")
print("  -> without decay the norm drifts up and the effective lr decays on its own; decay pins the norm at an equilibrium\n"
      "     where decay balances the orthogonal growth. The wd=0.05 run has the *worst* loss: not because it regularized,\n"
      "     but because the effective step became 7x too large. With LayerNorm/BatchNorm everywhere, lr and weight decay\n"
      "     are one knob with two names (what matters is lr*wd), and 'remove weight decay' == 'use a decaying lr'.")


# ---------------------------------------------------------------------------- C
section("C. Adam after a quiet period: the mechanism behind loss spikes, and why LLMs use beta2=0.95")


def adam_update_sizes(grads, b1=0.9, b2=0.999, eps=1e-8):
    m = v = 0.0
    out = []
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        out.append((m / (1 - b1 ** t)) / (math.sqrt(v / (1 - b2 ** t)) + eps))   # in units of lr
    return np.abs(np.array(out))


quiet, spike = 3000, 600
for ratio in (1000, 10, 3):
    grads = np.concatenate([np.full(quiet, 1e-3), np.full(spike, 1e-3 * ratio)])
    for b2 in (0.999, 0.95):
        u = adam_update_sizes(grads, b2=b2)[quiet:]
        print(f"  gradient jumps x{ratio:<4} beta2={b2}: steady step = 1.0 lr, peak step after the jump = {u.max():.1f} lr, "
              f"steps above 2 lr: {(u > 2).sum():3d}")
print("  -> v (the per-parameter scale) remembers the quiet past for ~1/(1-beta2) steps while m adapts in ~10 steps,\n"
      "     so the first large gradients get amplified ~6x with beta2=0.999 and ~1x with 0.95. Gradient clipping only helps\n"
      "     if the clip is tight relative to the *quiet* gradient level (x3 row). This is why GPT-3/LLaMA use beta2=0.95,\n"
      "     why clipping matters mostly to protect v, and one reason loss spikes follow quiet, low-gradient stretches.")


# ---------------------------------------------------------------------------- D
section("D. the critical batch size grows during training (gradient noise scale)")
Xd = torch.randn(4000, 20)
yd = (Xd @ torch.randn(20, 10) + 1.0 * torch.randn(4000, 10)).argmax(1)   # 10 classes with label noise
net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.SGD(net.parameters(), lr=0.1)


def grad_sq_norm(idx):
    opt.zero_grad()
    F.cross_entropy(net(Xd[idx]), yd[idx]).backward()
    return sum((p.grad ** 2).sum().item() for p in net.parameters())


def noise_scale(B_small=16, B_big=1024, reps=40):
    # E|g_B|^2 = |G|^2 + tr(Sigma)/B  -> two batch sizes give two equations (McCandlish et al. 2018)
    gs = np.mean([grad_sq_norm(torch.randint(0, 4000, (B_small,))) for _ in range(reps)])
    gb = np.mean([grad_sq_norm(torch.randint(0, 4000, (B_big,))) for _ in range(reps)])
    tr_sigma = (gs - gb) / (1 / B_small - 1 / B_big)
    G2 = gb - tr_sigma / B_big
    return tr_sigma / G2 if G2 > 0 else float("inf")


step = 0
for target in (0, 30, 300, 1500):
    while step < target:
        idx = torch.randint(0, 4000, (64,))
        opt.zero_grad()
        F.cross_entropy(net(Xd[idx]), yd[idx]).backward()
        opt.step()
        step += 1
    with torch.no_grad():
        loss = F.cross_entropy(net(Xd), yd).item()
    B = noise_scale()
    print(f"  step {step:4d}: train loss {loss:.3f}   critical batch size ~ {B:,.0f}" if np.isfinite(B)
          else f"  step {step:4d}: train loss {loss:.3f}   critical batch size -> very large (signal below noise)")
print("  -> early on, a batch of 64 already captures most of the signal; late in training you need thousands of examples\n"
      "     per step to beat the noise. Same model, same data: the *right* batch size is a function of training progress.\n"
      "     (This is why GPT-3 ramped batch size 32k -> 3.2M tokens, and why 'bigger batch' wastes compute early.)")


# ---------------------------------------------------------------------------- E
section("E. double descent: test error peaks when parameters ~ samples, then falls again")
rng = np.random.default_rng(0)
n_train, n_test, d_in = 100, 3000, 10
Xtr, Xte = rng.normal(size=(n_train, d_in)), rng.normal(size=(n_test, d_in))
beta = rng.normal(size=d_in)
f = lambda X: np.tanh(X @ beta / 2) + 0.5 * np.sin(X[:, 0] * 2)
ytr, yte = f(Xtr) + 0.1 * rng.normal(size=n_train), f(Xte)
W, b = 0.5 * rng.normal(size=(d_in, 2000)), rng.uniform(0, 2 * np.pi, 2000)
feats = lambda X, D: np.cos(X @ W[:, :D] + b[:D])                 # random Fourier features
print(f"  (variance of the target = {yte.var():.3f}; n_train = {n_train})")
for D in (10, 30, 60, 90, 100, 110, 150, 300, 1000, 2000):
    Ftr, Fte = feats(Xtr, D), feats(Xte, D)
    coef = np.linalg.lstsq(Ftr, ytr, rcond=None)[0]             # min-norm least squares; interpolates once D >= n
    coef_r = np.linalg.solve(Ftr.T @ Ftr + 1.0 * np.eye(D), Ftr.T @ ytr)
    print(f"  features D={D:5d}: test MSE no-ridge {((Fte @ coef - yte) ** 2).mean():7.3f}   ridge {((Fte @ coef_r - yte) ** 2).mean():6.3f}")
print("  -> the classical 'more parameters = more overfitting' story holds only up to D ~ n. Past it the min-norm solution\n"
      "     gets smoother again. Regularization removes the peak. The practical consequence: near the interpolation\n"
      "     threshold, adding data or parameters can *hurt*, and 'val loss went up when I added data' is not always a bug.")
print("\nDone.")
