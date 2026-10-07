"""Systems-level experiments: retrieval, serving, quantization, and experimentation
pitfalls that only show up when you run the thing.

~45 seconds on CPU.

Run:  python code/12_experiments_systems.py
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4, user_api="openmp")
rng = np.random.default_rng(0)
torch.manual_seed(0)


def section(t):
    print(f"\n=== {t} ===")


# ============================================================ 1. embedding spaces across retrains
section("1. two retrains of the same model are not in the same embedding space")
n_users, n_items, dim = 800, 500, 16
U_true, I_true = rng.normal(size=(n_users, 8)), rng.normal(size=(n_items, 8))
train_u, train_i, test = [], [], {}
for u in range(n_users):
    s = U_true[u] @ I_true.T
    p = np.exp(s - s.max()); p /= p.sum()
    items = rng.choice(n_items, 25, replace=False, p=p)
    train_u += [u] * 24; train_i += items[:-1].tolist(); test[u] = int(items[-1])
train_u, train_i = torch.tensor(train_u), torch.tensor(train_i)
seen = {u: set() for u in range(n_users)}
for u, i in zip(train_u.tolist(), train_i.tolist()):
    seen[u].add(i)


def train_mf(seed, epochs=15):
    torch.manual_seed(seed)
    P, Q = nn.Embedding(n_users, dim), nn.Embedding(n_items, dim)
    nn.init.normal_(P.weight, std=0.1); nn.init.normal_(Q.weight, std=0.1)
    opt = torch.optim.Adam(list(P.parameters()) + list(Q.parameters()), lr=0.01)
    for _ in range(epochs):
        perm = torch.randperm(len(train_u))
        for s in range(0, len(perm), 2048):
            idx = perm[s:s + 2048]
            j = torch.randint(0, n_items, (len(idx),))
            x = (P(train_u[idx]) * (Q(train_i[idx]) - Q(j))).sum(1)
            loss = -F.logsigmoid(x).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return P.weight.detach().numpy(), Q.weight.detach().numpy()


def recall_at_10(P, Q):
    scores = P @ Q.T
    for u in range(n_users):
        scores[u, list(seen[u])] = -np.inf
    top = np.argsort(-scores, axis=1)[:, :10]
    return np.mean([test[u] in top[u] for u in range(n_users)])


P1, Q1 = train_mf(1)
P2, Q2 = train_mf(2)
print(f"  run 1 users x run 1 items: Recall@10 {recall_at_10(P1, Q1):.3f}")
print(f"  run 2 users x run 2 items: Recall@10 {recall_at_10(P2, Q2):.3f}")
print(f"  run 1 users x run 2 items: Recall@10 {recall_at_10(P1, Q2):.3f}   <- user tower from one deploy, item index from another")
# the dot product is invariant to any rotation R applied to both sides, so independent runs differ by (at least) a rotation
Uu, _, Vt = np.linalg.svd(Q1.T @ Q2)
R = Uu @ Vt                                                  # orthogonal Procrustes: best rotation Q1 -> Q2
print(f"  after aligning run 1 onto run 2 with the best rotation: Recall@10 {recall_at_10(P1 @ R, Q2):.3f}")
print("  -> scores depend only on dot products, so the solution is defined up to a rotation (and more: a different local\n"
      "     optimum). A mismatched user-tower/item-index deploy returns near-random results with no error anywhere.\n"
      "     Deploy them atomically, version them together, and warm-start retrains from the previous weights so that\n"
      "     incremental index updates stay in a compatible space.")

# ============================================================ 2. MLP vs dot product
section("2. an MLP on [user; item] struggles to learn a dot product (Rendle et al. 2020)")
d, n_tr, n_te = 32, 60_000, 10_000
A = torch.randn(n_tr + n_te, d); B = torch.randn(n_tr + n_te, d)
y = (A * B).sum(1)
Xcat = torch.cat([A, B], 1)
mlp = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
bil = nn.Bilinear(d, d, 1)                                   # learns u^T W v: contains the dot product (W = I)
for name, model, inputs in (("MLP on concat", mlp, lambda i: mlp(Xcat[i]).squeeze(1)),
                            ("bilinear (dot-product family)", bil, lambda i: bil(A[i], B[i]).squeeze(1))):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(8):
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, 256):
            idx = perm[s:s + 256]
            loss = F.mse_loss(inputs(idx), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        te = torch.arange(n_tr, n_tr + n_te)
        mse = F.mse_loss(inputs(te), y[te]).item()
    print(f"  {name:<30} params {sum(p.numel() for p in model.parameters()):6d}   test MSE {mse:6.3f}   (target variance {y.var():.1f})")
print("  -> with 80x the parameters and the same data, the MLP is ~20x worse at a 32-d dot product. Multiplicative\n"
      "     interactions are hard for ReLU MLPs; this is why NCF-style results did not replicate against a\n"
      "     tuned dot product, why two-tower retrieval scores with a dot product (which ANN indexes also need), and why\n"
      "     ranking models add explicit crossing (FM / DCN cross layers / attention) instead of relying on 'deep = universal'.")

# ============================================================ 3. IVF approximate search
section("3. ANN search: recall vs. work (IVF from scratch), and where the recall goes missing")
N, D, n_q, n_list = 100_000, 64, 300, 100
centers = rng.normal(size=(300, D))
cl = rng.integers(0, 300, int(N * 0.95))                       # 95% of items live in clusters ...
base = np.concatenate([centers[cl] + 0.6 * rng.normal(size=(len(cl), D)),
                       3.0 * rng.normal(size=(N - len(cl), D))]).astype(np.float32)   # ... 5% are isolated 'tail' items
is_tail = np.r_[np.zeros(len(cl), bool), np.ones(N - len(cl), bool)]
base /= np.linalg.norm(base, axis=1, keepdims=True)
qidx = np.r_[rng.choice(np.where(~is_tail)[0], 200, replace=False), rng.choice(np.where(is_tail)[0], 100, replace=False)]
q_tail = np.r_[np.zeros(200, bool), np.ones(100, bool)]
queries = base[qidx] + 0.1 * rng.normal(size=(n_q, D)).astype(np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
t0 = time.perf_counter(); exact = np.argsort(-(queries @ base.T), axis=1)[:, :10]; t_exact = time.perf_counter() - t0
cent = base[rng.choice(N, n_list, replace=False)].copy()
for _ in range(10):                                           # k-means (cosine) for the coarse quantizer
    assign = np.argmax(base @ cent.T, axis=1)
    for c in range(n_list):
        m = assign == c
        if m.any():
            cent[c] = base[m].mean(0); cent[c] /= np.linalg.norm(cent[c])
lists = [np.where(assign == c)[0] for c in range(n_list)]
for nprobe in (1, 2, 5, 10):
    hits, work = np.zeros(n_q), 0
    probe = np.argsort(-(queries @ cent.T), axis=1)[:, :nprobe]
    for qi in range(n_q):
        cand = np.concatenate([lists[c] for c in probe[qi]])
        work += len(cand)
        top = cand[np.argsort(-(base[cand] @ queries[qi]))[:10]]
        hits[qi] = len(set(top) & set(exact[qi])) / 10
    print(f"  nprobe={nprobe:2d}: recall@10 on clustered queries {hits[~q_tail].mean():.3f}   on tail queries {hits[q_tail].mean():.3f}   "
          f"work {work / n_q / N:5.1%} of exact search")
print(f"  (exact search costs {t_exact * 1e3 / n_q:.2f} ms/query here)")
print("  -> at 5% of the work the index is near-perfect on the head and loses ~half of the tail: items that sit between\n"
      "     clusters. ANN recall is a tunable, but it is not uniform -- measure it on the slice you care about (new\n"
      "     items, niche content), because that is where the index silently fails.")

# ============================================================ 4. train/serve skew that monitoring cannot see
section("4. train/serve skew: feature distributions identical, rows wrong (a misaligned join)")
n = 60_000
days_since = rng.exponential(10, n)
spend, visits = rng.lognormal(3, 1, n), rng.poisson(5, n)
logit = -0.25 * days_since + 0.4 * np.log1p(spend) + 0.1 * visits - 1.5
yb = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
Xtab = np.c_[days_since, spend, visits]
tr = np.arange(n) < 40_000
gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0).fit(Xtab[tr], yb[tr])
served = Xtab[~tr].copy()
bad = rng.uniform(size=len(served)) < 0.3                     # 30% of requests get another user's feature row
served[bad] = served[rng.permutation(np.where(bad)[0])]        # (stale cache / wrong join key / batch misalignment)
for name, X_eval in (("features as trained", Xtab[~tr]), ("features as served", served)):
    p = gb.predict_proba(X_eval)[:, 1]
    print(f"  {name:<20} per-feature mean {np.round(X_eval.mean(0), 2)} std {np.round(X_eval.std(0), 2)}   "
          f"prediction mean {p.mean():.3f}   PR-AUC {average_precision_score(yb[~tr], p):.3f}")
print("  -> every marginal statistic is identical (drift alerts stay green, prediction distribution unchanged) while\n"
      "     30% of predictions are for the wrong user. Distribution monitoring catches outages; only row-level parity\n"
      "     checks (same entity, same timestamp, offline vs online value) or training on logged served features\n"
      "     catch this class of bug.")

# ============================================================ 5. int8 outlier channels
section("5. int8 activation quantization breaks on outlier channels; SmoothQuant moves the difficulty into the weights")
T, C = 512, 1024
Xa = rng.normal(size=(T, C)).astype(np.float32)
Xa[:, rng.choice(C, 6, replace=False)] *= 60                  # a few massive channels, as in OPT/LLaMA activations
Wq = (rng.normal(size=(C, C)) / np.sqrt(C)).astype(np.float32)
ref = Xa @ Wq


def q8(x, axis):
    s = np.abs(x).max(axis=axis, keepdims=True) / 127
    return np.round(x / s).clip(-127, 127) * s


def rel_err(out):
    return np.linalg.norm(out - ref) / np.linalg.norm(ref)


print(f"  per-tensor int8 activations:                {rel_err(q8(Xa, None) @ Wq):.1%} relative error")
print(f"  per-token (per-row) int8 activations:       {rel_err(q8(Xa, 1) @ Wq):.1%}  (every row still contains the outlier channels)")
s = np.sqrt(np.abs(Xa).max(0) / np.abs(Wq).max(1))           # SmoothQuant alpha=0.5: X/s and s*W, product unchanged
print(f"  SmoothQuant (X/s, s*W), per-token int8 both:  {rel_err(q8(Xa / s, 1) @ q8(Wq * s[:, None], 0)):.1%}")
print(f"  int8 weights only (what GPTQ/AWQ-style methods quantize): {rel_err(Xa @ q8(Wq, 0)):.1%}")
print("  -> LLM activations have a handful of channels 20-100x larger than the rest; naive int8 maps everything else to\n"
      "     a few levels. That is why weight-only int4/int8 is the default for decode (memory-bound anyway) and why\n"
      "     activation quantization needs per-channel smoothing or mixed precision for the outliers (LLM.int8()).")

# ============================================================ 6. A/B testing
section("6. A/B testing: peeking, clustered users, and CUPED")
n_tests, days, per_day, p0 = 2000, 14, 2000, 0.10
a = rng.binomial(per_day, p0, (n_tests, days)).cumsum(1)       # A/A test: both arms have the same CTR
b = rng.binomial(per_day, p0, (n_tests, days)).cumsum(1)
nn_ = per_day * np.arange(1, days + 1)
pa, pb = a / nn_, b / nn_
se = np.sqrt(pa * (1 - pa) / nn_ + pb * (1 - pb) / nn_)
z = np.abs(pa - pb) / se
print(f"  A/A tests, 14 daily looks at p<0.05: false positive rate if you stop at the first significant day = "
      f"{(z > 1.96).any(1).mean():.1%}; at the planned end only = {(z[:, -1] > 1.96).mean():.1%}")

n_users = 20_000
user_ctr = rng.beta(2, 18, n_users)                            # users differ a lot in base CTR (mean 0.1)
imps = rng.poisson(10, n_users) + 1
fp_naive = fp_user = 0
for _ in range(300):
    arm = rng.uniform(size=n_users) < 0.5                       # randomize by USER
    clicks = rng.binomial(imps, user_ctr)
    ctr = [clicks[arm == g].sum() / imps[arm == g].sum() for g in (0, 1)]
    se_naive = np.sqrt(sum(c * (1 - c) / imps[arm == g].sum() for g, c in enumerate(ctr)))   # treats impressions as iid
    se_user = 0.0
    for g in (0, 1):                                            # delta method for a ratio of user-level sums
        x, nI = clicks[arm == g], imps[arm == g]
        mx, mn, k = x.mean(), nI.mean(), len(x)
        var = (x.var() / mn ** 2 - 2 * mx * np.cov(x, nI)[0, 1] / mn ** 3 + mx ** 2 * nI.var() / mn ** 4) / k
        se_user += var
    se_user = np.sqrt(se_user)
    fp_naive += abs(ctr[0] - ctr[1]) / se_naive > 1.96
    fp_user += abs(ctr[0] - ctr[1]) / se_user > 1.96
print(f"  users randomized, impressions analysed as if independent: false positive rate {fp_naive / 300:.1%}; "
      f"with the delta method at user level: {fp_user / 300:.1%}")

pre = rng.normal(size=50_000)                                   # pre-experiment metric per user
post = 0.8 * pre + 0.6 * rng.normal(size=50_000)                # experiment-period metric, correlated with pre
theta_c = np.cov(post, pre)[0, 1] / pre.var()
adj = post - theta_c * (pre - pre.mean())                       # CUPED
print(f"  CUPED: corr(pre, post) = {np.corrcoef(pre, post)[0, 1]:.2f} -> variance reduced by {1 - adj.var() / post.var():.0%}, "
      f"i.e. the same power as {post.var() / adj.var():.1f}x more users")
print("  -> three things that silently inflate 'wins': stopping on the first significant day, computing variance at the\n"
      "     wrong unit (impression vs user), and not using the pre-period. The MDE you can afford is set by variance,\n"
      "     and variance is something you engineer.")
print("\nDone.")
