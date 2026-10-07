"""Matrix factorization for implicit feedback (BPR), evaluated the right way.

Shows: a per-user temporal hold-out, full-ranking Recall@K / NDCG@K, the popularity
baseline you must beat, and how "sampled negative" evaluation inflates the numbers.

Run:  python code/05_matrix_factorization_recsys.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)
torch.manual_seed(0)

# ------------------------------------------------------------ synthetic world
n_users, n_items, k_true = 600, 400, 8
U_true = rng.normal(size=(n_users, k_true))
I_true = rng.normal(size=(n_items, k_true))
pop_bias = rng.normal(0, 1.5, n_items)            # some items are popular with everyone (long tail)

interactions = []                                 # (user, item, t): t = order of consumption
for u in range(n_users):
    logits = U_true[u] @ I_true.T + pop_bias
    p = np.exp(logits - logits.max()); p /= p.sum()
    n_u = rng.integers(15, 40)
    items = rng.choice(n_items, size=n_u, replace=False, p=p)   # implicit positives only
    interactions += [(u, int(i), t) for t, i in enumerate(items)]
print(f"{len(interactions)} interactions, {n_users} users, {n_items} items, "
      f"density {len(interactions) / (n_users * n_items):.3%}")

# ---------------------------------------------------- split: last item per user is test
# A random split would leak the future (the model would train on what the user does later).
train, test = [], {}
for u in range(n_users):
    hist = sorted([x for x in interactions if x[0] == u], key=lambda x: x[2])
    train += hist[:-1]
    test[u] = hist[-1][1]
train_u = np.array([x[0] for x in train]); train_i = np.array([x[1] for x in train])
train_sets = {u: set() for u in range(n_users)}
for u, i in zip(train_u, train_i):
    train_sets[u].add(i)


# ------------------------------------------------------------ evaluation
def evaluate(score_fn, K=10, name=""):
    """score_fn(u) -> scores over all items. Full ranking, train items masked out."""
    hits, ndcg = 0.0, 0.0
    for u in range(n_users):
        s = score_fn(u).copy()
        s[list(train_sets[u])] = -np.inf                       # don't recommend what they already consumed
        rank = int((s > s[test[u]]).sum())                     # 0-based rank of the held-out item
        if rank < K:
            hits += 1
            ndcg += 1 / np.log2(rank + 2)                      # DCG with one relevant item; IDCG = 1
    print(f"{name:<28} Recall@{K} {hits / n_users:.3f}   NDCG@{K} {ndcg / n_users:.3f}")
    return hits / n_users


def evaluate_sampled(score_fn, K=10, n_neg=100, name=""):
    """The flawed protocol: rank the positive against 100 random negatives only."""
    hits = 0
    for u in range(n_users):
        s = score_fn(u)
        negs = rng.choice([i for i in range(n_items) if i not in train_sets[u] and i != test[u]], n_neg, replace=False)
        rank = int((s[negs] > s[test[u]]).sum())
        hits += rank < K
    print(f"{name:<28} Hit@{K} among {n_neg} sampled negatives: {hits / n_users:.3f}   <-- inflated")


print("\n--- baselines (full ranking) ---")
item_pop = np.bincount(train_i, minlength=n_items).astype(float)
evaluate(lambda u: rng.normal(size=n_items), name="random")
pop_recall = evaluate(lambda u: item_pop, name="most-popular")


# ------------------------------------------------------------ BPR matrix factorization
class MF(nn.Module):
    def __init__(self, n_users, n_items, dim=16):
        super().__init__()
        self.P = nn.Embedding(n_users, dim)        # user factors
        self.Q = nn.Embedding(n_items, dim)        # item factors
        self.b = nn.Embedding(n_items, 1)          # item bias: captures popularity explicitly
        for e in (self.P, self.Q):
            nn.init.normal_(e.weight, std=0.1)
        nn.init.zeros_(self.b.weight)

    def score(self, u, i):
        return (self.P(u) * self.Q(i)).sum(-1) + self.b(i).squeeze(-1)

    def forward(self, u, i, j):
        # BPR: observed item i should outscore sampled unobserved item j for user u
        return -F.logsigmoid(self.score(u, i) - self.score(u, j)).mean()


model = MF(n_users, n_items)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
u_t, i_t = torch.tensor(train_u), torch.tensor(train_i)
batch = 1024
for epoch in range(25):
    perm = torch.randperm(len(u_t))
    total = 0.0
    for s in range(0, len(u_t), batch):
        idx = perm[s:s + batch]
        j = torch.randint(0, n_items, (len(idx),))   # uniform negative sampling (a modeling decision!)
        loss = model(u_t[idx], i_t[idx], j)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    if epoch % 8 == 0 or epoch == 24:
        print(f"epoch {epoch:2d}  BPR loss {total / len(u_t):.3f}")


@torch.no_grad()
def mf_scores(u):
    model.eval()
    all_items = torch.arange(n_items)
    return model.score(torch.full((n_items,), u), all_items).numpy()


print("\n--- model (full ranking) ---")
mf_recall = evaluate(mf_scores, name="BPR-MF (dim 16)")
assert mf_recall > pop_recall, "a recommender that doesn't beat popularity is broken"

print("\n--- the sampled-negatives trap ---")
evaluate_sampled(lambda u: item_pop, name="most-popular")
evaluate_sampled(mf_scores, name="BPR-MF (dim 16)")

# what the embeddings learned: nearest neighbours in item space ("users who liked X also liked...")
Q = model.Q.weight.detach().numpy()
Qn = Q / np.linalg.norm(Q, axis=1, keepdims=True)
In = I_true / np.linalg.norm(I_true, axis=1, keepdims=True)
true_sim = In @ In.T
learned_sim = Qn @ Qn.T
i0 = 0
print(f"\nitem 0: top-5 neighbours by learned factors {np.argsort(-learned_sim[i0])[1:6].tolist()}"
      f"  by true factors {np.argsort(-true_sim[i0])[1:6].tolist()}")
print("Done.")
