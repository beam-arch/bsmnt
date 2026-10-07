"""Two-tower retrieval model with in-batch negatives (sampled softmax), the logQ
popularity correction, full-ranking evaluation, and how retrieval is served.

Run:  python code/06_two_tower_recsys.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

rng = np.random.default_rng(0)
torch.manual_seed(0)

# ------------------------------------------------------------ synthetic world
n_users, n_items, k_true = 3000, 1500, 12
n_country, n_age, n_cat = 10, 6, 25
user_country = rng.integers(0, n_country, n_users)
user_age = rng.integers(0, n_age, n_users)
item_cat = rng.integers(0, n_cat, n_items)

# true taste = side-feature effects + individual noise (so features help, IDs still matter)
country_vec, age_vec, cat_vec = (rng.normal(size=(n, k_true)) for n in (n_country, n_age, n_cat))
U_true = country_vec[user_country] + age_vec[user_age] + 0.8 * rng.normal(size=(n_users, k_true))
I_true = cat_vec[item_cat] + 0.8 * rng.normal(size=(n_items, k_true))
pop_bias = rng.lognormal(0, 0.8, n_items)          # heavy-tailed popularity: a few blockbusters

train_u, train_i, test = [], [], {}
for u in range(n_users):
    logits = 0.6 * (U_true[u] @ I_true.T) + pop_bias
    p = np.exp(logits - logits.max()); p /= p.sum()
    items = rng.choice(n_items, size=12, replace=False, p=p)
    train_u += [u] * 11; train_i += items[:-1].tolist()
    test[u] = int(items[-1])                         # last consumed item is held out
train_u, train_i = np.array(train_u), np.array(train_i)
train_sets = {u: set() for u in range(n_users)}
for u, i in zip(train_u, train_i):
    train_sets[u].add(int(i))

item_count = np.bincount(train_i, minlength=n_items).astype(float)
print(f"{len(train_u)} train interactions; top-1% items take {np.sort(item_count)[::-1][:n_items // 100].sum() / item_count.sum():.0%} of them")
head = set(np.argsort(-item_count)[: n_items // 10].tolist())   # top-10% most popular = "head"


# ------------------------------------------------------------ towers
class Tower(nn.Module):
    """Any features -> one embedding. The two towers never see each other's inputs."""

    def __init__(self, cardinalities, dim=32):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, dim) for c in cardinalities])
        self.mlp = nn.Sequential(nn.Linear(dim * len(cardinalities), 64), nn.ReLU(), nn.Linear(64, dim))

    def forward(self, *feats):
        x = torch.cat([e(f) for e, f in zip(self.embs, feats)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)          # unit length -> dot product = cosine


class TwoTower(nn.Module):
    def __init__(self, dim=32, temperature=0.05):
        super().__init__()
        self.user = Tower([n_users, n_country, n_age], dim)
        self.item = Tower([n_items, n_cat], dim)
        self.t = temperature

    def loss(self, u_feats, i_feats, item_ids, log_q=None):
        ue, ie = self.user(*u_feats), self.item(*i_feats)         # (B, d) each
        logits = ue @ ie.T / self.t                                # (B, B): every other row's item is a negative
        if log_q is not None:
            logits = logits - log_q[item_ids][None, :]             # logQ correction: don't over-penalize popular items
        same = item_ids[None, :] == item_ids[:, None]              # same item appearing twice in the batch
        same.fill_diagonal_(False)
        logits = logits.masked_fill(same, float("-inf"))           # ... is not a negative; mask it
        return F.cross_entropy(logits, torch.arange(len(ue)))      # softmax over the batch, target = diagonal


U = (torch.tensor(train_u), torch.tensor(user_country[train_u]), torch.tensor(user_age[train_u]))
I = (torch.tensor(train_i), torch.tensor(item_cat[train_i]))
log_q_all = torch.log(torch.tensor(item_count / item_count.sum() + 1e-12, dtype=torch.float32))


def train_model(use_logq, epochs=20, batch=512):
    model = TwoTower()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(train_u))
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            loss = model.loss([f[idx] for f in U], [f[idx] for f in I], I[0][idx], log_q_all if use_logq else None)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, K=50, name=""):
    model.eval()
    # SERVING PATTERN: item embeddings are computed once and stored in an index...
    item_index = model.item(torch.arange(n_items), torch.tensor(item_cat))            # (n_items, d)
    # ...user embeddings are computed per request, then a (approximate) nearest-neighbour search.
    user_emb = model.user(torch.arange(n_users), torch.tensor(user_country), torch.tensor(user_age))
    scores = user_emb @ item_index.T                                                   # exact search here; FAISS/HNSW in prod
    for u in range(n_users):
        scores[u, list(train_sets[u])] = -float("inf")
    topk = scores.topk(K, dim=1).indices.numpy()
    test_items = np.array([test[u] for u in range(n_users)])
    hit = (topk == test_items[:, None]).any(1)
    tail_mask = np.array([test[u] not in head for u in range(n_users)])
    coverage = len(set(topk[:, :10].ravel().tolist())) / n_items
    print(f"{name:<22} Recall@{K} {hit.mean():.3f}   tail Recall@{K} {hit[tail_mask].mean():.3f}   "
          f"head Recall@{K} {hit[~tail_mask].mean():.3f}   top-10 catalogue coverage {coverage:.1%}")
    return hit.mean()


print("\n--- baseline ---")
pop_scores = torch.tensor(item_count)[None, :].repeat(n_users, 1)
# evaluate popularity with the same protocol
with torch.no_grad():
    s = pop_scores.clone()
    for u in range(n_users):
        s[u, list(train_sets[u])] = -float("inf")
    topk = s.topk(50, dim=1).indices.numpy()
    test_items = np.array([test[u] for u in range(n_users)])
    pop_recall = (topk == test_items[:, None]).any(1).mean()
    print(f"{'most-popular':<22} Recall@50 {pop_recall:.3f}")

print("\n--- two-tower, in-batch negatives ---")
m_plain = train_model(use_logq=False)
r_plain = evaluate(m_plain, name="no correction")
m_logq = train_model(use_logq=True)
r_logq = evaluate(m_logq, name="with logQ correction")
assert max(r_plain, r_logq) > pop_recall
print("Lesson: in-batch negatives are drawn in proportion to popularity, so without correction the model\n"
      "        under-ranks head items (low head recall, very high coverage). logQ restores head recall at the\n"
      "        cost of coverage. Production systems mix in uniform negatives and re-rank for diversity.")

print("\n--- serving one request ---")
with torch.no_grad():
    m_logq.eval()
    index = m_logq.item(torch.arange(n_items), torch.tensor(item_cat))        # precomputed nightly / on item change
    u = 7
    q = m_logq.user(torch.tensor([u]), torch.tensor([user_country[u]]), torch.tensor([user_age[u]]))   # per request
    s = (q @ index.T).squeeze(0)
    s[list(train_sets[u])] = -float("inf")
    cands = s.topk(10).indices.tolist()
print(f"user {u}: retrieved candidates {cands} -> these go to the ranking stage with cross features")
print("Note: the towers never see (user, item) jointly, which is why a ranker follows retrieval.")
print("Done.")
