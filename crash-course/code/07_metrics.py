"""Every interview metric implemented from scratch and checked against scikit-learn.

Classification: precision/recall/F1, ROC-AUC (rank statistic), PR-AUC (average
precision), log loss, calibration error.  Ranking: Precision@K, Recall@K, MRR, MAP@K, NDCG@K.

Run:  python code/07_metrics.py
"""
import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, log_loss, ndcg_score,
                             precision_score, recall_score, roc_auc_score)

rng = np.random.default_rng(0)

# ------------------------------------------------------------ imbalanced binary data
N = 5000
y = (rng.uniform(size=N) < 0.03).astype(int)                 # 3% positives, like CTR / fraud
scores = 1 / (1 + np.exp(-rng.normal(-2.0 + 2.5 * y, 1.0)))   # sigmoid scores; positives score higher on average


def confusion(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def precision_recall_f1(y, pred):
    tp, fp, fn, _ = confusion(y, pred)
    p = tp / (tp + fp) if tp + fp else 0.0        # of what I flagged, how much was right
    r = tp / (tp + fn) if tp + fn else 0.0        # of what was there, how much I caught
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def roc_auc(y, s):
    """AUC = P(score of random positive > score of random negative), ties count 1/2.
    Computed via ranks (Mann-Whitney U), O(n log n)."""
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(y, s):
    """Area under the precision-recall curve (step-wise): mean precision at each positive."""
    order = np.argsort(-s)
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    precision_at_k = cum_tp / np.arange(1, len(y) + 1)
    return (precision_at_k * y_sorted).sum() / y.sum()


def logloss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)                   # never log(0)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()


def expected_calibration_error(y, p, n_bins=10):
    """Average |mean predicted prob - observed rate| over probability bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return ece


print("=== classification metrics (3% positives) ===")
pred_05 = (scores > 0.5).astype(int)
p, r, f1 = precision_recall_f1(y, pred_05)
assert np.isclose(p, precision_score(y, pred_05)) and np.isclose(r, recall_score(y, pred_05)) and np.isclose(f1, f1_score(y, pred_05))
print(f"threshold 0.5: precision {p:.3f} recall {r:.3f} F1 {f1:.3f}   accuracy {(pred_05 == y).mean():.3f}  "
      f"(predict-all-negative accuracy: {(y == 0).mean():.3f})")

auc, ap = roc_auc(y, scores), average_precision(y, scores)
assert np.isclose(auc, roc_auc_score(y, scores)), (auc, roc_auc_score(y, scores))
assert np.isclose(ap, average_precision_score(y, scores)), (ap, average_precision_score(y, scores))
print(f"ROC-AUC {auc:.3f}  (looks great)    PR-AUC / AP {ap:.3f}  (tells the truth under imbalance)")

# threshold selection from costs: say a false negative costs 10x a false positive
best = max(np.linspace(0.05, 0.95, 91), key=lambda t: -(10 * confusion(y, (scores > t).astype(int))[2] + confusion(y, (scores > t).astype(int))[1]))
p, r, _ = precision_recall_f1(y, (scores > best).astype(int))
print(f"cost-optimal threshold (FN = 10x FP): {best:.2f} -> precision {p:.3f} recall {r:.3f}")

ll = logloss(y, scores)
assert np.isclose(ll, log_loss(y, scores))
print(f"log loss {ll:.3f}   ECE {expected_calibration_error(y, scores):.3f}  (scores here are not calibrated probabilities)")

# calibration vs ranking: a monotone transform keeps AUC identical but changes log loss
sharpened = scores ** 3
print(f"after sharpening scores (monotone): AUC {roc_auc(y, sharpened):.3f} (same)   log loss {logloss(y, sharpened):.3f} (different)")


# ------------------------------------------------------------ ranking metrics
def precision_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / k


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr(ranked, relevant):
    for i, item in enumerate(ranked):
        if item in relevant:
            return 1 / (i + 1)
    return 0.0


def average_precision_at_k(ranked, relevant, k):
    hits, s = 0, 0.0
    for i, item in enumerate(ranked[:k]):
        if item in relevant:
            hits += 1
            s += hits / (i + 1)
    return s / min(len(relevant), k)


def dcg(gains):
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))   # rank 1 -> log2(2) = 1


def ndcg_at_k(ranked, rel_of, k):
    gains = [rel_of.get(item, 0) for item in ranked[:k]]
    ideal = sorted(rel_of.values(), reverse=True)[:k]
    return dcg(gains) / dcg(ideal) if dcg(ideal) > 0 else 0.0


print("\n=== ranking metrics for one user ===")
ranked = ["i5", "i2", "i9", "i1", "i7", "i3"]            # what we showed, in order
rel_of = {"i2": 3, "i7": 1, "i4": 2}                      # graded relevance (e.g. 3 = purchase, 1 = click)
relevant = set(rel_of)
print(f"P@3 {precision_at_k(ranked, relevant, 3):.3f}  R@3 {recall_at_k(ranked, relevant, 3):.3f}  "
      f"MRR {mrr(ranked, relevant):.3f}  MAP@5 {average_precision_at_k(ranked, relevant, 5):.3f}  "
      f"NDCG@5 {ndcg_at_k(ranked, rel_of, 5):.3f}")

# cross-check NDCG with sklearn (expects score/relevance matrices over a fixed item set)
items = ["i1", "i2", "i3", "i4", "i5", "i7", "i9"]
true_rel = np.array([[rel_of.get(i, 0) for i in items]], float)
pred_score = np.array([[len(ranked) - ranked.index(i) if i in ranked else 0 for i in items]], float)
assert np.isclose(ndcg_at_k(ranked, rel_of, 5), ndcg_score(true_rel, pred_score, k=5))

# NDCG rewards putting relevant items higher: swap i2 to the top
better = ["i2", "i5", "i9", "i1", "i7", "i3"]
print(f"moving the best item to rank 1: NDCG@5 {ndcg_at_k(ranked, rel_of, 5):.3f} -> {ndcg_at_k(better, rel_of, 5):.3f}  "
      f"(P@3 unchanged at {precision_at_k(better, relevant, 3):.3f})")
print("\nAll metric checks passed.")
