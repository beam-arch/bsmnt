"""Logged-data experiments: what goes wrong when your training data was produced by the
system you are trying to improve. These are the mechanisms behind most 'offline metric
up, online flat' stories in recommendation, search, ads, and fraud.

Pure numpy + sklearn, ~15 seconds.

Run:  python code/11_experiments_logged_data.py
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4, user_api="openmp")
rng = np.random.default_rng(0)


def section(t):
    print(f"\n=== {t} ===")


# ============================================================ 1. position bias
section("1. position bias is confounded with relevance in a good ranker's logs")
n_items, K = 50, 10
p_true = rng.beta(2, 8, n_items)                      # P(click | examined) per item
theta = 1 / np.arange(1, K + 1) ** 0.7                # P(examined | position), position-based click model
old_score = p_true + 0.06 * rng.normal(size=n_items)  # the incumbent ranker's (noisy) belief


def log_sessions(n, jitter, explore_frac):
    """Incumbent shows its top-K; `jitter` = per-session score noise, `explore_frac` = sessions with a random slate."""
    scores = old_score[None, :] + jitter * rng.normal(size=(n, n_items))
    slates = np.argsort(-scores, axis=1)[:, :K]
    n_exp = int(explore_frac * n)
    if n_exp:
        slates[:n_exp] = np.argsort(rng.uniform(size=(n_exp, n_items)), axis=1)[:, :K]
    examined = rng.uniform(size=(n, K)) < theta[None, :]
    clicks = examined & (rng.uniform(size=(n, K)) < p_true[slates])
    return slates, clicks


def ctr_by_position(slates, clicks):
    return clicks.mean(0)


def pbm_fit(slates, clicks, iters=200):
    """Position-based model fit by alternating updates: click = theta[pos] * p[item]. (This is what
    'position as a feature' learns, in its cleanest form.)"""
    th, p = np.ones(K), np.full(n_items, 0.1)
    pos = np.broadcast_to(np.arange(K), slates.shape)
    for _ in range(iters):
        p = np.bincount(slates.ravel(), clicks.ravel(), n_items) / np.maximum(np.bincount(slates.ravel(), np.broadcast_to(th, slates.shape).ravel(), n_items), 1e-9)
        th = np.bincount(pos.ravel(), clicks.ravel(), K) / np.maximum(np.bincount(pos.ravel(), p[slates].ravel(), K), 1e-9)
        p, th = p * th[0], th / th[0]                # identifiability: fix theta[0] = 1
    return th, p


print(f"  true examination probability by position:        {np.round(theta, 2)}")
for jitter, explore, label in ((0.0, 0.0, "deterministic ranker, no exploration"),
                               (0.03, 0.0, "slightly noisy ranker, no exploration"),
                               (0.03, 0.05, "noisy ranker + 5% random slates")):
    slates, clicks = log_sessions(100_000, jitter, explore)
    naive = ctr_by_position(slates, clicks)
    th_hat, p_hat = pbm_fit(slates, clicks)
    corr = np.corrcoef(p_hat, p_true)[0, 1]
    never_shown = n_items - len(np.unique(slates))
    print(f"  [{label}]")
    print(f"    naive CTR(pos)/CTR(pos 1):                       {np.round(naive / naive[0], 2)}  <- too steep: good items sit on top")
    print(f"    PBM (item effect + position effect) estimate:    {np.round(th_hat, 2)}   corr(p_hat, p_true) = {corr:.2f}, "
          f"items never shown: {never_shown}/{n_items}")
print("  -> a deterministic ranker makes position and relevance perfectly confounded: no model can separate them from its\n"
      "     logs. The joint item+position model is identified only by whatever variation the logging policy happened to\n"
      "     create (noise, model versions = 'intervention harvesting'): the noisy ranker recovers theta, yet most items\n"
      "     still have no estimate at all because they were never shown. A 5% random slice fixes both problems.\n"
      "     'Add position as a feature' works exactly as well as that variation allows -- no better.")

# ============================================================ 2. incumbency bias
section("2. offline evaluation on logged clicks favours the incumbent; IPS with exploration fixes the ordering")
slates, clicks = log_sessions(200_000, 0.03, 0.10)       # 10% exploration so every (item, position) has support
# candidate rankers (deterministic):
cand = {
    "A: incumbent + small noise": np.argsort(-(old_score + 0.02 * rng.normal(size=n_items))),
    "B: true relevance order":    np.argsort(-p_true),
    "C: rank by logged item CTR": np.argsort(-(np.bincount(slates.ravel(), clicks.ravel(), n_items) / np.maximum(np.bincount(slates.ravel(), minlength=n_items), 1))),
}
mu = np.zeros((n_items, K))                               # logging propensity of (item at position), estimated from logs
for k in range(K):
    mu[:, k] = np.bincount(slates[:, k], minlength=n_items) / len(slates)
print(f"  {'candidate':<30} {'online truth':>12} {'naive Recall@10':>16} {'IPS estimate':>13}")
for name, order in cand.items():
    online = (theta * p_true[order[:K]]).sum()            # expected clicks per session if deployed
    topk = set(order[:K].tolist())
    naive = clicks[np.isin(slates, list(topk))].sum() / clicks.sum()   # share of logged clicks in the candidate's top-10
    # position-based IPS: credit a logged click only if the candidate puts that item at that position; reweight by 1/propensity
    cand_pos = np.full(n_items, -1); cand_pos[order[:K]] = np.arange(K)
    match = cand_pos[slates] == np.arange(K)[None, :]
    w = np.where(match, 1 / np.maximum(mu[slates, np.arange(K)[None, :]], 1e-6), 0.0)
    ips = (clicks * np.minimum(w, 200)).sum(1).mean()      # clipped weights (bias-variance trade)
    print(f"  {name:<30} {online:12.3f} {naive:16.3f} {ips:13.3f}")
print("  -> naive offline metric: A (a copy of the incumbent) > C > B, because logged clicks only exist on what the\n"
      "     incumbent showed. Online truth and the IPS estimate: B > A. Without exploration traffic the IPS weights are\n"
      "     undefined for items the incumbent never showed, and no estimator can rescue you. Candidate C is the\n"
      "     feedback loop in one line: train on logged CTR and you reproduce the incumbent.")

# ============================================================ 3. negative downsampling
section("3. negative downsampling shifts the prior; recalibrate with odds * w (He et al. 2014)")
n = 300_000
Xc = rng.normal(size=(n, 6))
logit = Xc @ np.array([1.2, -0.8, 0.5, 0.3, 0.0, 0.0]) - 4.3
yc = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))     # ~3.6% positives
tr, te = np.arange(n) < 200_000, np.arange(n) >= 200_000
w = 0.1                                                  # keep 10% of negatives
keep = tr & (yc | (rng.uniform(size=n) < w))
full = LogisticRegression(max_iter=500).fit(Xc[tr], yc[tr])
down = LogisticRegression(max_iter=500).fit(Xc[keep], yc[keep])
p_full, p_down = full.predict_proba(Xc[te])[:, 1], down.predict_proba(Xc[te])[:, 1]
p_fixed = p_down / (p_down + (1 - p_down) / w)           # odds_true = odds_sampled * w
print(f"  training rows: all={tr.sum():,}  downsampled={keep.sum():,} ({keep.sum() / tr.sum():.0%})   positive rate in test {yc[te].mean():.3%}")
for name, p in (("full data", p_full), ("downsampled, raw", p_down), ("downsampled + correction", p_fixed)):
    print(f"  {name:<26} mean predicted {p.mean():.3%}   AUC {roc_auc_score(yc[te], p):.4f}")
print("  -> ranking (AUC) is untouched by downsampling; probabilities are inflated ~1/w. The fix is one line on the\n"
      "     logit (+log w), and it is exactly the Bayes prior-shift correction you would also apply when the base rate\n"
      "     changes between training and serving (seasonality, a new country, a new surface).")

# ============================================================ 4. delayed feedback
section("4. delayed feedback: recent clicks look like non-converters (Chapelle 2014)")
n = 400_000
age = rng.uniform(0, 30, n)                              # days since the click, as of 'now'
converts = rng.uniform(size=n) < 0.05                    # true CVR 5%
delay = rng.exponential(4.0, n)                          # click -> conversion delay, mean 4 days
observed = converts & (delay < age)                      # the label you can see today
print(f"  true CVR 5.00%   naive label CVR {observed.mean():.2%}   (window = 30 days of clicks)")
for lo, hi in ((0, 1), (1, 3), (3, 7), (7, 14), (14, 30)):
    m = (age >= lo) & (age < hi)
    print(f"    clicks {lo:2d}-{hi:2d} days old: observed CVR {observed[m].mean():.2%}")
mature = age > 20
F_delay = lambda t: 1 - np.exp(-t / 4.0)                 # P(delay < t); estimable from matured data
ipw = observed / F_delay(age)                            # inverse-probability weighting of observed positives
print(f"  fix 1, only matured clicks (>20 days): CVR {observed[mature].mean():.2%} but uses {mature.mean():.0%} of data and is 20+ days stale")
print(f"  fix 2, weight positives by 1/P(converted by now | converts): CVR {ipw.mean():.2%} using all data")
print("  -> a model with any recency-correlated feature (campaign age, new item, new user) learns 'new = low CVR'.\n"
      "     Production choices: attribution window + wait; importance weights (as here); or treat unconverted-as-yet as\n"
      "     negatives and re-insert them as positives when they convert (fake-negative weighting).")

# ============================================================ 5. selective labels
section("5. selective labels: you only have labels where the old model decided to look")
n = 200_000
Xf = rng.normal(size=(n, 4))
# two fraud patterns: the 'known' one (x0 high) and a 'new' one (x0 very low) the old model never flagged
fraud_p = 1 / (1 + np.exp(-(3 * (Xf[:, 0] - 1.5)))) * 0.6 + 0.5 * (Xf[:, 0] < -1.5) + 0.01
fraud = rng.uniform(size=n) < fraud_p
old_flag = (Xf[:, 0] + 0.3 * rng.normal(size=n)) > 1.0   # old rule: review only high-x0 cases -> labels only there
random_review = rng.uniform(size=n) < 0.02              # a 2% random audit stream
tr, te = np.arange(n) < 150_000, np.arange(n) >= 150_000
new_region = Xf[:, 0] < -1.5
gb = lambda: HistGradientBoostingClassifier(max_iter=100, random_state=0)
m_sel = gb().fit(Xf[tr & old_flag], fraud[tr & old_flag])
m_mix = gb().fit(Xf[tr & (old_flag | random_review)], fraud[tr & (old_flag | random_review)])
fraud_te, new_te = fraud[te], new_region[te]
for name, m in (("trained on reviewed-only labels", m_sel), ("+ 2% random audit labels", m_mix)):
    p = m.predict_proba(Xf[te])[:, 1]
    flag = p > 0.3
    rec_new = (flag & fraud_te & new_te).sum() / (fraud_te & new_te).sum()
    print(f"  {name:<34} AUC on all traffic {roc_auc_score(fraud_te, p):.3f}   recall on the NEW fraud pattern {rec_new:.1%}")
print(f"  (the new pattern is {(fraud_te & new_te).sum() / fraud_te.sum():.0%} of all fraud and is invisible in reviewed-only labels)")
print("  -> labels are produced by a policy. Fraud review queues, loan approvals, content moderation, and 'items shown'\n"
      "     all have this structure. A small randomized audit stream is worth more than 10x the biased labels.")
print("\nDone.")
