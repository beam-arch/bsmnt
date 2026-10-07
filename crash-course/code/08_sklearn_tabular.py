"""A correct end-to-end tabular pipeline with scikit-learn, and a leakage demonstration.

Shows: baselines first, Pipeline (scaler fit inside each fold), stratified CV, GBDT vs
logistic regression, PR-AUC vs ROC-AUC under imbalance, a leaky feature that makes the
metrics lie, calibration, and permutation importance.

Run:  python code/08_sklearn_tabular.py
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

# Real-world gotcha: on a many-core box, OpenMP threads in HistGradientBoosting oversubscribe
# a tiny dataset and make this script >10x slower (55s vs 4s on 16 cores).
threadpool_limits(limits=4, user_api="openmp")

rng = np.random.default_rng(0)

X, y = make_classification(n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
                           weights=[0.95, 0.05], flip_y=0.02, class_sep=0.8, random_state=0)
print(f"rows {len(y)}, positive rate {y.mean():.1%}")

# A held-out test set that nothing below touches until the very end.
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


def cv_report(name, model):
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["roc_auc", "average_precision"])
    auc, ap = res["test_roc_auc"], res["test_average_precision"]
    print(f"{name:<34} ROC-AUC {auc.mean():.3f} ± {auc.std():.3f}   PR-AUC {ap.mean():.3f} ± {ap.std():.3f}")
    return ap.mean()


print("\n=== baselines first ===")
cv_report("majority class (dummy)", DummyClassifier(strategy="prior"))
# The scaler lives INSIDE the pipeline so it is fit on each training fold only (no preprocessing leakage).
lr_ap = cv_report("logistic regression (scaled)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
gb_ap = cv_report("gradient boosting (no scaling needed)", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0))

print("\n=== leakage: add a feature that is a consequence of the label ===")
leak = y_tr + 0.3 * rng.normal(size=len(y_tr))          # e.g. 'refund_issued' when predicting fraud
X_leaky = np.c_[X_tr, leak]
cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = cross_validate(HistGradientBoostingClassifier(random_state=0), X_leaky, y_tr, cv=cv, scoring="roc_auc")
print(f"with the leaky feature: ROC-AUC {res['test_score'].mean():.3f}  <-- 'too good to be true' is the alarm")

print("\n=== fit the final model, evaluate once on the test set ===")
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(X_tr, y_tr)
p_te = gb.predict_proba(X_te)[:, 1]
print(f"test ROC-AUC {roc_auc_score(y_te, p_te):.3f}   PR-AUC {average_precision_score(y_te, p_te):.3f}   "
      f"Brier {brier_score_loss(y_te, p_te):.4f}")

print("\n=== calibration: does 0.3 mean 30%? ===")
for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.01)]:
    m = (p_te >= lo) & (p_te < hi)
    if m.sum():
        print(f"  predicted in [{lo:.1f},{hi:.1f}): mean pred {p_te[m].mean():.3f}  observed {y_te[m].mean():.3f}  (n={m.sum()})")
cal = CalibratedClassifierCV(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0), method="isotonic", cv=5).fit(X_tr, y_tr)
p_cal = cal.predict_proba(X_te)[:, 1]
print(f"isotonic recalibration: Brier {brier_score_loss(y_te, p_te):.4f} -> {brier_score_loss(y_te, p_cal):.4f}  "
      f"(ROC-AUC {roc_auc_score(y_te, p_cal):.3f}, ranking essentially unchanged)")

print("\n=== threshold from business costs (FN costs 20x FP) ===")
costs = {t: 20 * ((p_cal < t) & (y_te == 1)).sum() + ((p_cal >= t) & (y_te == 0)).sum() for t in np.linspace(0.02, 0.9, 45)}
t_best = min(costs, key=costs.get)
pred = p_cal >= t_best
tp = (pred & (y_te == 1)).sum(); fp = (pred & (y_te == 0)).sum(); fn = (~pred & (y_te == 1)).sum()
print(f"best threshold {t_best:.2f}: precision {tp / (tp + fp):.3f} recall {tp / (tp + fn):.3f}  (not 0.5!)")

print("\n=== which features matter? permutation importance on held-out data ===")
imp = permutation_importance(gb, X_te, y_te, scoring="average_precision", n_repeats=3, random_state=0)
top = np.argsort(-imp.importances_mean)[:5]
print("top features:", ", ".join(f"x{i} ({imp.importances_mean[i]:.3f})" for i in top))
print("\nDone.")
