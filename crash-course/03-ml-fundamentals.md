# 03 · ML fundamentals (90 minutes) — the module most interview questions come from

Run `code/07_metrics.py` (implements every metric below from scratch) and
`code/08_sklearn_tabular.py` (a correct end-to-end tabular pipeline) with this module.

## 1. What ML is, and the kinds of it

**Learn a function `f` from data by minimizing a loss on examples**, so that it works on
*new* examples (generalization).

- **Supervised**: inputs `x` with labels `y`. *Regression* (continuous `y`: price, watch
  time) or *classification* (discrete `y`: click/no-click, which category).
- **Unsupervised**: no labels. Clustering (k-means), dimensionality reduction (PCA),
  density estimation, anomaly detection.
- **Self-supervised**: labels manufactured from the data itself (predict the next token,
  predict a masked word, predict whether two augmented views are the same image). This is
  how LLMs and foundation models are pretrained.
- **Reinforcement learning**: an agent takes actions, gets rewards, learns a policy. In
  industry mostly as bandits (recsys exploration) and RLHF (LLM alignment).

## 2. The ML pipeline (and where the real work is)

```
problem framing → data collection → labeling → splitting → features → model → loss →
optimization → offline evaluation → deployment → online evaluation → monitoring → retrain
```

Hiring-manager truth: **problem framing and label definition are the hardest part**, and
the thing juniors skip. "Predict churn" — churn defined how? Over what window? Measured
when? Is the label available at prediction time? "Recommend videos" — optimize clicks
(clickbait), watch time (long videos), or satisfaction surveys (sparse)? Pick the wrong
label and the best model in the world optimizes the wrong thing.

## 3. Generalization and splits

- **Train / validation / test**: fit parameters on train, choose hyperparameters and make
  decisions on validation, report *once* on test. Every decision you make using a dataset
  "spends" some of its ability to estimate generalization; the test set must stay untouched.
- **Cross-validation (k-fold)**: rotate which fold is held out; average. Use when data is
  small. *Stratified* keeps class ratios. *Nested* CV if you also tune hyperparameters.
- **How to split is a modeling decision**:
  - **Random split** is only valid if examples are independent. Usually they aren't.
  - **Temporal split** (train on the past, validate on the future) whenever data has time:
    recsys, fraud, forecasting, ads. A random split lets the model "see the future" and
    gives wildly optimistic numbers. This is the #1 split mistake.
  - **Group split**: all rows from the same user / patient / device go to one side;
    otherwise the model memorizes entities.
- **Distribution shift**: test ≠ train distribution. *Covariate shift* (inputs change),
  *label shift* (class priors change), *concept drift* (the relationship changes — e.g.
  fraud patterns adapt). Your validation set should mimic *deployment* conditions.

## 4. Bias–variance, under/overfitting

- **Underfitting** (high bias): model too simple; train error high.
- **Overfitting** (high variance): fits noise; train error low, validation error high.
- **Diagnose with learning curves**: plot train and validation metric vs. training size or
  epochs. Gap = variance; both bad = bias.
- Fixes for overfitting: more data, regularization, early stopping, simpler model, data
  augmentation, ensembling, dropout, feature selection. Fixes for underfitting: bigger model,
  better features, train longer, lower regularization, higher learning rate.
- Modern nuance: **double descent** — very large neural nets can overfit *less* as they get
  bigger (interpolation regime). Don't over-apply the classical U-curve to deep learning.

## 5. Regularization

- **L2 / weight decay** shrinks weights toward zero (Gaussian prior); smooths the function.
- **L1** drives weights exactly to zero (sparsity, feature selection).
- **Early stopping**: stop when validation metric stops improving (implicit regularization).
- **Dropout**, **data augmentation**, **noise injection**, **label smoothing**, **ensembles**.
- Regularization strength is a hyperparameter; tune on validation.

## 6. Losses vs metrics (an interviewer favorite)

The **metric** is what the business cares about (accuracy, NDCG, revenue). The **loss** is a
differentiable surrogate you can optimize. Mismatch between them is a standing source of
problems ("log loss improved, NDCG didn't").

| Task | Loss | Notes |
|---|---|---|
| Regression | MSE | Penalizes big errors quadratically; sensitive to outliers; = Gaussian NLL |
| Regression | MAE / Huber | Robust to outliers; Huber = MSE near 0, MAE far away |
| Binary classification | Binary cross-entropy (log loss) | On *logits* via `BCEWithLogits` for stability |
| Multiclass | Cross-entropy | Softmax + NLL fused; expects raw logits |
| Imbalanced detection | Focal loss | Down-weights easy examples |
| Ranking (pairwise) | BPR / hinge | "Positive should score above negative" |
| Ranking (listwise) | Softmax over candidates / sampled softmax | Retrieval models, LLM next-token |
| Embeddings | Contrastive / InfoNCE / triplet | Pull positives together, push negatives apart |

## 7. Metrics — know them cold, with their failure modes

**Classification**
- **Accuracy**: fraction correct. Useless under imbalance (99% negatives → 99% accuracy by
  predicting "no").
- **Confusion matrix**: TP, FP, FN, TN.
- **Precision** = TP/(TP+FP): "of what I flagged, how much was right" (cost of false alarms:
  spam filter, content moderation).
- **Recall** = TP/(TP+FN): "of what was there, how much did I catch" (cost of misses: cancer
  screening, fraud).
- **F1** = harmonic mean of P and R; **Fβ** weights recall β times more.
- **Threshold**: P/R/F1 depend on the decision threshold; choose it from the business cost
  matrix on the validation set, not 0.5 by default.
- **ROC curve**: TPR vs FPR across thresholds; **ROC-AUC** = probability a random positive
  scores above a random negative (rank statistic; threshold-free; invariant to class ratio).
  *Failure mode*: under heavy imbalance, ROC-AUC looks great while precision is terrible,
  because FPR's denominator (all negatives) is huge.
- **PR curve / PR-AUC / average precision**: focuses on the positive class; use for
  imbalance (fraud, ads CTR ~1%).
- **Log loss**: penalizes confident mistakes; the only one of these that measures
  *calibration*. "AUC up, log loss down" means better ranking; "AUC same, log loss worse"
  means miscalibration.
- **Calibration**: reliability diagram, Expected Calibration Error; fix with Platt scaling
  (logistic fit), isotonic regression, or temperature scaling (NNs).

**Regression**: RMSE (same units as target; outlier-sensitive), MAE, R² (fraction of variance
explained; can be negative on a bad model), MAPE (breaks at zero targets).

**Ranking / retrieval / recsys** (per user, then averaged)
- **Precision@K / Recall@K / Hit rate@K**: of the top K, how many relevant / of all relevant
  items how many are in top K / was at least one.
- **MRR**: 1/rank of the first relevant item.
- **MAP**: mean of average precision over users.
- **NDCG@K**: discounted cumulative gain `Σ relᵢ / log₂(i+1)` normalized by the ideal
  ordering; rewards relevant items *higher* in the list; handles graded relevance.
- **Coverage / diversity / novelty**: non-accuracy metrics that matter in production.
- *Failure mode*: evaluating with *sampled* negatives (rank the positive against 100 random
  items) inflates and can even reorder model comparisons versus full ranking
  (Krichene & Rendle 2020). Say "I'd evaluate with full ranking, or be very careful about
  sampled metrics."

**Language**: perplexity (LM), BLEU/ROUGE (overlap; weak), exact match / F1 (QA), human eval
and LLM-as-judge (with known biases: length, position, self-preference).

## 8. Class imbalance

Options: **class weights** in the loss, **oversampling** the minority / **undersampling**
the majority, **SMOTE** (synthetic interpolation; often doesn't help strong models),
**focal loss**, **threshold moving**, and — most importantly — **use the right metric**
(PR-AUC, recall at fixed precision). Gotchas: resample *only the training set*, *after* the
split (resampling before splitting leaks duplicates into validation); re-weighting
destroys calibration, so recalibrate if you need probabilities (ads do).

## 9. Data leakage — the experience topic

Leakage = information available at training time that will not be available (or will not
be the same) at prediction time. It makes offline metrics lie.

- **Target leakage**: a feature that is a *consequence* of the label ("account_closed_date"
  when predicting churn, "number of fraud investigations" when predicting fraud).
- **Temporal leakage**: features computed with data from after the prediction time; random
  splits on time-series data; aggregates like "item total clicks" that include future
  clicks. The fix is **point-in-time correctness**: every feature is computed exactly as it
  would have been *at the moment of the event*.
- **Preprocessing leakage**: fitting the scaler / imputer / target encoder / vocabulary on
  the full dataset before splitting. Fit on train only; use `Pipeline`.
- **Train–test contamination**: duplicates or near-duplicates across splits (LLM benchmarks
  suffer from this; so do recsys datasets with repeated interactions).
- **Group leakage**: same user on both sides.
- **Detection heuristics**: a metric that is "too good" (AUC 0.99 on a hard problem); one
  feature with absurd importance; performance collapsing when you drop one feature;
  validation ≫ production performance. The experienced reflex: **"If it looks too good,
  suspect leakage before celebrating."**

## 10. Feature engineering (still most of applied ML)

- **Numerical**: standardize (`(x−μ)/σ`) or min-max for linear models / NNs / kNN / SVM;
  **trees don't need scaling**. Log-transform heavy-tailed counts. Clip outliers. Bin when
  the relationship is non-monotonic and the model is linear.
- **Categorical**: one-hot (low cardinality), ordinal (ordered), **target encoding** (mean
  label per category — must be out-of-fold + smoothed or it leaks), **hashing trick** (fixed
  size, handles unseen values, collisions are a tolerable noise), **learned embeddings**
  (NNs; the recsys default for IDs).
- **Missing values**: impute (median / mode / model) *plus a "was missing" indicator* —
  missingness is often informative. GBDTs handle NaNs natively.
- **Dates/time**: hour-of-day, day-of-week as cyclical (`sin/cos`), time-since-last-event,
  holidays.
- **Text**: bag-of-words / TF-IDF (strong baseline), pretrained embeddings.
- **Interactions / crosses**: `country × device`; linear models need them explicitly, trees
  and NNs learn them.
- **Aggregates / counts**: user's CTR over the last 7 days, item popularity — the bread and
  butter of recsys/ads features; must be computed point-in-time and with the same code at
  serving time (see module 09: train/serve skew).
- **Feature selection**: drop leaky/unavailable-at-serving features first, then correlated
  junk; use permutation importance rather than raw tree importance.

## 11. The classical algorithm roster

For each: the idea, when to use it, key hyperparameters, and the thing to say that shows depth.

**Linear regression** — `ŷ = wᵀx + b`, MSE loss; closed form `(XᵀX)⁻¹Xᵀy` or gradient descent.
Interpretable; needs scaling if regularized; multicollinearity makes coefficients unstable
(ridge fixes). Depth: "I'd start here as the baseline; if a NN can't beat it, the problem is
the data."

**Logistic regression** — `p = σ(wᵀx + b)`, log loss; convex (unique optimum); outputs are
log-odds, so coefficients are interpretable; regularization `C` (inverse strength); scales to
billions of rows (ads CTR models were LR with feature crosses for a decade). Depth: "with
good feature crosses it is a surprisingly strong CTR baseline, and it's calibrated out of
the box."

**k-nearest neighbors** — no training; predict by majority of the k closest points. Needs
scaling; suffers the curse of dimensionality (distances concentrate); O(N) per query unless
you use an ANN index — which is exactly what embedding retrieval does.

**Decision tree** — greedy recursive splits maximizing purity gain (Gini / entropy) or
variance reduction. Handles mixed types, no scaling, interpretable, captures interactions;
overfits unless depth/min-leaf is limited; axis-aligned (bad at diagonal boundaries).

**Random forest** — bagging (bootstrap samples) + random feature subsets per split; average
the trees. Reduces variance; robust defaults; out-of-bag error is a free validation
estimate. Feature importances are biased toward high-cardinality features.

**Gradient boosted trees (XGBoost / LightGBM / CatBoost)** — build trees *sequentially*,
each fitting the gradient of the loss w.r.t. current predictions (residuals for MSE); shrink
each tree by the learning rate. **The default winner on tabular data in industry.** Key
hyperparameters: number of trees (use early stopping on validation), learning rate (lower +
more trees = better), max depth / num leaves, min child weight, subsample & colsample
(regularization), L1/L2. Depth signals: LightGBM grows **leaf-wise** (best-first) and uses
**histogram binning** of features, which is why it is fast; CatBoost uses **ordered target
statistics** to encode categoricals without leakage; GBDTs handle missing values natively
and are insensitive to feature scaling; they extrapolate poorly outside the training range;
they are strong on tabular data because real tabular features are heterogeneous and
irregular, where NN inductive biases don't help.

**SVM** — maximize the margin; kernel trick for nonlinearity. Historically important; today
mostly replaced by GBDTs/NNs; needs scaling; doesn't scale past ~1e5 rows with kernels.

**Naive Bayes** — assumes feature independence given the class; absurd assumption, decent
text baseline, trains instantly.

**k-means** — alternate: assign points to nearest centroid; move centroids to the mean
(Lloyd's algorithm). Needs k (elbow / silhouette score), scaling, and good seeding
(**k-means++**); finds local minima (run several inits); assumes roughly spherical,
equal-size clusters. Used in production for IVF ANN indexes, user segmentation, codebooks.

**PCA** — rotate to the directions of maximum variance (eigenvectors of the covariance / SVD
of the centered data); keep top k. Standardize first; linear only; used for visualization,
decorrelation, compression. Nonlinear alternatives: t-SNE/UMAP (visualization only — don't
cluster on them naively), autoencoders.

**Also know exist**: DBSCAN (density clustering, no k, finds outliers), hierarchical
clustering, Gaussian mixture models (soft k-means, fit by EM), isolation forest (anomaly
detection), ARIMA/Prophet (classical forecasting; GBDTs with lag features usually win).

## 12. Hyperparameter tuning and experiment hygiene

- Random search beats grid search (Bergstra & Bengio); Bayesian optimization (Optuna) beats
  random for expensive models; use early stopping/pruning.
- **Always establish baselines first**: majority class, popularity, logistic regression,
  then a GBDT. A deep model's job is to beat them; if it can't, debug before scaling up.
- Change one thing at a time; fix seeds; log everything (config, data version, code commit,
  metrics); run multiple seeds for small differences — a 0.3% improvement with one seed is
  noise.
- **Ablations**: remove components to prove they matter. Managers love ablations because
  they separate "we added 5 things" from "which thing worked".

## 13. Interpretability (brief)

Coefficients (linear), feature importance (trees; biased), **permutation importance**
(model-agnostic; breaks with correlated features), **SHAP** (Shapley values; local
explanations; the industry standard for tabular), saliency/attention maps (NNs; attention
is not explanation). Why it matters: debugging leakage, regulatory needs (credit), trust.

## Experience signals

- "First thing I do with a new dataset: check label balance, duplicates, time range, and
  whether every feature is actually available at prediction time."
- "The sanity checks: overfit a tiny subset to 100%; check that the initial loss equals
  `ln(C)` for a C-class problem; shuffle the labels and confirm the model *can't* learn."
- "Random split on time-dependent data is leakage. I'd use a temporal split and a group
  split by user."
- "AUC is the wrong headline metric for a 0.5% positive rate; I'd report PR-AUC and recall
  at the precision the business can tolerate."
- "GBDT is my default on tabular; neural nets when there's unstructured data, very large
  data, or I need embeddings / multi-task / online learning."
- "Target encoding must be out-of-fold or it leaks."
- "Improvement of 0.2% on one seed is not an improvement."

## Follow-up chains

- "What's overfitting?" → "How would you detect it?" → "Validation loss is going *up* but
  validation accuracy is still going *up* — what's happening?" → *the model is becoming
  over-confident on the examples it gets wrong (log loss punishes confident errors) while
  the argmax is still improving; it's a calibration problem; early-stop on the metric you
  care about, or apply temperature scaling.*
- "What is AUC?" → "When is it misleading?" → "Offline AUC improved 2 points, online CTR
  dropped. Why?" → *candidates: offline eval on logged data biased by the old policy
  (position/exposure bias), calibration changed and downstream consumers (bidding, blending)
  depend on absolute scores, train/serve skew in a new feature, the metric was computed on
  sampled negatives, a bug in the serving path. Order of investigation: check serving
  features match training, check calibration, then check the evaluation methodology.*
- "How do you handle missing values?" → "What if missingness correlates with the label?" →
  *add an indicator; think about why — maybe the pipeline fails for a user segment; that's
  a data quality signal.*
- "Why do GBDTs beat neural nets on tabular data?" → *heterogeneous, irregular, often
  non-smooth feature-target relationships; trees are invariant to monotone transforms; NN
  inductive biases (smoothness, translation equivariance) don't match; with enough data and
  careful tuning NNs catch up but rarely win decisively.*
- "Precision or recall for X?" → always answer with the **cost of each error type** and the
  threshold-selection procedure.
