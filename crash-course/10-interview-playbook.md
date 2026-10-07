# 10 · The interview playbook (60 minutes)

## 1. What the interview will probably look like

An exec-referred MLE interview is usually a conversation with the hiring manager (45–60
min) and possibly a technical screen. Expect, in rough order of likelihood:

1. **Background and motivation** (5–10 min). Your honest story, told well.
2. **Conceptual ML with depth probes** (15–20 min). Modules 03–06.
3. **ML system design** (15–20 min). "How would you build X?" Modules 08–09.
4. **Coding** (20–45 min, maybe a separate round). Module 01 + `code/`.
5. **Behavioral** (10 min). Ownership, collaboration, failure, learning.
6. **Your questions** (5 min). Prepared, specific, revealing.

## 2. The depth-probe pattern (and how to prepare for it)

Interviewers ask one question and then **follow the thread** until you run out. The
scoring is: *how deep did they get before the candidate hit the wall, and how did they
behave at the wall?* Two rules:

- Prepare every topic to level 3 (the "Follow-up chains" sections in each module).
- At the wall, say so cleanly and reason aloud: *"I haven't done this, so let me reason
  from first principles: …"*. Reasoning at the wall scores higher than a memorized answer
  one level below it.

## 3. The honesty script (use it)

If asked "tell me about your ML experience":

> "I'll be direct: I don't have production ML experience. I got this interview through
> meeting [X], and rather than pretend, I spent the time since building a structured
> understanding of the field — from backprop and GPU memory to the two-stage recommender
> funnel and train/serve skew — and I wrote and ran the code for the core pieces. I
> know what I don't know. What I can offer is that learning speed, and I'd rather show it
> on a take-home or a trial project than claim things I can't back up."

Then let the interviewer steer. This is not a weakness statement; it is a calibration
statement, and calibration is one of the things being scored. If the role is strictly
senior, this won't get you that role — nothing will — but it is the path to a referral,
a junior role, or a second conversation.

If asked something you don't know:

> "I don't know that one. My guess would be [reasoned guess], because [mechanism]. How
> does it actually work here?"

Never: invent a project, invent a number, or say "I've done that" about something in
this course. Everything in here you can say "I understand" or "I've implemented a toy
version of" — and the second is true after you run the scripts.

## 4. ML system design template (15–20 min answer)

Spend 2 minutes clarifying, 2 on metrics/data, 8 on the architecture, 3 on serving/
monitoring, 2 on risks and iteration. Draw the funnel. Say the tradeoff for every choice.

1. **Clarify** goal, users, scale (items, QPS, data volume), latency, freshness, constraints.
2. **Metrics**: online primary + guardrails; offline proxy; how they relate.
3. **Data & labels**: sources, label definition, delays, negatives, splits.
4. **Features**: by entity; point-in-time; feature store; serving parity.
5. **Model(s)**: baseline → production architecture; why; training details.
6. **Serving**: request flow, latency budget, caching, fallback, versioning.
7. **Evaluation & rollout**: offline gates, shadow/canary/A-B, rollback.
8. **Monitoring & iteration**: drift, retraining, known biases, next steps.

Practice prompts (answer each aloud in 10 minutes):
- Video recommendations for a streaming app (module 08 is the answer).
- Fraud detection for payments (imbalance; PR-AUC; GBDT; real-time features; delayed
  chargeback labels; adversarial drift; cost-weighted thresholds; human review queue).
- Search ranking for e-commerce (query understanding; retrieval = inverted index + dense;
  LTR with GBDT/LambdaMART or neural ranker; click models for position bias; NDCG;
  interleaving).
- Ad click prediction (calibration critical: bids × pCTR; logistic regression / DCN;
  hashing trick; online learning; delayed conversions; auction effects).
- Customer-support assistant over internal docs (RAG; module 07 §6).
- Churn prediction (label window; temporal split; leakage from post-churn signals; GBDT;
  SHAP; intervention — the model is only useful with an action attached).

## 5. Coding round tactics

- Restate; ask input sizes and edge cases (empty, duplicates, ties, NaNs); brute force →
  improve; talk while typing; test on a tiny example by hand; state complexity.
- Likely asks and where you practised them: metrics (`code/07`), softmax/logistic
  regression/k-means (`code/01`, module 01 §6), attention (`code/04`), batching iterator
  (`code/03`), top-k per group with pandas/dicts (module 01 §4), two-sum/sliding window.
- If you freeze: write the loop version first; vectorize second. Correct and slow beats
  wrong and clever.

## 6. Question bank with level-3 answers

Answers are deliberately compact — expand with mechanisms from the modules.

### Fundamentals

**Bias vs variance?** Bias: systematic error from a model too simple to capture the
pattern (underfit). Variance: sensitivity to the particular training sample (overfit).
Diagnose with train vs validation curves; fix bias with capacity/features, variance with
data/regularization/ensembles. *Probe:* "Big nets overfit less as they grow — double
descent — so the classical U-curve is a guide, not a law."

**How do you split data?** By the independence structure: temporal split when time
matters (almost always in product data), group split by entity, stratified if small and
imbalanced. Random split is the exception. Keep a never-touched test set.

**What is regularization?** Anything that constrains the model to improve generalization:
L2/weight decay (Gaussian prior), L1 (sparsity), dropout, early stopping, augmentation,
label smoothing. *Probe:* "AdamW decouples decay from the adaptive step so decay is
uniform across parameters."

**Precision vs recall — which matters?** Depends on the cost of false positives vs false
negatives; choose the operating threshold from the business cost matrix on validation data,
report the full PR curve, and watch calibration if scores are consumed downstream.

**ROC-AUC vs PR-AUC?** ROC-AUC is the probability a random positive outranks a random
negative; insensitive to class imbalance, which makes it *misleading* for rare positives
(FPR denominator is huge). PR-AUC focuses on positives; use it for fraud/ads/medical.

**Log loss vs AUC?** AUC measures ranking only; log loss also measures calibration. A
model can improve AUC while worsening log loss (over-confidence). Downstream systems that
use absolute probabilities need log loss/calibration monitoring.

**Data leakage?** Information at training time that won't be identically available at
prediction time: target leakage, temporal leakage (random splits, future aggregates),
preprocessing fit on all data, duplicates across splits, group leakage. Detect: "too
good" results, one dominant feature, feature availability audit, point-in-time checks.

**Handle class imbalance?** Right metric first (PR-AUC, recall@precision); class weights or
resampling on the training set only; threshold tuning; focal loss; recalibrate afterwards
because re-weighting breaks calibration; make sure validation reflects the real prior.

**Why GBDT on tabular?** Heterogeneous, non-smooth feature-target relationships; trees are
invariant to monotone transforms, handle missing values and interactions natively; NN
inductive biases don't help; GBDTs are fast to train and tune. NN wins when there are
embeddings/multi-task/online learning needs or unstructured inputs.

**XGBoost vs LightGBM vs CatBoost?** LightGBM: leaf-wise growth + histogram binning, fastest,
can overfit small data (limit `num_leaves`). XGBoost: level-wise by default, robust,
widely supported. CatBoost: ordered target statistics for categoricals (no leakage),
symmetric trees, strong defaults.

**k-means limitations?** Must choose k; sensitive to init (k-means++) and scaling; assumes
spherical equal-variance clusters; local optima; use silhouette/elbow; for arbitrary
shapes use DBSCAN/GMM.

**PCA vs autoencoder?** PCA is linear, closed-form, orthogonal components ordered by
variance; an autoencoder is a nonlinear learned compression. PCA first (fast, stable).

**What's cross-entropy and why?** Negative log-likelihood of the true class under the
model's softmax; MLE for categorical outputs; gradient w.r.t. logits is `p − y`, which is
well-behaved; numerically computed with log-sum-exp.

### Deep learning

**Explain backprop.** Reverse-mode automatic differentiation: compute the loss forward,
store activations, apply the chain rule backward layer by layer to get gradients for all
parameters in one pass costing ~2× forward. Memory cost = stored activations.

**Vanishing/exploding gradients?** Products of many Jacobians shrink or grow
exponentially with depth. Fixes: ReLU-family activations, careful init (He/Xavier),
normalization, residual connections, gradient clipping, gated RNNs; for transformers,
pre-LN and warmup.

**BatchNorm vs LayerNorm?** BN normalizes each feature over the batch (needs batch
statistics; running stats at inference; train/eval modes; breaks with small batches and
sequences). LN normalizes over features per example (batch-independent; transformers).
*Probe:* "Forgetting `model.eval()` makes BN use batch stats at test time."

**Adam vs SGD?** Adam adapts per-parameter step sizes using first/second moment
estimates; robust, default for transformers/embeddings. SGD+momentum often generalizes
slightly better for CNNs and is cheaper in memory. AdamW for decoupled weight decay.

**Why warmup?** Early Adam moment estimates are noisy and the model is in a random state;
large early steps can wreck it (especially post-LN transformers). Linear warmup over a few
hundred–thousand steps, then cosine decay.

**Dropout at test time?** Disabled; training activations are scaled by `1/(1−p)` so
expected values match. (MC-dropout keeps it on deliberately to estimate uncertainty.)

**Why do residual connections help?** Identity path lets gradients flow unchanged through
depth; each block learns a residual correction; solves the degradation problem; makes
very deep nets trainable.

**Explain attention.** Each token forms a query, key, value; attention weights =
softmax(QKᵀ/√d); output = weighted sum of values. Scaling by √d keeps the logits' variance
controlled. Multi-head = parallel attention in subspaces. Causal mask for generation.
Complexity O(n²d).

**Why transformers over RNNs?** Parallel over sequence length (GPU-friendly), direct
token-to-token paths (no vanishing over distance), scale predictably. Cost: quadratic
attention and no inherent order (positional encodings).

**What's an embedding?** A learned dense vector per discrete ID, trained by backprop;
similarity structure emerges from the task; sparse gradients; dominates memory in recsys.

**Parameter count of a transformer block?** ≈ 12·d² (4d² attention projections + 8d² FFN
with 4× expansion). GPT-3: 96 layers, d=12288 → ~175B.

**How do you debug a model that won't train?** Overfit one batch; check loss at init;
check data/labels alignment (shuffle labels → should fail to learn); check LR; check loss/
output pairing; check `zero_grad`/`step`; check gradient norms; check normalization of
inputs; look at data samples by eye.

### PyTorch

**Walk through a training step.** zero_grad → forward → loss → backward → clip → step →
scheduler step; eval with `model.eval()` + `no_grad()`; checkpoint `state_dict`s.

**`model.eval()` vs `torch.no_grad()`?** Mode flag for dropout/BN vs disabling graph
building (memory/speed). Both for evaluation.

**DDP vs FSDP?** DDP replicates the model per GPU and all-reduces gradients; FSDP shards
params/grads/optimizer states and gathers them per layer on demand. FSDP when the
replicated state (≈16 bytes/param with Adam mixed precision) doesn't fit.

**What does `torch.compile` do?** Captures Python ops into a graph (Dynamo), fuses and
generates Triton kernels (Inductor); speeds memory-bound workloads; costs compile time and
recompiles on shape changes.

**Why is my GPU at 30%?** Dataloader-bound (CPU augmentation), host–device syncs
(`.item()` per step), tiny batches/kernels (launch-bound), or memory-bound ops. Profile,
then: more workers / offline preprocessing, remove syncs, bigger batch, compile.

### GPUs

**How does a GPU accelerate training?** Thousands of simple cores running the same
instruction on different data (SIMT), huge memory bandwidth, Tensor Cores for matmul;
deep learning is dominated by big matmuls, which are compute-bound and map perfectly.

**Memory-bound vs compute-bound?** Arithmetic intensity vs the ridge point
(~300 FLOP/byte on H100). Matmuls: compute-bound. Elementwise/norm/softmax/attention
(naive)/LLM decoding: memory-bound → fuse kernels, lower precision, batch.

**Mixed precision?** bf16/fp16 matmuls with fp32 master weights and fp32 reductions;
fp16 needs loss scaling (range), bf16 doesn't. 2× speed, half the activation memory.

**How much memory to train a 7B model?** Adam mixed precision ≈ 16 B/param → 112 GB +
activations → sharding (FSDP/ZeRO) or LoRA/QLoRA.

**What is FlashAttention?** Exact attention computed in SRAM tiles with online softmax;
never materializes the n×n matrix; memory linear in n; 2–4× faster.

**KV cache?** Stored keys/values of past tokens so decoding is O(n) per token; size
2·layers·kv_heads·head_dim·bytes per token (0.5 MB/token for LLaMA-2-7B); GQA and paging
reduce the pressure.

### Recommendation systems

**Design a recommender.** Module 08 §13 template: objective & label → metrics → funnel
(retrieval: two-tower + ANN + item-kNN + popularity; ranking: multi-task DCN/DLRM-style with
cross features; re-rank: diversity/rules) → features with point-in-time correctness and
served-feature logging → training cadence/negatives/logQ → serving flow & latency → eval
(temporal split, full-ranking Recall@K, GAUC; interleaving → A/B; long-term holdout) →
biases/cold start/monitoring.

**Why two stages?** Retrieval must be cheap over millions of items (independent towers,
precomputed index); ranking needs cross features and heavy models but can only afford
hundreds of candidates.

**Negatives in retrieval training?** Implicit feedback has none; use in-batch negatives
(efficient, popularity-biased → logQ correction), random negatives, hard negatives with
care.

**Position bias?** Top slots get clicked regardless; model position as a feature during
training, fix it at inference; or IPS with propensities from randomization.

**Offline AUC up, online flat/down?** Check deployment/feature parity, calibration,
evaluation protocol (temporal? sampled negatives?), power and duration, segment effects,
feedback-loop bias of logged data.

**Cold start?** Items: content embeddings, exploration slots, fast indexing. Users:
onboarding, context priors, session models, bandits.

**How do you evaluate?** Temporal split; full-ranking Recall@K/NDCG@K against popularity
and item-kNN baselines; GAUC/log loss for rankers; online interleaving then A/B with
guardrails; long-term holdout.

**ANN tradeoffs?** HNSW (recall/latency best, memory heavy, slow updates), IVF-PQ
(billion-scale, lower recall), ScaNN; measure ANN recall vs exact; version index with
the tower.

### MLOps / systems

**Train/serve skew?** Features computed differently offline and online. Fix: same code
path or log served features and train on them; parity tests; feature store.

**How do you monitor a model?** System metrics; data quality (nulls, ranges, freshness);
drift (PSI/KL per feature, prediction distribution); delayed true metrics by slice;
alerts on statistical anomalies, not just exceptions.

**Rollout plan?** Offline gates → shadow → canary with guardrails → A/B → full; old model
hot for rollback; versioned artifacts (model + tokenizer + features + index).

**Batch vs online inference?** Freshness need, latency, QPS, cost. Precompute when inputs
are slow-changing; online when context matters.

**What's a feature store?** Registry + offline store with point-in-time joins + online KV
store, fed by the same transformations; solves skew and reuse.

### LLMs

**How are LLMs trained?** Next-token prediction on trillions of tokens (6ND FLOPs), then
SFT, then preference optimization (RLHF/DPO). Scaling laws; Chinchilla ≈ 20 tokens/param
compute-optimal; production over-trains.

**LoRA?** Low-rank update ΔW = BA on frozen weights; ~1% trainable params; small optimizer
memory; mergeable at inference; QLoRA adds a 4-bit base.

**RAG vs fine-tuning?** RAG for changing/citable knowledge; fine-tuning for behavior,
format, domain style, latency. Evaluate retrieval separately.

**Why is decoding slow and how to speed it up?** Memory-bound: each token reads all weights
and the KV cache. Batch (continuous batching), paged KV, quantize weights, speculative
decoding, GQA, tensor parallel.

**Hallucination?** The model samples plausible continuations without grounding; mitigate
with retrieval + citations, constrained outputs, verification steps, calibrated refusals,
evaluation sets; monitor in production.

### Statistics / experimentation

**p-value?** Probability of data at least this extreme under the null; not the probability
the null is true. Pair with effect size and confidence interval; pre-register the primary
metric; beware peeking and multiple comparisons.

**How long to run an A/B test?** Compute sample size from the minimum detectable effect
and variance (power 80%, α 5%); ≥ 1–2 full weekly cycles; account for novelty; don't stop
early on significance.

**Randomize by user or request?** User (consistency and independence); analyze at the
same unit or correct the variance (delta method / clustered SE).

## 7. Behavioral questions without work experience

Use real stories from anywhere (school, jobs, projects, this 12-hour sprint) with STAR
(Situation, Task, Action, Result) and a lesson. Themes they probe: ownership (you fixed
something nobody asked you to), handling failure (what you changed afterwards), learning
(this course is literally the example), collaboration/conflict (disagreeing with data),
prioritization (triage order under a deadline — you have one tonight).

## 8. Questions to ask them (pick 3; they double as signals)

- "How do you measure a model's success online, and how well do your offline metrics
  predict it?"
- "What was your last train/serve skew or silent model failure, and what did you change?"
- "How often do models retrain, and what gates promotion?"
- "Do you have a feature store or do teams log served features?"
- "How do you handle cold start and exploration in the recommender?"
- "What does the on-call look like for ML services?"
- "For someone with my background, what would the first 90 days look like, and what
  would make you say the hire worked?"

## 9. Red flags to avoid

Buzzword answers without mechanism; claiming experience; absolute statements ("always
use Adam"); not asking clarifying questions in design; proposing deep learning for a
tabular problem without a baseline; ignoring metrics, cost, and latency; dismissing data
quality; getting defensive when corrected — instead say "that's a good point, so then…"

## 10. Last-hour checklist

- Say the whole-field paragraph (README) from memory.
- Draw the recsys funnel and the training loop on paper.
- Recite: 16 B/param, 6ND, ~20 tokens/param, 0.5 MB/token KV, ~300 FLOP/byte ridge,
  fp16 max 65,504, loss at init = ln(C).
- Run `code/03_torch_training_loop.py` once more and read every comment.
- Rehearse the honesty script and three questions to ask.
- Sleep.
