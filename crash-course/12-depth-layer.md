# 12 · The depth layer: mechanisms, predictions, and your own numbers

Run `code/10_experiments_training_dynamics.py`, `code/11_experiments_logged_data.py`,
`code/12_experiments_systems.py` alongside this module. Every ★ item below refers to a
section of those scripts; the numbers quoted are what they print on CPU in under a minute.

## 0. Honest assessment of modules 01–11

You asked whether the course is regurgitation. Measured against a concrete standard —
*does the material give you a mechanism from which you can derive a prediction a listicle
could not?* — the answer for most of it is no, and here is the evidence:

- A text search over the modules for mechanism words ("because", "which is why", "so
  that") finds **one such line in 256 lines** of the deep-learning module, 4 in the
  PyTorch module, 4 in the GPU module. They are organized *lists of true facts*.
- The content is the canon every prep guide and every model reproduces: accuracy under
  imbalance, leakage, train/serve skew, two-tower + logQ, the roofline. Correct,
  necessary as vocabulary, and undifferentiated. An interviewer has heard it all.
- The "experience signals" sections are the weakest part. They coach phrasings that
  imply habits ("I check df.shape after every merge") that you do not have. That is a
  soft form of the fabrication the README warns against. Do not use them as written.
- What *was* above the canon: the running experiments with numbers (logQ changing head
  recall from 0.31 to 0.84 while coverage fell from 98% to 56%; sampled metrics inflating
  Recall@10 from 0.27 to 0.59; KV-cache equivalence; the gradient check), the follow-up
  chains, and a few mechanisms (why AdamW, why FlashAttention is faster, why two towers
  can't see cross features).

Why every model gives the same "99% accuracy is misleading / don't invent war stories"
advice: it is the shared training canon. Rarity is not the test of insight. The test is
whether the idea lets you *predict* what a system will do in a situation you have never
seen — which is exactly what a depth probe measures. That is the standard for this
module. Nothing here is unknown to experts (that is the point: it is what experts know
that the canon leaves out), but each item is given as **mechanism → prediction →
experiment you ran**, and several are *reframings* that collapse a dozen facts into one
idea (§4: labels are produced by a policy; §2.2: weight decay is a learning-rate knob).

Priorities: ★★★ run and understand tonight; ★★ read and understand; ★ if time allows.

## 1. Making "experience" true in one night: the lab notebook

A war story you invent collapses at "what was the number?". An experiment you ran last
night does not — and it is a *better* signal, because it shows how you learn. Keep a file
(`notebook.md`) and for each experiment section write four lines:

```
Expected:   <what the textbook answer predicted>
Observed:   <the number the script printed>
Mechanism:  <one sentence, from this module>
In prod:    <what I would therefore do / monitor>
```

Example, from `11_experiments_logged_data.py` §2:

> *Expected*: the ranker closest to true relevance should score best offline.
> *Observed*: the naive Recall@10 on logged clicks ranked a near-copy of the incumbent
> (0.87) above the truly best ranker (0.73); the online truth and the IPS estimate both
> reversed that. *Mechanism*: logged clicks exist only on items the incumbent showed.
> *In prod*: log propensities, keep an exploration slice, evaluate new policies with IPS,
> and treat offline wins of incumbent-like models with suspicion.

In the interview, say it exactly like that: "I haven't run this at scale, but I simulated
it, saw X, and the mechanism is Y, so I'd do Z." It is true, specific, and it invites a
probe into mechanism — which you can now answer. It also answers the behavioral question
"tell me about something you learned recently" with real content.

## 2. Training dynamics

### 2.1 ★★★ bf16 does not add noise to small updates; it deletes them (`10` §A)

**Mechanism.** bf16 keeps 8 significant bits → relative precision 2⁻⁸ ≈ 0.4%. An update
smaller than half an ulp of the weight rounds to *no change*. Adam steps have magnitude
≈ lr per parameter, weights are ~0.02 at init, so at lr = 1e-5 only ~16% of bf16 weights
change, at 1e-6 ~2%. fp16 has the opposite problem: range (max 65,504; `exp(12)` is
already inf; a sum of ones stalls at 2048 because the ulp there is 2).

**Predictions.** Pure-bf16 training stalls at the end of a cosine schedule; fp32 master
weights (or stochastic rounding / Kahan-summation optimizers) fix it. Adam's `eps=1e-8`
is *zero* in fp16 → NaNs for parameters with zero gradient. Reductions (loss sums,
LayerNorm variance, softmax denominators) must run in fp32, which is what `autocast`
does; "loss became NaN at step N" in fp16 usually coincides with a long sequence or an
attention-logit growth, which is why QK-norm exists.

### 2.2 ★★★ With normalization, weight decay is an effective-learning-rate knob (`10` §B)

**Mechanism.** For a weight matrix W followed by LayerNorm/BatchNorm, the function is
invariant to the scale of W: f(3W) = f(W) (the script checks this exactly). Two exact
consequences: the gradient scales as 1/‖W‖ (ratio 0.333 at 3W), and the gradient is
orthogonal to W (cos = 0.0000). So an SGD step can only *increase* ‖W‖
(‖W+g‖² = ‖W‖² + ‖g‖²), which shrinks the effective step size lr/‖W‖². Weight decay
pulls the norm down until decay balances the orthogonal growth — an equilibrium norm.
Observed: no decay → ‖W‖ 4.7 → 10.9, effective lr 0.18× its initial value; wd = 0.05 →
‖W‖ pinned at 1.8, effective lr 6.8×, and the *worst* loss of the three runs — not
because it regularized, but because the effective step became too large.

**Predictions.** (1) Removing weight decay from a normalized network makes late training
*slower*, not more overfit. (2) lr and wd are one knob with two names; what matters is
lr·wd (Van Laarhoven 2017; Zhang, Wang & Grosse 2019; Li & Arora 2019 show an exponential
LR schedule is equivalent). (3) Do *not* decay LayerNorm gains and biases: they are not
scale-invariant, so decay there really shrinks the function. (4) Adam + L2 breaks the
equilibrium logic because the decay term gets divided by √v per parameter; AdamW
restores it — that is the real reason AdamW exists, beyond "decoupled is cleaner".

### 2.3 ★★★ Adam after a quiet period: the mechanism of loss spikes, and β2 = 0.95 (`10` §C)

**Mechanism.** m (momentum) adapts in ~1/(1−β1) ≈ 10 steps; v (the per-parameter scale)
remembers ~1/(1−β2) steps — 1,000 at β2 = 0.999. If a parameter's gradient has been small
for a while (a rare token's embedding, a dormant head) and then becomes large, m grows
fast while v is still tiny → the step m/√v is amplified. Observed: a ×1000 gradient jump
gives a peak step of 6.4× lr for 275 steps at β2 = 0.999, and 1.1× lr at β2 = 0.95. Even a
×3 jump gives 2.6×. Clipping the gradient only helps if the clip is tight *relative to the
quiet level*, because the ratio is scale-free.

**Predictions.** GPT-3, LLaMA, and most large runs use β2 = 0.95; PaLM anneals it. Loss
spikes tend to follow quiet, low-gradient stretches and to start in embeddings/attention
logits (Molybog et al. 2023; Wortsman et al. 2023 — mitigations: QK-norm, z-loss, lower
β2, warmup). Gradient clipping's main job under Adam is protecting v from pollution, not
limiting the step. The standard operational response — roll back to the last checkpoint
and skip the offending batches — works because v recovers.

### 2.4 ★★ The right batch size changes during training (`10` §D)

**Mechanism.** The gradient-noise scale B_noise = tr(Σ)/‖G‖² (McCandlish et al. 2018) is
the batch size at which noise and signal are comparable; above it, a bigger batch gives
almost no extra progress per step. It is estimated from two batch sizes
(E‖g_B‖² = ‖G‖² + tr Σ/B). Observed on a small MLP: ~90 at init, ~1,300 after 300 steps,
~2,700 after 1,500 — it grows as the loss falls because the true gradient shrinks while
the per-example noise does not.

**Predictions.** Large batches waste compute early and are necessary late — hence batch
ramps (GPT-3: 32k → 3.2M tokens). The linear LR-scaling rule only holds below B_noise.
"8 GPUs is not 8× fewer steps to target" is expected, not a bug. Measuring B_noise is a
cheap way to decide how many GPUs a job can use efficiently.

### 2.5 ★★ Double descent: the classical overfitting curve is half the picture (`10` §E)

**Mechanism.** With a min-norm (or SGD-trained) solution, test error peaks when the
parameter count ≈ sample count (the interpolation threshold: the fit is forced to pass
through the noise with the fewest degrees of freedom to spare) and *falls again* beyond it.
Observed with random-feature regression, n = 100: MSE 0.46 at D = 10, **478** at D = 100,
0.25 at D = 2,000; ridge removes the peak entirely (0.43 at D = 100).

**Predictions.** Near the threshold, adding data or parameters can hurt (Nakkiran et al.
2019: also epoch-wise — validation loss can rise and then fall). "Validation got worse
when I added data" is not automatically a bug. Regularization and early stopping are what
make the classical story hold in practice.

### 2.6 ★★ Mechanisms without experiments (state them with the mechanism, not the slogan)

- **BatchNorm** does not work by reducing "internal covariate shift" (the original claim);
  it smooths the loss landscape (lower gradient Lipschitz constant), allowing higher LR
  (Santurkar et al. 2018). Its train/eval mismatch is a *second* effect: at train time the
  batch statistics are stochastic (a regularizer); at eval they are fixed.
- **Dropout before BatchNorm** causes a variance shift: dropout changes activation
  variance between train and test, BN's running variance was computed under train-time
  dropout → mismatch at eval (Li et al. 2018). Put dropout after the last BN, or don't mix.
- **Label smoothing hurts distillation**: it collapses the logit geometry ("dark
  knowledge") the student needs (Müller, Kornblith & Hinton 2019). A teacher trained with
  label smoothing is a worse teacher even though it is a better classifier.
- **Dead ReLUs** come from a single large step pushing a unit's bias far negative; the unit
  then has zero gradient forever. Mechanism → prediction: the fraction of dead units is a
  function of LR and init, and it is permanent, so monitor it early.
- **Pre-LN vs post-LN**: in post-LN the gradient norm at init shrinks with depth (Xiong et
  al. 2020), which is why it needs warmup; pre-LN's residual stream norm *grows* with depth,
  so later layers contribute relatively less — the reason for variants like sandwich/deep
  norm.
- **μP (maximal update parametrization)**: under standard parametrization the optimal LR
  shifts with width because the output's update scale grows with width; μP rescales init
  and LR per layer so hyperparameters tuned on a small model transfer to a large one (Yang
  et al. 2022). That is how labs tune 100B-parameter models without sweeping them.
- **Activation memory** per transformer layer in bf16 ≈ s·b·h·(34 + 5·a·s/h) bytes
  (Korthikanti et al. 2022: s = sequence, b = batch, h = hidden, a = heads). The 5·a·s² term
  is attention; FlashAttention removes most of it, and *selective* recomputation of just
  that term gets most of checkpointing's memory win for a fraction of its compute.
- **Gradient clipping** by global norm makes the update size ≤ lr·c; if most steps are
  clipped you are doing normalized-gradient descent at a different LR than you think.
  The clip fraction is a metric to log.

## 3. Evaluation: a metric is a decision rule, not a number

### 3.1 ★★★ AUC measures orderings you never act on

**Mechanism.** AUC = P(random positive scores above random negative). If the system acts
only on the top 1% of scores (fraud review, a ranked slate), most pairs that AUC counts
involve negatives you would never surface. Two models with equal AUC can differ hugely at
the operating point, and a model can gain AUC by reordering the bottom 90% while losing
precision at the top. Under heavy imbalance the FPR axis (denominator = all negatives)
compresses the region that matters.

**Predictions and practice.** Report the metric at the operating point (precision@k,
recall at fixed FPR, partial AUC); in recsys use GAUC (per-user AUC) because global AUC
rewards telling heavy users from light users, which no slate ranking needs. AUC is
invariant to monotone transforms, so it *cannot* see calibration — "AUC up, revenue down"
is consistent with a calibration regression in a system that multiplies scores by bids.

### 3.2 ★★ Log loss moves for two unrelated reasons

Log loss = ranking quality + calibration. A 2% log-loss improvement may be pure
calibration (temperature scaling the old model would have achieved it). Test: fit a
temperature on the old model's validation logits; if the gap closes, the new model did not
rank better. Facebook's *normalized entropy* (log loss ÷ entropy of the base rate, He et
al. 2014) exists so the number is comparable across base rates.

### 3.3 ★★★ The offline–online gap has structure; diagnose it in order

1. **Deployment reality** — is the online model actually the new one, with the same
   features (§5.6, `12` §4)? Most "the A/B is flat" tickets end here.
2. **Exposure bias in the offline metric** — logged positives only exist on what the old
   policy showed (`11` §2); incumbent-like models score better offline for that reason
   alone.
3. **Calibration** — downstream consumers (bidding, blending, thresholds) use absolute
   scores (`11` §3).
4. **Power and duration** — the test may be unable to see the effect (`12` §6); check the
   MDE before concluding "no effect".
5. **Metric mismatch** — the offline metric is a proxy for the online one; check the
   historical correlation between offline deltas and online deltas, not just this one.

### 3.4 ★★★ Experimentation: variance is something you engineer (`12` §6)

Observed: with 14 daily looks and stop-at-first-significance, an A/A test "wins" 21.8% of
the time (5.2% at the planned end). Randomizing users but computing variance over
impressions gives a 14.7% false-positive rate; the delta method at the user level gives
5.0%. CUPED with a 0.8-correlated pre-period metric removes 64% of the variance — the
power of 2.8× more users for free (Deng et al. 2013).

**Mechanisms.** Peeking: each look is a new chance to cross the threshold. Unit mismatch:
impressions from the same user are correlated, so the effective sample size is the number
of users, not impressions (and ratio metrics like CTR need the delta method: Deng et al.
2018). CUPED: subtract the part of the metric predictable from pre-experiment data; the
residual has lower variance and the same expectation under randomization.

**Interleaving** is 10–100× more sensitive than A/B for ranker comparisons (Chapelle et
al. 2012) for the *same reason CUPED works*: it is a paired, within-user comparison, so
between-user variance cancels. Interference (two-sided marketplaces: the treatment's extra
exposure comes out of control's creators) violates SUTVA; cluster or switchback designs
exist for that.

## 4. The unifying idea: labels are produced by a policy

Almost every "offline looked great, online didn't" story in recommendation, search, ads,
fraud, lending, and moderation is the same structure: **the labels you train on were
generated by a decision process (the old model, a review queue, a slate layout), and your
new model changes that process.** Once you see the shape you can predict the failure.

### 4.1 ★★★ Position bias is confounded by the ranker that produced the logs (`11` §1)

Observed: with a deterministic incumbent, the item-plus-position model learns θ = [1, 1,
1, …] — nothing — because item and position are perfectly collinear. With a slightly noisy
incumbent it recovers θ but has no estimate at all for most items (never shown). With 5%
random slates it recovers both (corr 0.98).

**Mechanism.** Good items sit on top, so CTR-by-position overstates the position effect;
any model, including "position as a feature", can only separate the two using whatever
variation in positions the logging policy *happened* to create. Intervention harvesting
(Agarwal et al. 2019) deliberately uses variation across ranker versions; randomized
swaps/slices are the clean fix; IPS-weighted learning-to-rank (Joachims et al. 2017) uses
the estimated propensities.

**Prediction.** If the team tells you "we add position as a feature and set it to 1 at
serving", ask what variation identifies it. If the answer is "none", the model has learned
an arbitrary split between item and position effects.

### 4.2 ★★★ Offline evaluation has incumbency bias; off-policy evaluation fixes it only with exploration (`11` §2)

Observed: naive Recall@10 on logged clicks ranks "incumbent + noise" (0.87) above the true
relevance order (0.73); online truth is 1.22 vs 1.42 expected clicks — reversed. The IPS
estimate (clicks weighted by 1/propensity, position-based) recovers the right order. The
third candidate, "rank by logged item CTR", is the feedback loop in one line: it reproduces
the incumbent.

**Mechanism.** Clicks exist only where the old policy created exposure. A metric that
counts logged positives credits a candidate for agreeing with the incumbent. Inverse
propensity scoring reweights each logged outcome by how much more (or less) the new policy
would have shown it (Li et al. 2011 for bandits; Li et al. 2018 for slates). Requirements:
propensities must be logged or estimable, and must be non-zero (positivity) wherever the
new policy puts mass — which is what an exploration slice buys. Variance explodes for
policies far from the incumbent; clipping and self-normalized / doubly-robust estimators
trade bias for variance.

**Prediction.** Without logged propensities and some exploration, a team cannot evaluate
any substantially different policy offline; they will systematically ship incremental
changes and call it "offline metrics don't transfer". Ask whether they log propensities.

### 4.3 ★★★ Negative downsampling is a prior shift; the fix is one line (`11` §3)

Observed: keeping 10% of negatives leaves AUC exactly unchanged (0.8537) and inflates the
mean prediction from 3.6% to 19.3%; `q = p / (p + (1−p)/w)` restores 3.6%.

**Mechanism.** Subsampling negatives multiplies the odds by 1/w; a logistic model learns
the shifted intercept; the correction multiplies the odds back (He et al. 2014). The same
formula handles *class weights* (which shift calibration the same way) and any *base-rate
change* between training and serving (seasonality, new market, new surface):
`logit' = logit + log(π_new/π_train) − log((1−π_new)/(1−π_train))`. In ads, a
miscalibration of 10% on a segment misallocates roughly 10% of that segment's spend,
because the auction uses pCTR × bid — which is why calibration layers are retrained more
often than the ranker and checked per segment.

### 4.4 ★★★ Delayed feedback makes recent examples look negative (`11` §4)

Observed: true CVR 5.00%, naive labels 4.36% overall, **0.60%** for clicks from the last
day. Using only matured clicks is unbiased but discards 66% of data and is three weeks
stale; weighting observed positives by 1/P(converted by now | converts) is unbiased on all
data (Chapelle 2014).

**Prediction.** Any feature correlated with recency (campaign age, new item, new user)
learns "new = low CVR", and the model under-bids on exactly the inventory the business
wants to grow. Production patterns: attribution window + wait; importance weights; fake
negatives re-inserted as positives on conversion (Ktena et al. 2019).

### 4.5 ★★ Selective labels: you only have labels where the old policy looked (`11` §5)

Observed: a model trained on reviewed-only fraud cases has 0% recall on a fraud pattern
the old rule never flagged (one third of all fraud) and AUC 0.69 on full traffic; adding a
2% random audit stream gives 100% recall on that pattern and AUC 0.87.

**Mechanism.** Labels are missing *not at random*: their presence depends on the old
model's decision. Where the model is well-specified and selection depends only on
observed features, there is no bias — but real models are misspecified and new patterns
live precisely where no labels exist (Lakkaraju et al. 2017; "reject inference" in
lending). Only randomization creates support there.

### 4.6 ★★ Three more label mechanisms

- **Purged, embargoed temporal splits**: with label latency L, a training example near the
  split boundary has a label determined by events inside the test window; purge the last L
  of training and embargo the first L of test (de Prado). Otherwise temporal splits still
  leak.
- **Label noise is memorized late**: networks fit clean patterns first and noisy labels
  afterwards (Arpit et al. 2017), so early stopping is a noise-robustness tool and
  "small-loss" examples are the clean ones — the basis of co-teaching-style methods.
- **Ranking-stage selection bias**: the ranker is trained only on items retrieval surfaced,
  and CVR models only on clicked impressions. ESMM's pCTCVR = pCTR × pCVR trick exists to
  train over the full impression space; the same shape recurs at every funnel stage.

## 5. Recommendation-system mechanisms

### 5.1 ★★★ An MLP on [user; item] struggles to learn a dot product (`12` §2)

Observed: with 80× the parameters and the same 60k examples, an MLP on the
concatenation reaches MSE 1.66 on a 32-d dot-product target; a bilinear model reaches
0.09.

**Mechanism.** Multiplicative interactions are not a natural function of ReLU MLPs (they
need many piecewise-linear pieces to emulate a product); Rendle et al. 2020 showed that
the NCF results did not replicate against a tuned dot product. Consequences: two-tower
retrieval uses a dot product (which is also what ANN indexes require); ranking models that
can afford it add explicit crossing (FM, DCN cross layers, attention over history) rather
than trusting "deep is universal".

### 5.2 ★★★ Two retrains are not in the same embedding space (`12` §1)

Observed: run 1 and run 2 each reach Recall@10 ≈ 0.19–0.21; run 1's user tower against
run 2's item index gives 0.03 (random). Aligning run 1 onto run 2 with the best rotation
(orthogonal Procrustes) restores 0.196.

**Mechanism.** Scores depend only on dot products, so any rotation applied to both towers
leaves them unchanged; independent trainings land in different rotations (and different
local optima). Nothing errors: the index serves, latency is fine, every dashboard is green,
and every recommendation is garbage. Practice: version tower and index together, swap
atomically, warm-start retrains from the previous checkpoint so that spaces stay
compatible for incremental index updates, and monitor the alignment residual between
consecutive item-embedding versions as a deploy check.

### 5.3 ★★ ANN recall is not uniform; it fails on the tail (`12` §3)

Observed: an IVF index at 5% of exact-search work has 99.5% recall on clustered items and
58% on isolated ones.

**Mechanism.** Inverted-file (and graph) indexes assume the nearest neighbors share a
coarse region; items between clusters (niche, new, unusual) get probed last. The
catalogue slice you most want to surface is the one the index loses. Measure ANN recall
against exact search on *that* slice, and expect index parameters to be tuned against the
end metric, not a global recall number.

### 5.4 ★★ Sampled softmax and the logQ correction, precisely

In-batch negatives are drawn in proportion to item popularity. The gradient of the sampled
softmax then penalizes popular items in proportion to how often they appear as negatives
— a bias, not just noise. Subtracting log Q(item) from the logit makes the sampled
objective a consistent estimator of the full softmax (Yi et al. 2019, who estimate Q with
a streaming hash-based frequency estimator). With mixed negatives (in-batch + uniform) the
correction uses the *mixture* probability. You saw the effect in `06`: without correction,
head recall 0.31 and coverage 98%; with it, 0.84 and 56%. Hard negatives add a second
problem: a "hard negative" that is actually an unlabeled positive teaches the model the
wrong thing, so practice uses semi-hard mining (harder than random, easier than the
positive) or filters the top of the candidate list.

### 5.5 ★★ Evaluate each stage against the next stage, not only against clicks

Retrieval's job is to surface what the ranker would have ranked highest; its right offline
metric is "recall of the ranker's top-k over the full catalogue", which you can compute
by scoring everything offline. Clicks-based recall confounds retrieval quality with
ranking quality and exposure bias (§4). The same logic gives the ranker a stage-specific
check: does it preserve the order of the re-ranker on the candidates it never saw?

### 5.6 ★★★ Train/serve skew that monitoring cannot see (`12` §4)

Observed: serving 30% of requests with another user's feature row leaves every per-feature
mean and standard deviation and the prediction distribution identical, while PR-AUC drops
from 0.51 to 0.40.

**Mechanism.** Misaligned joins, stale caches, and batch misalignment permute rows; the
marginals are preserved by construction, so PSI/KL drift monitors stay green. Only
row-level parity (same entity, same timestamp, offline vs online value), training on
logged served features, or a canary comparing offline and online scores for the same
requests can catch it. "We monitor feature distributions" is necessary and insufficient.

### 5.7 ★ Freshness and staleness

A model trained on historical logs learns a time-averaged popularity; serving it today
under-predicts new content, which is what YouTube's "example age" feature (set to 0 at
serving) corrects. Ads CTR systems retrain hourly not because the world changes that fast
but because the *candidate set* does; training on the most recent hours alone imports an
hour-of-day artifact unless the data are stratified.

## 6. GPU and serving mechanisms

### 6.1 ★★★ The ridge point rises every generation; fusion is a trend, not a trick

Ridge point = peak FLOP/s ÷ bandwidth: A100 bf16 312/2.0 ≈ 156 FLOP/byte; H100
989/3.35 ≈ 295; H100 fp8 ≈ 590. Compute grows faster than memory bandwidth, so each
generation pushes more kernels (and smaller GEMMs: an M = 16 decode-style GEMM has
intensity 16) into the memory-bound regime. FlashAttention, fused optimizers,
`torch.compile`, and fp8 all exist because of this arithmetic; expect the pressure to
increase.

### 6.2 ★★★ int8 activations break on outlier channels (`12` §5)

Observed: a few channels 60× larger than the rest give 10.8% relative error with
per-tensor int8, 5.0% per-token, 1.1% after SmoothQuant's per-channel rescaling (X/s,
s·W: product unchanged), 0.8% for weight-only int8.

**Mechanism.** One scale per tensor is set by the largest value; everything else collapses
onto a few quantization levels. LLM activations contain such channels (Dettmers et al.
2022 handled them in mixed precision; Xiao et al. 2023 migrated the difficulty into the
weights). Consequence: weight-only int4/int8 is the serving default — decode is
memory-bound, so shrinking weights is what speeds it up — and activation quantization
needs smoothing or outlier handling.

### 6.3 ★★ Decode economics

Each decode step reads all weights plus the KV cache to produce one token per sequence,
so tokens/s/GPU grows almost linearly with batch size until the ridge point; the cost per
token drops accordingly, and latency per token rises slowly then sharply. Speculative
decoding is a free lunch only while decode is memory-bound: at high batch, verification is
no longer free and the speedup collapses. Prefill is compute-bound and decode is
memory-bound, which is why serving systems disaggregate them onto different pools and why
TTFT and tokens/s are tuned separately. Temperature 0 is not deterministic across
requests because continuous batching changes the GEMM shapes, kernel selection, and
therefore floating-point rounding.

### 6.4 ★ Four more mechanisms

- **Tile and wave quantization**: a GEMM is executed as tiles over 132 SMs (H100); 136 tiles
  need two waves, the second 3% full, so the kernel runs at roughly half efficiency. Pad
  vocabularies and hidden sizes to multiples of 64–128.
- **Attention sinks**: softmax must sum to 1, so heads park "no-op" mass on the first
  tokens; evicting them from a sliding-window KV cache destroys quality (Xiao et al.
  2023), which is why StreamingLLM keeps the first tokens and why dropping BOS breaks
  models.
- **RoPE and context extension**: positions beyond the trained length produce rotation
  frequencies the model never saw; *interpolating* positions into the trained range works
  with little fine-tuning where extrapolation fails (Chen et al. 2023).
- **Caching-allocator OOMs**: PyTorch reserves and reuses blocks; an OOM with "20 GB free"
  is fragmentation (no contiguous block) or a transient peak, not a leak;
  `expandable_segments` and length-bucketed batches address it, `empty_cache()` does not.

## 7. What changes in the interview

1. Drop the "experience signals" phrasing from modules 01–09. Replace with notebook
   statements (§1): expected → observed → mechanism → what I'd do.
2. The three-sentence answer that reads as depth: **mechanism**, **prediction**, **what
   I'd monitor**. Example — "Why β2 = 0.95?": *v remembers 1/(1−β2) steps while m adapts in
   10, so after a quiet period the step is amplified several-fold; I'd expect spikes after
   low-gradient stretches; I'd log the clip fraction and the update/weight ratio per layer.*
3. Calibrated scope: you now hold mechanisms for roughly 25 topics. For everything else,
   "I understand the behavior but not the mechanism — my guess is …" is the right answer.
4. Questions to ask them that carry signal (pick two):
   - "Do you log propensities, and is there an exploration slice for off-policy evaluation?"
   - "How are the user tower and the item index kept in the same space across retrains?"
   - "How do you handle delayed conversions — attribution window, importance weights, or
     fake-negative correction?"
   - "Are there row-level parity checks between offline and online features, or only
     distribution monitoring?"
   - "Which offline metric has historically predicted your A/B results, and how well?"

## 8. If you have only one hour

Run `11_experiments_logged_data.py` and read §4 (the policy-generated-labels idea covers
five interview topics at once); then `10` §B–C with §2.2–2.3; then `12` §1, §4, §6 with
§5.2, §5.6, §3.4. Write the notebook lines for each as you go.
