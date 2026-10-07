# 08 · Recommendation systems (75 minutes) — where "experience" questions concentrate

Run `code/05_matrix_factorization_recsys.py` (BPR matrix factorization with a proper
temporal split, full-ranking Recall@K/NDCG@K vs a popularity baseline) and
`code/06_two_tower_recsys.py` (two-tower retrieval with in-batch negatives and the logQ
correction) with this module.

## 1. Problem framing (what managers probe first)

"Recommend items to users" hides the real questions:
- **What is the objective?** Clicks (cheap, abundant, biased toward clickbait), watch
  time/dwell (favors long content), purchases/conversions (sparse, delayed), explicit
  ratings (rare), surveys (very sparse but closest to satisfaction), long-term retention
  (the thing that matters; nearly impossible to optimize directly). Real systems predict
  several and combine them with a **value model** `Σ wₖ · pₖ` whose weights are tuned via
  A/B tests.
- **What is the label?** Implicit feedback is *positive-only*: a non-click is not a
  negative, it may be "never seen". Delayed labels need an **attribution window**
  (purchase within 7 days of the click). Accidental clicks (<2 s dwell) are often
  removed. **Label definition decides what the model learns.**
- **Constraints**: catalog size (10⁴ vs 10⁹), QPS, latency budget (typically 100–300 ms
  end-to-end, tens of ms for the model), freshness (news: minutes; movies: days), cold
  start, business rules, fairness/diversity.

## 2. The industrial funnel (draw this on the whiteboard)

```
  10⁶–10⁹ items        ~10³ candidates        ~10²                 ~10
  ┌──────────┐   ┌───────────────────┐   ┌─────────────┐   ┌────────────────┐
  │ RETRIEVAL│ → │ PRE-RANK (light)  │ → │ RANKING     │ → │ RE-RANKING     │ → slate
  │ several  │   │ small model on    │   │ heavy model │   │ diversity,     │
  │ sources  │   │ cheap features    │   │ rich feats  │   │ rules, policy  │
  └──────────┘   └───────────────────┘   └─────────────┘   └────────────────┘
    ms, high recall   ms                    10s of ms          ms
```

- **Retrieval / candidate generation** must be *cheap* and *high-recall*: it cannot score
  every item with a big model. Multiple sources are merged: two-tower ANN, item-to-item
  collaborative filtering ("users who watched X also watched"), content/semantic
  similarity, popularity/trending, recently viewed, social/graph, editorial. Each source
  covers a failure mode of the others.
- **Ranking** scores a few hundred candidates with a heavy model that can use **user–item
  cross features** and full context. Usually **pointwise** (predict pCTR, pWatch, pLike…)
  with calibrated outputs, then combined.
- **Re-ranking** applies diversity (MMR, DPP), dedup, freshness, business constraints,
  exploration slots, and sometimes a slate-level model.
- Everything is **logged** (impressions, positions, features used, model versions) because
  the logs are tomorrow's training data.

## 3. Classical collaborative filtering (still used, still asked)

- **Content-based**: recommend items similar (in features) to what the user liked. No cold
  start for items; no serendipity; needs good item features.
- **Neighborhood CF**: user-based (find similar users) or **item-based** (find items
  co-engaged with the user's items; precomputable, scales better — Amazon's classic
  item-to-item). Similarity: cosine/Jaccard on interaction vectors, with shrinkage for
  low-support pairs.
- **Matrix factorization (MF)**: `r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤ·qᵢ` with learned user and item
  vectors; fit by SGD or **ALS**. For implicit data, **iALS** (Hu, Koren, Volinsky 2008)
  treats all unobserved as weak negatives with confidence weights; **BPR** (Rendle 2009)
  trains pairwise: `σ(r̂ᵤᵢ − r̂ᵤⱼ)` for observed `i` vs sampled unobserved `j`. MF *is* a
  two-tower model with ID-only towers.
- **Factorization Machines**: generalize MF to arbitrary features with pairwise
  interactions via low-rank embeddings — the bridge to deep ranking models.
- Netflix Prize lesson: the winning 100+ model ensemble was never fully deployed —
  engineering cost and latency outweighed the accuracy gain. Managers love this point.

## 4. Deep retrieval: the two-tower model

- **User tower** `f(user features, history)` and **item tower** `g(item features)` produce
  embeddings; score = dot product (or cosine with temperature). Trained with
  **sampled softmax / in-batch negatives**: for each (user, clicked item) pair in a batch,
  the other items in the batch are the negatives (InfoNCE — identical to CLIP/word2vec).
- **Why two towers?** Because the towers are *independent*, item embeddings can be
  **precomputed and indexed**; at request time you compute one user embedding and run an
  **ANN search**. The price: **no user×item cross features** (the towers only meet at the
  dot product) — which is exactly why a separate ranking stage with cross features exists.
- **Sampling bias**: in-batch negatives are sampled ∝ popularity, so popular items get
  pushed down too hard. **logQ correction** (Yi et al. 2019): subtract `log P(item in
  batch)` from the logit. Also: **hard negative mining** (negatives the model currently
  ranks high; use carefully — too hard = noise), mixed random + in-batch negatives.
- The **YouTube DNN** (Covington 2016) framing: retrieval as extreme multiclass
  classification over the catalog; the **"example age"** feature to fix freshness bias
  (models trained on historical logs otherwise favor old content; set it to 0 at serving).
- Hyperparameters that matter: embedding dim (64–256), temperature, batch size (more
  negatives), history length and aggregation (mean pooling → attention/transformer),
  ID hashing for huge vocabularies, **feature freshness**.

## 5. Approximate nearest neighbor search

Exact search over 10⁸ items per request is impossible at ms latency. **ANN** trades a
little recall for speed:
- **IVF** (cluster with k-means; search only the nearest `nprobe` clusters),
- **HNSW** (navigable small-world graph; the default for < 10⁸ vectors; high recall, memory
  hungry, slow to build/update),
- **Product quantization (PQ)** (compress vectors into codebook indices; `IVF-PQ` for
  billion-scale), **ScaNN** (anisotropic quantization).
- Libraries/services: FAISS, hnswlib, ScaNN, Milvus, Vespa, Elasticsearch/OpenSearch
  kNN, pgvector, Pinecone.
- Operational issues: index rebuild cadence vs new items (new items need a fallback until
  indexed), **embedding version consistency** (user tower and index must come from the
  same training run — a mismatched deploy makes every recommendation garbage while all
  dashboards look green), recall@k of the ANN vs exact (measure it), filtering (ANN + a
  filter can return too few results — pre- or post-filtering strategies).

## 6. Ranking models

- Historical: logistic regression with manual crosses; **GBDT + LR** (Facebook 2014: trees
  as feature transformers).
- **Wide & Deep** (Google 2016): a linear "wide" part memorizes crosses, a deep part
  generalizes via embeddings.
- **DeepFM**: FM for 2nd-order interactions + DNN, no manual crosses.
- **DCN / DCN-v2**: explicit bounded-degree feature crossing layers + DNN — a strong,
  cheap default.
- **DLRM** (Meta): embedding tables for sparse features, bottom MLP for dense features,
  pairwise dot interactions, top MLP. The tables are tens to hundreds of GB → sharded
  across GPUs (**model parallel for embeddings, data parallel for MLPs**); embedding
  lookups are memory-bandwidth-bound; training throughput is dominated by all-to-all
  communication of embedding rows.
- **User-history attention**: **DIN** (attend over the user's history conditioned on the
  candidate item), DIEN, **transformers over sequences** (SASRec, BERT4Rec, BST).
- **Multi-task**: shared-bottom; **MMoE** (several experts, per-task gating; handles
  conflicting tasks like click vs. finish); PLE. Multi-task is the norm (click, like,
  share, watch time, skip, dislike…).
- **ESMM**: estimate conversion over the *entire* impression space by modeling
  `pCTCVR = pCTR × pCVR`, fixing the sample selection bias of training CVR only on clicks.
- Ranking training data = **impressions with labels** (clicked / not clicked); negatives
  are real "shown but ignored" examples, which is why ranking can learn finer distinctions
  than retrieval (whose negatives are mostly random).
- **Calibration** is a hard requirement when scores are combined, thresholded, or used in
  auctions; check calibration per segment, recalibrate after any re-weighting.

## 7. Biases and feedback loops (the level-3 material)

- **Position bias**: items at the top get clicked more regardless of relevance. Fixes:
  include position as a feature in training and set it to a fixed value at inference
  (position-aware training), or inverse-propensity weighting with estimated propensities,
  or randomized exposure experiments; **PAL**-style separable models.
- **Exposure / selection bias**: the logs only contain items the *previous* model chose to
  show. Training on them reinforces the previous model (**feedback loop**), hurts new
  items, and makes offline evaluation of a very different policy unreliable. Fixes:
  exploration traffic (ε-greedy slots, Thompson sampling/contextual bandits), off-policy
  evaluation (IPS, doubly robust), randomized holdouts.
- **Popularity bias**: the long tail is under-recommended; measure coverage and
  tail-item share; logQ correction; re-rank for diversity.
- **Survivorship in labels**: users who churned stop generating data.
- **Novelty effects** in A/B tests; **cannibalization** between surfaces; **network
  effects** that violate the independence assumption of user-level randomization.

## 8. Features and point-in-time correctness

- **User**: demographics, long-term aggregates (CTR by category over 90 days), short-term
  session signals (last 20 items, real-time), learned embeddings.
- **Item**: metadata, content embeddings (text/image models), popularity and CTR stats
  with time decay, age, creator stats.
- **Context**: time, device, location, surface, position.
- **Cross**: user-category affinity × item category (ranking only).
- **Point-in-time correctness**: a training row for an event at time t must use feature
  values *as they were at t*. Using today's item CTR for an impression from last month
  leaks the future and inflates offline metrics. Feature stores exist largely to make
  point-in-time joins correct.
- **Train/serve skew**: the feature computed offline (Spark) differs from the one computed
  online (Java service) — time zones, null handling, rounding, different dedup. The robust
  pattern is **log the features at serving time and train on the logged values**.

## 9. Cold start and exploration

- **New items**: content-based embeddings instead of ID embeddings (or ID embeddings
  initialized from content), guaranteed exploration slots with a bandit, fast index
  refresh, creator-level priors.
- **New users**: onboarding questionnaire, popularity/contextual defaults, session-based
  models that work from the first few actions, demographics/device priors.
- **Bandits**: ε-greedy, UCB, **Thompson sampling**; contextual bandits with features;
  good for exploration slots, headline selection, small action spaces. Off-policy
  evaluation lets you estimate a new policy's reward from logged data with propensities.

## 10. Evaluation: offline and online

**Offline**
- Retrieval: Recall@K, Hit rate@K over *full* ranking (or be explicit about sampling bias).
- Ranking: AUC and log loss; **GAUC** (AUC averaged per user; global AUC can be dominated
  by between-user differences that don't matter for ranking within a user's slate).
- Split **by time** (train on days 1–28, validate on day 29, test on day 30). Leave-one-out
  random splits leak and overstate.
- Baselines you must beat: **popularity**, item-kNN, previous production model. A new
  model that doesn't beat most-popular by a wide margin on Recall@K is suspicious.
- Beyond accuracy: coverage, diversity, novelty, calibration, latency, fairness slices.

**Online**
- **A/B test** randomized by user; primary metric agreed in advance (e.g. sessions per
  user, watch time), **guardrail metrics** (latency, errors, complaints, revenue), enough
  power; run ≥ 1–2 weeks to cover weekly cycles and novelty decay; watch for network
  effects and multi-surface cannibalization.
- **Interleaving** (merge two rankers' lists, credit clicks) is far more sensitive than A/B
  for ranking comparisons; use it to pre-screen.
- **Long-term holdouts** (a small group kept on an old model for months) to measure
  cumulative effects and feedback-loop drift.
- The **offline–online gap** is normal: offline metrics are on logged, biased, stale data.
  Offline wins are necessary but not sufficient; the correlation is empirically checked
  (and when it breaks, fix the offline protocol).

## 11. Serving architecture (walk through a request)

1. Request arrives (user id, context). Fetch user features from the **online feature
   store** (Redis/DynamoDB/Bigtable; ms) and the precomputed or on-the-fly **user
   embedding**.
2. Query ANN indexes and other candidate sources in parallel; merge, dedup, apply hard
   filters (already seen, policy) → ~1,000 candidates.
3. Batch-fetch item features for the candidates; build feature vectors with the **same
   feature code as training**; run the ranking model (GPU or CPU batch inference; tens of ms;
   p99 matters more than p50).
4. Re-rank; apply exploration; assemble the slate.
5. **Log** impressions with position, features, candidate source, model versions; later
   join with clicks/conversions (attribution window) to form training data.
6. Fallbacks at every stage (timeouts → cached/popular results), caching of hot users,
   canary/shadow deployment of new models, automatic rollback on metric guardrails.

**Training pipeline**: daily/hourly jobs join impressions with delayed labels, compute
point-in-time features (or read logged ones), train (often **warm-started** from the
previous model, sometimes online/streaming updates for embeddings), validate against the
incumbent on the latest day, check calibration, export; the **item index** and **user
tower** are versioned *together*.

Scale anchors: catalog 10⁶–10⁹; embedding tables 10–1000 GB; 10⁴–10⁶ QPS; latency budget
~100–200 ms; ranking ~500–2000 candidates per request; daily training on billions of rows.

## 12. Emerging: LLMs in recsys (one paragraph)

LLM-generated item/user descriptions and embeddings for cold start; **semantic IDs**
(quantize content embeddings into discrete tokens; generative retrieval, e.g. TIGER);
transformer "generative recommenders" (HSTM/HSTU-style scaling of sequence models);
conversational recommendation. Mostly still research or early production; say
"promising, but the two-stage funnel with two-tower + ranking is still the backbone".

## 13. The design-interview template (use it verbatim)

1. **Clarify**: surface, objective, catalog size, QPS/latency, freshness, cold start,
   constraints, what exists today.
2. **Metrics**: online primary + guardrails; offline proxies and their known gaps.
3. **Data & labels**: events, attribution, negatives, logging, splits (temporal).
4. **Funnel**: retrieval sources → ranking → re-ranking; why each stage.
5. **Features**: user/item/context/cross; point-in-time; feature store; train/serve parity.
6. **Models per stage**: two-tower + ANN; DCN/DLRM-style multi-task ranker; calibration.
7. **Training**: cadence, warm start, negative sampling, logQ, hard negatives, validation
   against the incumbent.
8. **Serving**: request flow, latency budget, caching, fallbacks, versioning of
   embeddings/index.
9. **Experimentation**: interleaving → A/B; long-term holdout; rollout/rollback.
10. **Biases & cold start & monitoring**: position bias, exploration, drift, feedback loops.
11. **Iteration plan**: what to build first (popularity + item-kNN + LR), what to add next.

## Experience signals

- "The towers can't see cross features — that's the whole reason for a ranking stage."
- "Random splits on interaction data leak the future; I'd split by time and evaluate with
  full ranking, because sampled metrics can reorder models."
- "A new model must beat popularity by a lot, or something's wrong with the setup."
- "Position as a training feature, fixed at serving — or you learn 'top slot = good'."
- "If the user tower and the item index come from different runs, everything breaks and no
  alert fires. Version them together."
- "Offline AUC up, online down → I'd check serving features vs training features first,
  then calibration, then evaluation bias."
- "Log features at serving time; train on the logs. Train/serve skew is the silent killer."
- "Implicit feedback has no true negatives; how you sample negatives is a modeling decision."

## Follow-up chains

- "What's a two-tower model?" → "Why not use the ranker for retrieval?" → "What's wrong
  with in-batch negatives?" → "How do you serve it?" → "What breaks when you retrain?"
- "How do you evaluate a recommender?" → "Why not random split?" → "Why not AUC alone?" →
  "Offline beat the baseline but A/B is flat. Next steps?" → *check the online model is
  actually the new one (deployment), check feature parity, check power/duration, look at
  segment-level effects and novelty, consider interleaving for sensitivity.*
- "How would you handle a new item?" → "How much exploration traffic?" → "How do you
  measure its cost?" → *holdout / bandit regret.*
- "Clicks went up 5%, but 30-day retention is flat. Ship?" → *depends on the objective
  hierarchy; investigate clickbait; check satisfaction proxies; maybe ship with a long-term
  holdout.*
