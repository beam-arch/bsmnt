# 09 · ML systems and MLOps (45 minutes)

The gap between "trained a model in a notebook" and "ML engineer" is this module. Hiring
managers for MLE roles weight production awareness heavily because it is what distinguishes
an engineer from a data scientist.

## 1. Where the time actually goes

Roughly: 60–80% data (collection, labeling, cleaning, pipelines, features), 10–20%
modeling, the rest serving/monitoring. The model is the smallest part of a production ML
system (Sculley et al., "Hidden Technical Debt in ML Systems"). Saying this out loud is a
credibility signal.

## 2. Data engineering for ML

- **Batch** (Spark, dbt, SQL warehouses: BigQuery/Snowflake) vs **streaming** (Kafka,
  Flink, Spark Streaming) for real-time features. Columnar formats (**Parquet**) for
  analytics; row formats for serving.
- **Schemas and validation**: enforce types, ranges, null rates, cardinalities at
  ingestion (Great Expectations, TFDV, pandera). Most model regressions are upstream data
  changes: a renamed column, a client that started sending `-1` for missing, a timezone
  change.
- **Data versioning and lineage**: know exactly which data trained which model (DVC,
  lakeFS, Delta/Iceberg time travel, or simply immutable dated partitions).
- **Labeling**: in-house vs vendor vs programmatic (weak supervision); label quality audits;
  inter-annotator agreement; active learning to label the most informative examples.

## 3. Feature stores

A **feature store** (Feast, Tecton, Vertex/SageMaker feature stores, in-house) provides
(1) a registry of feature definitions, (2) an **offline store** for training (with
**point-in-time joins**), and (3) an **online store** (low-latency KV) for serving — both
populated by the *same* transformation code. It solves **train/serve skew** and feature
reuse across teams. Costs: infra complexity, freshness lag, TTLs, backfills.

## 4. Training infrastructure

- **Experiment tracking** (MLflow, Weights & Biases): config, code commit, data version,
  metrics, artifacts. Non-negotiable for reproducibility and for answering "why is model
  v12 better".
- **Config management** (Hydra/YAML), **seeds**, containerized environments, pinned
  dependencies.
- **Orchestration**: Airflow/Dagster/Kubeflow/Metaflow pipelines; Kubernetes or Slurm for
  GPU jobs; **checkpointing** for preemptible/spot instances; retries and idempotency.
- **Hyperparameter search** at scale (Ray Tune, Optuna) with early stopping.
- **Model registry**: versioned artifacts with metadata, stage (staging/production),
  lineage back to data and code; model cards documenting intended use and limitations.

## 5. Serving patterns

| Pattern | When | Tradeoffs |
|---|---|---|
| **Batch prediction** (precompute nightly into a table) | predictions don't need fresh inputs; e.g. churn scores, email recs | cheap, simple; stale; can't react to session |
| **Online inference** (request → model → response) | needs current context; search, ranking, fraud | latency/cost engineering; needs online features |
| **Streaming** (score events as they arrive) | anomaly detection, real-time features | complexity |
| **Embedded / edge** | mobile, privacy, offline | model size, update cadence |

- Model servers: TorchServe, Triton Inference Server (multi-framework, dynamic batching,
  GPU sharing), TF Serving, BentoML, KServe/Seldon on Kubernetes, Ray Serve; vLLM/TGI for
  LLMs. REST vs gRPC; **dynamic batching** to use GPUs efficiently; autoscaling on queue
  depth; warm pools (cold starts of GPU containers take minutes).
- **Latency budget** decomposition: network + feature fetch + preprocessing + inference +
  post-processing. Measure **p50/p95/p99**; tail latency is driven by GC pauses, batch
  stragglers, cache misses, long inputs. Techniques: caching, precomputation, smaller/
  distilled models, quantization, compiled runtimes (TensorRT/ONNX Runtime), request
  hedging, timeouts with fallbacks.
- **CPU vs GPU inference**: GPUs win at high throughput with batching; CPUs win for low
  QPS, small models, and simpler ops. Cost per 1k predictions is the deciding number.

## 6. Deployment strategies

**Shadow** (new model runs on live traffic, outputs logged, not shown) → **canary**
(1–5% of traffic with guardrails and auto-rollback) → **A/B** (measure business metric) →
full rollout; **blue/green** for instant switch-back. Keep the previous model hot for
rollback. Feature flags for model selection. Every model deploy is also a *data contract*
deploy: input schema and feature versions must match.

## 7. Monitoring (what pages you at 3 am)

- **System**: latency, error rate, throughput, GPU/CPU utilization, memory, queue depth.
- **Data quality**: schema violations, null rates, value ranges, cardinality,
  feature-freshness lag, volume anomalies ("the clicks table has 90% fewer rows today").
- **Drift**: input distribution shift (PSI, KL, KS tests per feature; embedding drift),
  prediction distribution shift (mean score moved from 0.12 to 0.30 → something upstream
  changed), label drift, concept drift (relationship changed; visible only when labels
  arrive).
- **Model performance**: when ground truth arrives late (conversions, fraud chargebacks),
  monitor proxies now and true metrics when available; slice by segment (country, device,
  new vs returning).
- **Alerting on silent failures**: a model that returns the *same* score for everyone, a
  feature that became constant, an embedding index that stopped updating. These don't
  throw exceptions; they need statistical alerts.
- **Retraining triggers**: scheduled (daily/weekly), drift-triggered, performance-triggered;
  always validated against the incumbent before promotion ("champion/challenger").

## 8. Testing ML systems

- Unit tests for feature transforms and data loaders; property tests (output shape,
  no NaNs); **contract tests** between pipeline stages.
- Model tests: overfit-one-batch smoke test, loss-at-init test, invariance/directional
  expectation tests ("adding 'excellent' should not decrease the sentiment score"), minimum
  functionality tests on a golden set, performance thresholds as CI gates.
- Integration tests on a tiny end-to-end run in CI; data validation gates before training;
  evaluation gates before deployment; shadow comparisons after.

## 9. Reliability patterns

Idempotent, re-runnable pipelines; backfills; handling late-arriving data; schema
evolution; dead-letter queues; defaulting strategies when a feature is unavailable (and
training with the same defaults — otherwise serving sees values the model never saw);
graceful degradation (fallback to a simpler model or cached results); circuit breakers;
capacity planning for peak (Black Friday); runbooks and on-call for ML services;
blameless postmortems.

## 10. Cost

GPU-hours (training and serving), storage/egress, labeling, people. Levers: right-size
models, batch/async where latency allows, quantize/distill, cache, spot instances with
checkpointing, reduce retraining frequency when drift is slow, delete unused features
(every feature has a pipeline cost). Always be able to say the **cost per prediction** and
the **business value per model improvement**.

## 11. Responsible AI (what to say briefly and correctly)

Fairness (demographic parity vs equalized odds — you can't satisfy all definitions at
once; slice metrics by protected groups), **privacy** (PII handling, minimization, retention,
deletion requests → retraining implications, differential privacy and federated learning
as tools), **explainability** (SHAP for tabular; "right to explanation" in credit),
**safety** for LLMs (prompt injection, jailbreaks, toxic outputs, guardrail models),
**documentation** (model cards, datasheets).

## 12. How MLEs work with others

Data scientists prototype and own metrics; data engineers own pipelines; MLEs own
productionization, training infrastructure, serving, and often the models at scale; PMs
own the objective; SRE owns the platform. Hand-offs fail at: undocumented notebooks,
features that can't be computed online, metrics nobody can reproduce. The MLE's job is to
make the whole loop reliable and repeatable.

## 13. Production incident catalog (level-3 answers in disguise)

1. **Silent feature outage**: an upstream job failed; the feature store served defaults
   (0); the model kept serving, CTR dropped 8%; no error alert. *Fix*: data-freshness and
   prediction-distribution alerts; feature-level null-rate monitoring; fallbacks that
   degrade gracefully.
2. **Train/serve skew**: offline feature "days since last purchase" used event time; online
   used request time with a different timezone → off by one day everywhere. *Fix*: log
   features at serving time and train on them; parity tests comparing offline vs online
   feature values for the same entity.
3. **Leakage via a "last_updated" column** → offline AUC 0.98, online useless. *Fix*:
   feature availability review; point-in-time joins; "too good" alarm.
4. **Embedding version mismatch**: user tower updated, item index stale → recommendations
   became random; dashboards green. *Fix*: atomic versioned deploys; a canary that
   checks recall@k against a golden set after every deploy.
5. **Retrained on a bad day**: a logging bug dropped 40% of positive labels for one day;
   the daily retrain learned a lower CTR and the ranker shifted toward popular items.
   *Fix*: data validation gates (label rate within tolerance), champion/challenger
   promotion with offline *and* canary checks.
6. **Dataloader bottleneck**: 8 GPUs at 20% utilization because JPEG decoding was on 8 CPU
   cores. *Fix*: profile first; pre-resize to the training resolution; more workers; GPU
   decoding.
7. **OOM at p99**: a rare 10k-token input blew up inference memory at 3 am. *Fix*: input
   length caps, length-based batching, memory headroom, load tests with adversarial inputs.
8. **Non-reproducible result**: a 0.5% win vanished on rerun — unseeded data order plus
   `cudnn.benchmark`. *Fix*: seeds, deterministic flags for eval, multiple seeds before
   claiming wins.
9. **Sampling bug**: negatives sampled with a fixed seed every epoch → the same negatives
   each epoch; model overfit them. *Fix*: reseed per epoch; test the sampler.
10. **Metric computed on sampled negatives** → claimed Recall@10 of 0.6; full ranking
    gave 0.08 and a different model ordering. *Fix*: full-ranking eval or explicit caveats.
11. **Tokenizer drift**: serving used a tokenizer with an extra special token → inputs
    shifted by one ID. *Fix*: bundle tokenizer with model artifact; hash-check at load.
12. **Integer overflow in a count feature** after 2.1B events → negative values → tree
    model routed every row to one branch. *Fix*: int64; range validation.

## Experience signals

- "I'd log the served features and train on those — that kills the biggest class of bugs."
- "Models fail silently; I alert on prediction distributions and feature null-rates, not
  just on exceptions."
- "Champion/challenger: a retrained model must beat the incumbent on the latest data
  *and* pass a canary before promotion."
- "I estimate cost per prediction before choosing GPU vs CPU serving."
- "Shadow, then canary, then A/B; keep the old model hot for rollback."

## Follow-up chains

- "How would you deploy a new ranking model?" → "What could go wrong on day 1?" → "How
  would you know?" → "How do you roll back?"
- "Model performance degraded over three months. Why?" → *concept drift, feature drift,
  feedback loop, upstream data change, seasonal shift, competitor change; diagnosis order:
  data quality → feature drift → label drift → retrain → investigate concept change.*
- "What's the difference between data drift and concept drift?" → "How do you detect
  concept drift without labels?" → *you mostly can't; use proxies, delayed labels, and
  canary cohorts with faster labels.*
- "Batch vs online inference for X?" → always answer with freshness need, latency budget,
  QPS, and cost.
