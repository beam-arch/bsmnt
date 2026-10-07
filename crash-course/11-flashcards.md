# 11 · Flashcards (30 minutes) — cover the right column, answer aloud

## Fundamentals

| Q | A |
|---|---|
| What is ML in one sentence? | Fit a function to data by minimizing a loss so it generalizes to new data |
| Supervised vs unsupervised vs self-supervised? | Labels given / no labels / labels made from the input itself (next token) |
| Why a validation set and a test set? | Validation for decisions and tuning; test for one final unbiased estimate |
| When is a random split wrong? | Time-dependent data, grouped entities (users), duplicates |
| Bias vs variance? | Underfit systematic error vs overfit sensitivity to the sample |
| Three fixes for overfitting? | More data, regularization/early stopping, simpler model or ensembles |
| Three fixes for underfitting? | Bigger model, better features, train longer / higher LR / less regularization |
| L1 vs L2? | L1 sparsity (Laplace prior), L2 shrinkage (Gaussian prior) |
| Loss vs metric? | Loss is the differentiable surrogate you optimize; metric is what you care about |
| MSE corresponds to which noise assumption? | Gaussian (MLE) |
| Cross-entropy is the NLL of which distribution? | Categorical / Bernoulli |
| Gradient of softmax-CE w.r.t. logits? | p − y |
| Accuracy fails when? | Class imbalance |
| Precision? | TP / (TP + FP) — of flagged, how many correct |
| Recall? | TP / (TP + FN) — of actual positives, how many caught |
| F1? | Harmonic mean of precision and recall |
| ROC-AUC meaning? | P(random positive scores higher than random negative) |
| When is ROC-AUC misleading? | Heavy imbalance — use PR-AUC |
| Which metric sees calibration? | Log loss (and ECE / reliability diagrams) |
| Calibration fixes? | Platt scaling, isotonic regression, temperature scaling |
| NDCG? | Sum of rel/log2(rank+1) normalized by the ideal ordering |
| MRR? | Mean of 1/rank of the first relevant item |
| Recall@K? | Fraction of relevant items in the top K |
| Sampled-negative evaluation problem? | Inflates metrics and can reorder models (Krichene & Rendle) |
| Target leakage? | A feature that is a consequence of the label |
| Point-in-time correctness? | Features computed as they were at the event time, not later |
| Preprocessing leakage? | Fitting scaler/encoder on all data before splitting |
| First reaction to AUC 0.99? | Suspect leakage |
| Target encoding safety? | Out-of-fold + smoothing |
| Trees need feature scaling? | No |
| Missing values best practice? | Impute + missing indicator; GBDTs handle NaN natively |
| Default model for tabular? | Gradient boosted trees |
| LightGBM speed tricks? | Leaf-wise growth, histogram binning |
| CatBoost's categorical trick? | Ordered target statistics |
| Random forest's free validation? | Out-of-bag error |
| k-means seeding? | k-means++ |
| PCA computes? | Eigenvectors of the covariance / SVD of centered data |
| Random vs grid search? | Random wins; Bayesian (Optuna) better for expensive models |
| Sanity checks before training long? | Overfit one batch; initial loss = ln(C); shuffled labels can't be learned |
| Significant improvement rule? | Multiple seeds; confidence intervals; a single-seed 0.3% is noise |

## Deep learning

| Q | A |
|---|---|
| Why nonlinear activations? | Stacked linear layers collapse to one linear map |
| Dead ReLU? | Unit's pre-activation always negative → zero gradient forever |
| GELU used where? | Transformers |
| Backprop is? | Reverse-mode autodiff via the chain rule, reusing stored activations |
| Backward cost vs forward? | ~2× |
| What dominates training memory? | Activations (then optimizer states) |
| Activation checkpointing? | Store fewer activations, recompute in backward (~30% more compute) |
| Softmax before CrossEntropyLoss? | Never — it applies log-softmax internally |
| Binary output + loss? | Single logit + BCEWithLogits |
| Mini-batch SGD noise is? | Regularization and escape from saddles |
| Linear scaling rule? | Batch ×k → LR ×k, with warmup |
| Momentum? | EMA of gradients; damps oscillation |
| Adam? | Momentum + per-parameter RMS scaling with bias correction |
| AdamW vs Adam+L2? | Decay decoupled from the adaptive step |
| Don't weight-decay what? | Biases and normalization parameters |
| Most important hyperparameter? | Learning rate |
| Typical transformer schedule? | Linear warmup, cosine decay, grad clip 1.0 |
| Gradient accumulation? | Sum grads over k micro-batches before stepping |
| Zero init problem? | Symmetry — all units identical |
| Kaiming init variance? | 2 / fan_in (ReLU) |
| Xavier init variance? | 2 / (fan_in + fan_out) (tanh/sigmoid) |
| BatchNorm train vs eval? | Batch stats vs running stats |
| BatchNorm fails when? | Tiny batches, sequences, per-GPU stats in DDP |
| LayerNorm normalizes over? | Features of each example |
| RMSNorm? | LayerNorm without mean-centering (LLaMA) |
| Pre-LN vs post-LN? | Pre-LN more stable, less warmup needed |
| Dropout at inference? | Off; training scaled by 1/(1−p) |
| Label smoothing does? | Soft targets; better calibration/robustness |
| Residual connection formula? | y = x + F(x) |
| Conv output size? | floor((n + 2p − k)/s) + 1 |
| Conv params? | k·k·C_in·C_out + C_out |
| Why CNNs for images? | Locality, weight sharing, translation equivariance |
| Why LSTMs over RNNs? | Gates keep gradients flowing across time |
| Why transformers over LSTMs? | Parallel over sequence, direct long-range paths, scale |
| Attention formula? | softmax(QKᵀ/√d_k)V |
| Why √d_k? | Keeps logit variance ~1; avoids softmax saturation |
| Causal mask? | Token t can't attend to t+1… |
| Multi-head purpose? | Different relations in different subspaces |
| Attention complexity? | O(n²d) time; naive O(n²) memory |
| Params per transformer block? | ≈ 12·d² |
| Positional encodings in modern LLMs? | RoPE (rotary) |
| KV cache per token? | 2 × layers × kv_heads × head_dim × bytes |
| GQA/MQA? | Share K/V heads to shrink the KV cache |
| FlashAttention is approximate? | No — exact, IO-aware tiling with online softmax |
| word2vec ≈ ? | A two-tower model with sampled negatives |
| Contrastive loss name? | InfoNCE |
| Diffusion models learn? | To denoise progressively noised data |
| Loss spikes in LLM training — reflex? | Roll back to checkpoint, skip batch, lower LR/clip, check fp16 |

## PyTorch

| Q | A |
|---|---|
| Training step order? | zero_grad → forward → loss → backward → clip → step → scheduler |
| Why zero_grad? | `.grad` accumulates by design |
| `model.eval()` vs `no_grad()`? | Mode flags vs no graph recording; use both |
| Modules in a Python list? | Not registered — use nn.ModuleList |
| CrossEntropyLoss inputs? | Raw logits (N,C) + int64 class indices (N,) |
| MSE shape bug? | (N,1) vs (N,) broadcasts to (N,N) |
| `.item()` cost? | Host–device sync |
| Storing `loss` in a list? | Memory leak (keeps graphs) |
| `view` vs `reshape`? | view needs contiguous memory; reshape copies if needed |
| `pin_memory`? | Page-locked host memory → async H2D copies |
| `num_workers`? | Subprocesses for `__getitem__`; processes because of the GIL |
| Worker RNG gotcha? | Forked workers share generator state → identical augmentations |
| Save what in a checkpoint? | state_dicts of model, optimizer, scheduler + epoch + RNG |
| `torch.load` safety? | `weights_only=True`; pickle executes code |
| bf16 needs GradScaler? | No; fp16 does |
| TF32? | fp32-range matmul with 10-bit mantissa on Tensor Cores |
| `torch.compile` risks? | Compile time, recompiles on dynamic shapes, graph breaks |
| DDP does what in backward? | All-reduces gradient buckets, overlapped with compute |
| FSDP shards? | Params, grads, optimizer states |
| DistributedSampler per epoch? | `set_epoch(epoch)` |
| `cudnn.benchmark`? | Autotune conv algos; faster, nondeterministic |
| Device-side assert usually means? | Index out of range (labels, embeddings) |

## GPUs

| Q | A |
|---|---|
| CPU vs GPU design goal? | Latency vs throughput |
| Warp? | 32 threads executing in lockstep (SIMT) |
| SM? | Streaming multiprocessor; 132 on H100 |
| Tensor Cores do? | Small matrix multiply-accumulate in low precision |
| Memory hierarchy order? | Registers → shared/L1 → L2 → HBM → NVLink → PCIe → network |
| H100 HBM bandwidth? | ~3.35 TB/s (80 GB) |
| Arithmetic intensity? | FLOPs per byte moved |
| Ridge point H100? | ~295 FLOP/byte |
| Matmul regime? | Compute-bound (large sizes) |
| LayerNorm/softmax/elementwise regime? | Memory-bound |
| LLM decode regime? | Memory-bound (re-reads weights per token) |
| Fix for memory-bound ops? | Kernel fusion, lower precision |
| Fix for launch-bound? | Bigger batches, fusion, CUDA graphs |
| fp16 max value? | 65,504 |
| bf16 vs fp16? | bf16 has fp32's range with 7 mantissa bits |
| Why fp32 master weights? | Tiny updates underflow in low precision |
| Adam mixed-precision bytes/param? | 16 |
| 7B training memory before activations? | ~112 GB |
| 7B inference bf16 weights? | 14 GB |
| KV cache LLaMA-2-7B per token? | 0.5 MB |
| nvidia-smi 100% util means? | Some kernel was running; not SM occupancy |
| MFU? | Achieved FLOP/s ÷ peak; 40–55% is good for LLMs |
| Ring all-reduce cost? | 2(n−1)/n × bytes per GPU |
| Tensor parallel where? | Inside a node (NVLink) |
| Prefill vs decode? | Compute-bound TTFT vs memory-bound tokens/s |
| PagedAttention? | KV cache in blocks; no fragmentation; more concurrency |
| Speculative decoding? | Draft proposes k tokens, target verifies in one pass; exact |
| Int4 weight quantization helps because? | Decode is memory-bound |
| Coalesced access? | Adjacent threads read adjacent addresses |
| Triton? | Python DSL for block-level GPU kernels |
| Driver vs runtime error? | Driver must be ≥ the bundled CUDA runtime |

## Recsys

| Q | A |
|---|---|
| Funnel stages? | Retrieval → pre-rank → ranking → re-ranking |
| Retrieval requirement? | Cheap and high recall over millions |
| Why ranking separately? | Cross features and heavy models on hundreds of candidates |
| Two-tower score? | Dot product of independent user and item embeddings |
| Two-tower limitation? | No user×item cross features |
| In-batch negatives bias? | Popularity → logQ correction |
| Implicit feedback negatives? | Don't exist; sampled — a modeling decision |
| MF prediction formula? | μ + b_u + b_i + p_u·q_i |
| BPR? | Pairwise: positive should outscore sampled negative |
| iALS? | Implicit ALS with confidence weights |
| ANN methods? | HNSW, IVF, PQ, ScaNN |
| HNSW tradeoff? | High recall/speed; memory heavy; slow updates |
| Ranking model families? | Wide&Deep, DeepFM, DCN, DLRM, DIN, MMoE |
| DLRM memory hog? | Embedding tables; sharded (model parallel) |
| MMoE? | Experts with per-task gates for multi-task |
| ESMM? | pCTCVR = pCTR × pCVR to fix CVR selection bias |
| Position bias fix? | Position feature in training, fixed at serving; or IPS |
| Exposure bias? | Logs only contain what the old model showed |
| Feedback loop fix? | Exploration traffic, off-policy eval, holdouts |
| Example age (YouTube)? | Feature to correct freshness bias; set 0 at serving |
| Train/serve skew fix? | Log served features, train on them |
| Temporal split why? | Random split leaks the future |
| GAUC? | AUC averaged per user |
| Must-beat baselines? | Popularity, item-kNN, incumbent |
| Interleaving? | Merge rankers' lists; much more sensitive than A/B |
| Long-term holdout? | Small cohort on old model for months |
| Cold-start items? | Content embeddings, exploration slots, fast indexing |
| Cold-start users? | Onboarding, context priors, session models, bandits |
| Thompson sampling? | Sample from posterior, act greedily — exploration bandit |
| Index/tower version rule? | Deploy together, atomically |
| Value model? | Weighted combination of predicted engagement probabilities |
| Delayed labels handling? | Attribution window; join later |
| Netflix Prize lesson? | Winning ensemble too costly to deploy |

## MLOps / systems

| Q | A |
|---|---|
| Where does time go? | 60–80% data |
| Feature store parts? | Registry, offline store (point-in-time), online KV store |
| Deploy sequence? | Shadow → canary → A/B → full; old model hot |
| Data drift vs concept drift? | Inputs change vs relationship changes |
| Drift metric? | PSI / KL / KS per feature; prediction distribution |
| Silent failure examples? | Constant feature, same score for all, stale index |
| Champion/challenger? | New model must beat incumbent on fresh data + canary |
| Batch vs online? | Freshness, latency, QPS, cost |
| p99? | Tail latency — what users notice |
| Cost unit? | Cost per prediction / per 1k requests |
| Model artifact must include? | Weights + tokenizer/feature spec + versions |
| Spot instances need? | Checkpointing |
| Experiment tracking records? | Config, commit, data version, metrics, artifacts |
| ML test that catches most bugs? | Overfit one batch in CI on tiny data |
| Hidden technical debt paper? | Sculley et al. — the model is the smallest box |

## LLMs

| Q | A |
|---|---|
| Training FLOPs? | 6 × params × tokens |
| Inference FLOPs per token? | 2 × params |
| Chinchilla ratio? | ~20 tokens per parameter (compute-optimal) |
| Why over-train? | Inference cost scales with params |
| Tokenizer types? | BPE, WordPiece, SentencePiece/Unigram |
| Post-training stages? | SFT → preference optimization (RLHF/DPO) |
| RLHF components? | Reward model + PPO + KL penalty |
| DPO? | Direct preference loss; no reward model/RL |
| LoRA? | ΔW = BA low-rank adapters on frozen weights; mergeable |
| QLoRA? | 4-bit base + LoRA |
| RAG pipeline? | Chunk → embed → index → retrieve (hybrid) → rerank → generate with citations |
| RAG vs fine-tune? | Knowledge vs behavior |
| Temperature? | Scales logits before softmax |
| Top-p? | Sample from smallest set with cumulative prob ≥ p |
| MoE? | Router picks top-k expert FFNs per token; many params, few active |
| Benchmark caveat? | Contamination |
| LLM-as-judge biases? | Length, position, self-preference |
| Perplexity? | exp(mean cross-entropy per token) |

## Numbers to have ready

| Number | Value |
|---|---|
| fp16 max | 65,504 |
| Adam mixed-precision bytes/param | 16 |
| H100 bf16 dense TFLOPS / HBM | ~989 / 3.35 TB/s, 80 GB |
| A100 bf16 TFLOPS / HBM | 312 / 2.0 TB/s (80 GB) |
| Ridge point H100 | ~295 FLOP/byte |
| Training FLOPs | 6ND |
| KV cache 7B per token | ~0.5 MB |
| Loss at init (C classes) | ln C (2.30 for 10) |
| Transformer block params | 12d² |
| Tokens per word (English) | ~1.3 |
| Typical recsys latency budget | 100–200 ms end-to-end |
| Good LLM training MFU | 40–55% |
| Warp size | 32 |
