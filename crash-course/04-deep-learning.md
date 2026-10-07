# 04 · Deep learning (90 minutes)

Run `code/02_numpy_mlp_from_scratch.py` with this module: it trains a 2-layer network with
hand-written backprop and checks the gradients numerically. If you understand that file you
understand what every framework does under the hood.

## 1. Why deep learning

Classical ML needs humans to design features. Deep nets **learn the features** from raw
data (pixels, tokens, audio), and their accuracy keeps improving with more data and compute
("scaling"). Costs: data-hungry, compute-hungry, harder to debug, poorly calibrated, and on
tabular data usually not better than GBDTs.

## 2. The building block

A **neuron** computes `a = σ(w·x + b)`. A **layer** computes `A = σ(XW + b)` for a whole
batch at once (`X: N×D_in`, `W: D_in×D_out`). An **MLP** (multi-layer perceptron) stacks
layers. Without the nonlinearity `σ`, stacked layers collapse into one linear map — the
nonlinearity is what gives expressive power.

**Activations**
- **Sigmoid** `1/(1+e⁻ˣ)`: outputs (0,1); saturates → tiny gradients (vanishing); only used
  for output probabilities/gates now.
- **Tanh**: zero-centered sigmoid; same saturation issue; used in RNNs/LSTMs.
- **ReLU** `max(0,x)`: cheap, no saturation for positives, sparse; "dead ReLU" when a unit's
  pre-activation is always negative (gradient 0 forever). The default for CNNs/MLPs.
- **Leaky ReLU / ELU**: small negative slope to avoid dead units.
- **GELU / SiLU (Swish)**: smooth ReLU-like; the transformer default (BERT, GPT, LLaMA uses
  SwiGLU).
- **Softmax**: turns a vector of logits into a probability distribution; used at the
  output and inside attention.

## 3. Forward, loss, backward

1. **Forward pass**: compute predictions layer by layer; store intermediate activations.
2. **Loss**: scalar measuring error (cross-entropy, MSE).
3. **Backward pass (backpropagation)**: chain rule from the loss backwards; each layer
   receives `dL/d(output)`, computes `dL/d(weights)` and `dL/d(input)`, passes the latter
   down. Cost ≈ 2× forward. Needs stored activations → **memory ∝ batch × depth × width**.
4. **Update**: optimizer changes weights using gradients.

For a 2-layer net with ReLU and softmax cross-entropy (as in the code):
```
z1 = X W1 + b1;  h = relu(z1);  logits = h W2 + b2;  p = softmax(logits)
L = -mean(log p[y])
dlogits = (p - onehot(y)) / N
dW2 = hᵀ dlogits;  db2 = sum(dlogits);  dh = dlogits W2ᵀ
dz1 = dh * (z1 > 0)
dW1 = Xᵀ dz1;  db1 = sum(dz1)
```
**Gradient checking**: compare analytic gradients with finite differences
`(L(w+ε) − L(w−ε)) / 2ε`. Standard practice when writing custom layers.

**Output/loss pairing** (classic bug source): regression → linear output + MSE; binary →
single logit + `BCEWithLogitsLoss`; multiclass → C logits + `CrossEntropyLoss` (which applies
log-softmax itself; **never softmax before it**).

## 4. Optimization

- **(Mini-batch) SGD**: gradient on a batch of B examples; noisy but cheap; the noise acts
  as regularization. **Batch size** trades off: larger = better GPU utilization and less
  noisy gradients, but fewer updates per epoch and often worse generalization unless the LR
  is scaled (**linear scaling rule**: multiply LR by k when batch grows by k, with warmup).
- **Momentum**: exponential moving average of gradients; damps oscillation, accelerates
  along consistent directions. `β = 0.9`.
- **RMSProp / Adagrad**: divide by a running RMS of gradients → per-parameter learning
  rates (good for sparse features like embeddings).
- **Adam** = momentum + RMSProp with bias correction. Defaults `β1=0.9, β2=0.999, ε=1e-8`,
  LR ~ `1e-4` to `3e-4` for transformers, `1e-3` for small nets. Robust; the default for
  transformers, recsys, anything with embeddings.
- **AdamW**: Adam with **decoupled weight decay** (decay applied directly to weights rather
  than added to the gradient). In plain Adam, the L2 term gets divided by the adaptive
  denominator, so parameters with large historical gradients barely get decayed — AdamW
  fixes this. Use AdamW; typical `weight_decay=0.01–0.1`; **don't decay biases and
  normalization parameters**.
- **SGD + momentum** still often generalizes better for CNNs on images (ResNets); Adam wins
  for transformers and sparse problems.
- **Learning rate** is the most important hyperparameter. Too high: divergence / NaN / loss
  bouncing. Too low: slow, stuck. Find it with an LR range test; tune it in log space.
- **Schedules**: **warmup** (linear ramp over the first few hundred/thousand steps —
  protects against bad early Adam statistics and unstable early transformer training), then
  **cosine decay** (LLMs, vision) or **step decay** or **reduce-on-plateau**; **one-cycle**
  for fast convergence.
- **Gradient clipping** (by global norm, e.g. 1.0): prevents exploding gradients; standard
  for transformers and RNNs.
- **Gradient accumulation**: sum gradients over k micro-batches before stepping to emulate
  a k× larger batch when memory is tight. (Divide the loss by k, or average appropriately.)
- **Second-order / others**: L-BFGS (small problems), Shampoo/Muon (recent; large-scale
  training), LAMB/LARS (huge batches).

## 5. Initialization

- Zeros: every unit computes the same thing forever (symmetry). Must break symmetry randomly.
- Scale matters: too small → activations/gradients vanish with depth; too large → explode.
- **Xavier/Glorot** (`Var = 2/(fan_in+fan_out)`) for tanh/sigmoid; **Kaiming/He**
  (`Var = 2/fan_in`) for ReLU. PyTorch's `nn.Linear` default is a Kaiming-uniform variant.
- Modern transformers rely on normalization + residuals so init is less fragile, but output
  projections are often scaled down by `1/√(2·layers)` (GPT-2 trick) for stability.

## 6. Normalization layers

- **BatchNorm**: normalize each feature across the batch (`(x−μ_B)/σ_B`), then learn scale
  `γ` and shift `β`. Stabilizes and accelerates CNN training; mild regularizer. Keeps
  **running mean/var** for inference → behaves differently in `train()` vs `eval()` mode
  (a classic bug when you forget `model.eval()`). Fails with tiny batches; batch
  statistics differ per GPU in DDP (use `SyncBatchNorm` if batch-per-GPU is small);
  awkward for sequences.
- **LayerNorm**: normalize across the features of each example independently; no batch
  dependence; the transformer standard. **RMSNorm** drops the mean-centering (LLaMA, faster).
- **GroupNorm / InstanceNorm**: for small-batch vision / style transfer.
- **Pre-LN vs Post-LN** transformers: putting LayerNorm *before* the sublayer (pre-LN) is
  far more stable and needs less warmup; the original 2017 paper used post-LN.

## 7. Regularization in deep nets

- **Dropout**: randomly zero units with prob `p` during training (and scale by `1/(1−p)`);
  identity at eval. Forces redundancy. Less used in big CNNs (BatchNorm) and modern LLMs
  (they're data-rich); still common in small MLPs, recsys ranking models, fine-tuning.
- **Weight decay**, **early stopping**, **label smoothing** (targets `1−ε` / `ε/(C−1)` — improves
  calibration and robustness), **data augmentation** (flips/crops/color jitter; mixup/cutmix
  in vision; token masking/dropout, back-translation in NLP), **stochastic depth**.

## 8. Residual connections and the depth problem

Plain deep nets get *worse* with depth (degradation — optimization, not overfitting).
**Residual connection** `y = x + F(x)` lets each block learn a *correction*; gradients flow
through the identity path unchanged (a "gradient highway"); enables 100+ layer nets
(ResNet, 2015) and is in every transformer block. Combined with normalization this is the
main cure for vanishing gradients; gradient clipping and gating (LSTM) are the others.

## 9. Architectures by data type

### CNNs (images, and anything with local structure)
- **Convolution**: slide a small kernel (e.g. 3×3×C_in) over the input, computing dot
  products → a feature map per output channel. Properties: *local connectivity*, *weight
  sharing* (same kernel everywhere → translation equivariance), few parameters.
- Output size `⌊(n + 2·pad − k)/stride⌋ + 1`. Parameters per conv layer
  `k·k·C_in·C_out + C_out`. Stacking layers grows the **receptive field**.
- **Pooling** (max/avg) downsamples; **stride-2 convs** do the same with learned filters;
  **1×1 convs** mix channels cheaply; **depthwise-separable** convs (MobileNet) cut FLOPs ~9×.
- Lineage: LeNet → AlexNet (2012, GPUs + ReLU + dropout) → VGG (3×3 everywhere) → ResNet
  (residuals, BatchNorm) → EfficientNet (compound scaling) → ConvNeXt. Today vision
  transformers (ViT) compete; CNNs still win on small data and for latency.
- **Transfer learning**: start from ImageNet weights; fine-tune all layers with a small LR,
  or freeze the backbone and train a new head. Default for any small vision dataset.

### RNNs (sequences; mostly historical now, but asked)
- Process tokens one at a time with a hidden state `h_t = f(h_{t-1}, x_t)`; trained with
  backprop through time. Problem: gradients vanish/explode across many steps.
- **LSTM/GRU** add gates (input/forget/output) that let gradients flow; this was the
  state of the art for text/speech 2014–2017.
- Replaced by transformers because RNNs are *sequential* (can't parallelize across the
  sequence on a GPU) and struggle with long-range dependencies. Still used in small
  on-device models, some time series, and as the inspiration for modern linear-attention /
  state-space models (Mamba).

### Transformers (the current universal architecture)
- Input: sequence of token embeddings + **positional information** (learned, sinusoidal,
  or **RoPE** rotary embeddings in modern LLMs; **ALiBi** biases). Attention itself is
  permutation-invariant, so positions must be injected.
- **Scaled dot-product attention**: `Attention(Q,K,V) = softmax(QKᵀ/√d_k) V`. Every token
  builds a query, compares it with every token's key, and takes a weighted average of
  their values. Scaling by `√d_k` keeps the logits' variance ~1 so softmax doesn't
  saturate.
- **Masks**: *causal* (token t can't see t+1…; decoder/LLM), *padding* (ignore pad tokens).
- **Multi-head**: run h attentions in parallel on `d/h`-dimensional subspaces, concatenate,
  project. Different heads learn different relations (syntax, coreference, copying).
- **Block** = `x = x + Attn(LN(x)); x = x + FFN(LN(x))`, FFN = two linear layers with a
  4× expansion and GELU/SwiGLU. Parameters per block ≈ `12·d²` (4d² attention, 8d² FFN).
  FFN holds most parameters; attention holds most *memory* at long context.
- **Cost**: attention is `O(n²·d)` in time and (naively) `O(n²)` memory per head; FFN is
  `O(n·d²)`. At typical sizes FFN dominates FLOPs until sequences get long.
- **Encoder** (bidirectional; BERT: classification, embeddings, retrieval), **decoder**
  (causal; GPT/LLaMA: generation), **encoder–decoder** (T5, Whisper: translation, ASR).
- **KV cache**: at inference, cache each layer's K and V for past tokens so generating a
  new token costs `O(n)` instead of recomputing `O(n²)`. Memory per token = `2 × layers ×
  kv_heads × head_dim × bytes`. **GQA/MQA** share K/V across heads to shrink it.
- **FlashAttention**: computes exact attention block-by-block in on-chip SRAM so the n×n
  matrix never hits GPU memory → 2–4× faster, memory linear in n (module 06).
- Why transformers won: fully parallel over the sequence, direct paths between any two
  tokens, and they scale predictably with data/compute.

### Embeddings
- An **embedding** is a learned lookup table: ID → dense vector. Tokens, users, items,
  categories. Trained by backprop like any weight; similar things end up close.
- **word2vec** (skip-gram with negative sampling) is literally a two-tower model:
  `P(context|word) ∝ exp(u_word · v_context)` with sampled negatives — the same math as
  recsys retrieval.
- Embedding tables are **sparse**: only the rows in the batch get gradients. Adam's sparse
  handling / Adagrad / row-wise Adagrad matter. In recsys the tables are the model
  (hundreds of GB), sharded across GPUs, and looked up with memory-bound gathers.
- Dimension: 16–64 for recsys IDs, 768–4096 for language models; too large overfits rare IDs.

### Generative & self-supervised (one paragraph each)
- **Autoencoder**: compress then reconstruct; bottleneck learns representations. **VAE**:
  probabilistic latent + KL regularizer; blurry samples. **GAN**: generator vs discriminator;
  sharp samples, unstable training, mode collapse. **Diffusion**: learn to denoise
  progressively noised data; the current image/audio/video standard (Stable Diffusion runs
  in a VAE latent space; classifier-free guidance trades diversity for prompt adherence).
- **Contrastive learning** (SimCLR, CLIP): pull positive pairs together, push others apart
  with InfoNCE (= softmax over in-batch negatives — again the same math as two-tower
  retrieval). **Masked modeling** (BERT, MAE). **Next-token prediction** (GPT).

## 10. The training recipe (what experienced people actually do)

1. Look at the data. Really look. Plot it, read samples, check label noise.
2. Build the end-to-end skeleton with a tiny model; fix seeds; **verify the loss at init**
   (`ln(C)` for classification); **overfit a single batch to zero loss** — if you can't,
   there's a bug.
3. Baseline (linear / small MLP / pretrained model fine-tune).
4. Scale up until it overfits, then regularize (augmentation, dropout, weight decay, early
   stopping) until the validation metric is best.
5. Tune LR first, then batch size/schedule, then the rest; use early stopping; log every run.
6. Ensembles / SWA / EMA of weights for the last few percent.

## 11. Debugging deep learning

| Symptom | Likely causes |
|---|---|
| Loss flat from step 0 | LR too low, dead ReLUs, wrong loss/output pairing, labels all one class, data not shuffled, forgot `optimizer.step()` |
| Loss decreases then explodes / NaN | LR too high, no gradient clipping, fp16 overflow (use bf16 or loss scaling), `log(0)`, division by zero in a norm, bad data (inf in input) |
| Train great, val bad | Overfitting; or data leakage between splits; or val preprocessing differs |
| Val bad only at eval time | Forgot `model.eval()` (dropout/BN active), different normalization at test, tokenizer mismatch |
| Loss decreases but metric doesn't | Loss/metric mismatch, threshold not tuned, bug in metric code, label mapping bug |
| Loss jumps every epoch boundary | Data order not reshuffled per epoch, BatchNorm with last tiny batch (`drop_last=True`) |
| Slow convergence | No normalization, bad init, LR too low, no momentum, unscaled inputs |
| Works on 1 GPU, breaks on 8 | Different effective batch size → rescale LR, BatchNorm stats, DistributedSampler not seeded per epoch |
| Non-reproducible | Seeds not set, nondeterministic CUDA kernels, data loader worker RNG, atomic adds in scatter ops |

## Experience signals

- "First I overfit one batch. Then I check the initial loss is `ln(C)`. Those two catch
  most bugs before I waste GPU hours."
- "Learning rate is the hyperparameter; everything else is second-order."
- "AdamW, not Adam-with-L2; exclude LayerNorm and bias params from decay."
- "Pre-LN transformers with warmup and gradient clipping at 1.0 — that's the stable recipe."
- "BatchNorm has train/eval modes; half the 'my model is worse in production' tickets I've
  read about are `model.eval()` or preprocessing mismatches."
- "Activation memory, not parameters, is what OOMs you in training; gradient checkpointing
  trades ~30% compute for a big memory cut."

## Follow-up chains

- "Explain backprop." → "Why do we need to store activations?" → "How would you train a
  model that doesn't fit in memory?" → *checkpointing, mixed precision, smaller batch +
  accumulation, sharding (FSDP/ZeRO), tensor/pipeline parallelism, offloading.*
- "What's Adam?" → "Why AdamW?" → "Why warmup?" → "You see loss spikes at step 10k in a
  transformer — what do you do?" → *check for bad batches/outliers, lower LR, clip grads,
  check for fp16 overflow, use pre-LN/QK-norm, consider z-loss; roll back to the last good
  checkpoint and skip the batch (what LLM labs actually do).*
- "Why √d_k?" → "What happens without it?" → *softmax saturates, gradients vanish, attention
  becomes near one-hot early and training stalls.*
- "Why did transformers replace LSTMs?" → "What's the cost of attention?" → "How do you
  serve long contexts?" → *KV cache, GQA, FlashAttention, paged KV cache, sliding window.*
- "Dropout at inference?" → *off; the `1/(1−p)` scaling at training makes expectations
  match.*
