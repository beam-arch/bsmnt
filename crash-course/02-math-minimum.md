# 02 · The math minimum (45 minutes)

Only the math that interviews actually touch, and always tied to where it shows up in ML.
You do not need to derive anything; you need to know what each object *is* and *why it
matters*.

## 1. Linear algebra

- A **vector** is a list of numbers: a data point (`x ∈ ℝ^D`), a weight vector, an embedding.
- The **dot product** `a·b = Σ aᵢbᵢ = |a||b|cos θ` measures alignment. Normalize both and
  you get **cosine similarity** — this is how embeddings are compared in search, recsys, RAG.
- A **matrix** is a table of numbers; multiplying by it is a linear map. In ML a matrix is
  usually either *data* (`X`: N rows of examples × D features) or *weights* (`W`: D_in × D_out).
- **Matrix multiplication** `(m×k)(k×n) → (m×n)`, inner dims must match, costs `2mkn` FLOPs.
  A linear layer is `Y = XW + b`: `(N×D_in)(D_in×D_out)`. Batches are just rows — that is
  why GPUs love ML: **almost all the compute is matmul**.
- **Transpose** swaps axes. `XᵀX` is the (unnormalized) covariance — `D×D`.
- **Norms**: L2 `‖x‖₂ = √Σxᵢ²` (Euclidean length; weight decay penalizes it), L1 `Σ|xᵢ|`
  (encourages sparsity: lasso).
- **Rank** = number of independent directions. A **low-rank** matrix `W ≈ UVᵀ` with
  `U: m×r, V: n×r, r ≪ m,n` stores `r(m+n)` numbers instead of `mn`. This single idea is
  matrix factorization for recommenders (`user_emb · item_emb`) *and* LoRA for LLM
  fine-tuning (`ΔW = BA`).
- **Eigenvectors/eigenvalues**: directions a matrix only stretches. For a covariance
  matrix they are the principal components; eigenvalue = variance along that direction.
  **SVD** `X = UΣVᵀ` generalizes this to any matrix; truncating to top-k singular values gives
  the best rank-k approximation (PCA, classic "SVD" recommenders).
- **Orthogonal** matrices preserve lengths (rotations). Orthogonal init keeps activations'
  scale stable in RNNs.

## 2. Calculus (just enough for backprop)

- The **derivative** `df/dx` is sensitivity: how much `f` changes per tiny change in `x`.
- The **gradient** `∇L(w)` is the vector of partial derivatives of the loss w.r.t. every
  parameter. It points in the direction of steepest *increase*. So we step the other way:
  **`w ← w − η ∇L(w)`** (gradient descent; `η` is the learning rate).
- The **chain rule**: if `L = f(g(w))` then `dL/dw = f'(g(w)) · g'(w)`. A neural network is a
  long chain of functions; **backpropagation is just the chain rule applied from the loss
  backwards, reusing intermediate results.** That reuse is why the backward pass costs
  only ~2× the forward pass, and why we must *store activations* from the forward pass
  (→ activation memory dominates GPU memory in training).
- **Jacobian**: matrix of all partials for a vector-valued function. Autograd never builds
  it explicitly; it computes vector-Jacobian products (reverse mode), which is efficient
  when you have many inputs (parameters) and one output (the scalar loss).
- **Hessian**: matrix of second derivatives (curvature). Newton's method uses it; too
  expensive at NN scale (`P×P` for P parameters), so we use first-order methods with
  adaptive per-parameter scaling (Adam) as a cheap curvature proxy.
- **Convex** functions have a single global minimum (linear/logistic regression). Neural
  net losses are non-convex; in high dimensions the practical obstacle is saddle points and
  flat regions, not "bad local minima" — SGD noise helps escape them.
- Derivatives you may be asked to write: `d/dx sigmoid(x) = σ(x)(1−σ(x))`,
  `d/dx ReLU(x) = 1[x>0]`, `d/dz softmax-cross-entropy = p − y` (probabilities minus one-hot
  target — elegant, and the reason softmax+CE are fused in every framework).

## 3. Probability

- A **distribution** assigns probabilities to outcomes. Key ones: **Bernoulli** (click / no
  click), **Categorical** (which of C classes / which next token), **Gaussian** (noise,
  weights at init), **Uniform**, **Poisson** (counts). Expectation `E[X]` = mean; variance
  `Var[X] = E[(X−μ)²]`.
- **Conditional probability** `P(A|B) = P(A,B)/P(B)`. **Bayes' rule**:
  `P(θ|data) ∝ P(data|θ) P(θ)` — posterior ∝ likelihood × prior.
- **Likelihood** `P(data | θ)`: how probable the observed data is under parameters θ.
  **Maximum likelihood estimation (MLE)** picks θ to maximize it. Since logs turn products
  into sums, we minimize the **negative log-likelihood (NLL)**.
- **This is the unifying fact of ML losses** (interviewers love it):
  - Gaussian noise assumption → NLL is **mean squared error**.
  - Bernoulli labels → NLL is **binary cross-entropy**.
  - Categorical labels → NLL is **cross-entropy** (softmax + log + pick the true class).
  - Laplace noise → **MAE**.
  - Adding a Gaussian prior on weights (MAP instead of MLE) → **L2 regularization**;
    Laplace prior → **L1**.
- **Entropy** `H(p) = −Σ p log p` measures uncertainty. **Cross-entropy** `H(p,q) = −Σ p log q`
  is the expected code length using `q` when truth is `p`. **KL divergence**
  `KL(p‖q) = H(p,q) − H(p) ≥ 0` measures how different `q` is from `p` (used in distillation,
  VAEs, RLHF's KL penalty keeping the policy near the reference model).
- **Perplexity** `= exp(cross-entropy per token)`: "how many equally likely tokens the model
  is choosing between on average". Lower is better.
- **Independence** and i.i.d.: standard ML assumes training and test examples are drawn
  independently from the same distribution. Most production failures are violations of
  this (distribution shift, temporal correlation, duplicated users across splits).
- **Calibration**: a model is calibrated if, among examples where it predicts 0.7, about
  70% are positive. Cross-entropy training encourages calibration; modern deep nets and
  class re-weighting break it; ads/recsys need it because scores get multiplied by money.

## 4. Statistics for experiments

- **Mean vs median**: median is robust to outliers (latency p50 is a median; p99 is a tail).
- **Standard error** of a mean shrinks like `σ/√n`: quadrupling data halves the noise.
- **Central limit theorem**: averages of many samples are approximately Gaussian → this is
  why A/B tests use t-tests/z-tests on metric averages.
- **Hypothesis test**: null hypothesis "no difference"; the **p-value** is the probability
  of seeing an effect at least this large *if the null were true*. It is *not* the
  probability the null is true. `p < 0.05` at a 5% false-positive rate.
- **Power** = probability of detecting a real effect of a given size. Compute required
  sample size *before* the test. Underpowered tests produce "no effect" that means nothing.
- **Confidence interval**: range of effect sizes consistent with the data. Always report
  the interval, not just significance; a "significant" +0.01% CTR lift may be worthless.
- **Multiple comparisons**: test 20 metrics at p<0.05 and one is "significant" by chance.
  Correct (Bonferroni) or pre-register the primary metric.
- **Peeking**: checking an A/B test daily and stopping when significant inflates false
  positives. Fix with fixed horizons or sequential tests.
- **Bootstrapping**: resample your data with replacement many times to get confidence
  intervals for any metric (e.g. NDCG, AUC) without formulas.
- **Correlation ≠ causation**; **Simpson's paradox**: a trend reverses when groups are
  combined (e.g. a new model wins on mobile and desktop separately but loses overall
  because the traffic mix changed).
- **Unit of randomization** in A/B tests must match the unit of analysis (randomize by user,
  analyze by user; analyzing by request when users are randomized underestimates variance).

## 5. Numerics (the part people actually get wrong)

- **Floating point** has finite precision. `fp32`: 1 sign, 8 exponent, 23 mantissa bits
  (~7 decimal digits). `fp16`: 5 exponent, 10 mantissa, max value 65,504 — *overflows
  easily* (gradients, attention logits). `bf16`: 8 exponent, 7 mantissa — same range as
  fp32, less precision; the default for LLM training because it doesn't overflow.
  `tf32`: fp32 range with 10 mantissa bits, used inside Tensor Cores for matmul.
- Products of probabilities underflow → always work in **log space**.
  `log Σ exp(xᵢ) = m + log Σ exp(xᵢ − m)` with `m = max xᵢ` (**log-sum-exp trick**). This is
  why you call `log_softmax` + `nll_loss` (or `CrossEntropyLoss`, which fuses them) rather
  than `log(softmax(x))`.
- The `ε` in Adam (`1e-8`), LayerNorm (`1e-5`), and `log(p + ε)` exists to avoid dividing
  by or taking the log of zero. In fp16, `1e-8` *is* zero — a real source of NaNs with
  mixed precision.
- Floating-point addition is not associative, so summing in a different order (different
  GPU, different batch size, different number of threads) gives slightly different results.
  Bitwise reproducibility across hardware is unrealistic; "same seed, same machine, same
  software, deterministic kernels" is the achievable standard.
- Integer overflow still exists in NumPy (`int32` counters wrapping at 2.1 billion) — a real
  class of feature-pipeline bugs.

## Follow-up chains

- "Why is cross-entropy the loss for classification?" → "What's its relationship to MLE?"
  → "What's the gradient w.r.t. logits?" → `p − y`.
- "What does the learning rate do?" → "What happens if it's too large / too small?" →
  "Why warmup?" → *early steps have unreliable gradient/second-moment estimates; big early
  steps can destroy the init (especially in post-LN transformers).*
- "What's a p-value?" → "The test shows p=0.03 after 2 days of a planned 14-day test, do
  you ship?" → *no: peeking inflates false positives, and short tests capture novelty
  effects; wait for the planned horizon or use a sequential method.*
- "Why bf16 over fp16?" → *range; no loss scaling needed.* → "Why keep fp32 master
  weights?" → *tiny updates (lr × grad) underflow when added to a low-precision weight.*
