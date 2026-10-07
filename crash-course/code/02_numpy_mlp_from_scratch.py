"""A 2-layer neural network with hand-written backprop, in NumPy only.

This is what PyTorch does for you. If you can explain every line here, you can
explain backprop, cross-entropy, Adam, and why activations must be stored.

Run:  python code/02_numpy_mlp_from_scratch.py
"""
import numpy as np

rng = np.random.default_rng(0)


# ------------------------------------------------------------------ data
def make_moons(n, noise=0.15):
    """Two interleaved half-circles: not linearly separable, so a linear model fails."""
    t = rng.uniform(0, np.pi, n // 2)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([a, b]) + noise * rng.normal(size=(n, 2))
    y = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    perm = rng.permutation(n)
    return X[perm], y[perm]


X, y = make_moons(1000)
X_train, y_train, X_val, y_val = X[:800], y[:800], X[800:], y[800:]
mu, sd = X_train.mean(0), X_train.std(0)             # fit normalization on TRAIN only
X_train, X_val = (X_train - mu) / sd, (X_val - mu) / sd


# ------------------------------------------------------------------ model
def init_params(d_in, d_h, n_classes):
    # Kaiming init for ReLU: std = sqrt(2 / fan_in); zeros would make every unit identical.
    return {
        "W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_h)), "b1": np.zeros(d_h),
        "W2": rng.normal(0, np.sqrt(2 / d_h), (d_h, n_classes)), "b2": np.zeros(n_classes),
    }


def forward(P, X):
    z1 = X @ P["W1"] + P["b1"]            # (N, H)  pre-activation
    h = np.maximum(z1, 0)                 # (N, H)  ReLU
    logits = h @ P["W2"] + P["b2"]        # (N, C)
    cache = (X, z1, h)                    # stored for backward -> "activation memory"
    return logits, cache


def log_softmax(logits):
    m = logits.max(1, keepdims=True)      # log-sum-exp trick for numerical stability
    return logits - m - np.log(np.exp(logits - m).sum(1, keepdims=True))


def loss_fn(logits, y):
    logp = log_softmax(logits)
    return -logp[np.arange(len(y)), y].mean()     # cross-entropy = mean NLL of the true class


def backward(P, cache, logits, y):
    X, z1, h = cache
    N = len(y)
    p = np.exp(log_softmax(logits))
    dlogits = p.copy()
    dlogits[np.arange(N), y] -= 1                 # d(CE)/d(logits) = p - onehot(y)
    dlogits /= N                                  # because the loss is a mean
    grads = {}
    grads["W2"] = h.T @ dlogits                   # (H, C)   chain rule through logits = h W2 + b2
    grads["b2"] = dlogits.sum(0)
    dh = dlogits @ P["W2"].T                      # (N, H)
    dz1 = dh * (z1 > 0)                           # ReLU gradient: pass where z1 > 0, else 0
    grads["W1"] = X.T @ dz1                       # (D, H)
    grads["b1"] = dz1.sum(0)
    return grads


# ------------------------------------------------------------ gradient check
P = init_params(2, 16, 2)
Xs, ys = X_train[:8], y_train[:8]
logits, cache = forward(P, Xs)
grads = backward(P, cache, logits, ys)
eps = 1e-5
max_err = 0.0
for name in P:
    for idx in [np.unravel_index(i, P[name].shape) for i in rng.choice(P[name].size, 5)]:
        old = P[name][idx]
        P[name][idx] = old + eps
        lp = loss_fn(forward(P, Xs)[0], ys)
        P[name][idx] = old - eps
        lm = loss_fn(forward(P, Xs)[0], ys)
        P[name][idx] = old
        numeric = (lp - lm) / (2 * eps)           # central finite difference
        max_err = max(max_err, abs(numeric - grads[name][idx]) / (abs(numeric) + abs(grads[name][idx]) + 1e-12))
print(f"gradient check: max relative error = {max_err:.2e} (should be < 1e-5)")
assert max_err < 1e-5


# ---------------------------------------------------------------- training
def adam_step(P, grads, state, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    for k in P:
        m, v = state.setdefault(k, (np.zeros_like(P[k]), np.zeros_like(P[k])))
        m = b1 * m + (1 - b1) * grads[k]                  # first moment (momentum)
        v = b2 * v + (1 - b2) * grads[k] ** 2             # second moment (per-parameter scale)
        state[k] = (m, v)
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)   # bias correction: moments start at 0
        P[k] -= lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * P[k])   # AdamW: decay decoupled from grad


def accuracy(P, X, y):
    return (forward(P, X)[0].argmax(1) == y).mean()


P = init_params(2, 32, 2)
init_loss = loss_fn(forward(P, X_train)[0], y_train)
print(f"loss at init = {init_loss:.3f}; expected ~ ln(2) = {np.log(2):.3f}  (sanity check #1)")

state, t = {}, 0
batch_size, lr = 64, 1e-2
for epoch in range(60):
    perm = rng.permutation(len(X_train))              # reshuffle every epoch
    for i in range(0, len(X_train), batch_size):
        idx = perm[i:i + batch_size]
        logits, cache = forward(P, X_train[idx])
        grads = backward(P, cache, logits, y_train[idx])
        t += 1
        adam_step(P, grads, state, lr, t, wd=1e-4)
    if epoch % 10 == 0 or epoch == 59:
        tr = loss_fn(forward(P, X_train)[0], y_train)
        print(f"epoch {epoch:2d}  train loss {tr:.3f}  train acc {accuracy(P, X_train, y_train):.3f}  val acc {accuracy(P, X_val, y_val):.3f}")

val_acc = accuracy(P, X_val, y_val)
assert val_acc > 0.9, val_acc

# sanity check #2: a linear model (no hidden layer) can't solve moons -> proves the nonlinearity matters
w = np.linalg.lstsq(np.c_[X_train, np.ones(len(X_train))], y_train * 2 - 1, rcond=None)[0]
lin_acc = ((np.c_[X_val, np.ones(len(X_val))] @ w > 0) == (y_val == 1)).mean()
print(f"\nlinear baseline val acc {lin_acc:.3f} vs MLP val acc {val_acc:.3f}")
print("Done.")
