"""Python + NumPy essentials for ML, as runnable assertions.

Read each section, predict the output, then run:  python code/01_python_basics.py
"""
import copy
import heapq
from collections import Counter, defaultdict

import numpy as np


def section(title):
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- containers
section("containers and slicing")
xs = [10, 20, 30, 40, 50]
assert xs[0] == 10 and xs[-1] == 50
assert xs[1:3] == [20, 30]          # end-exclusive
assert xs[::-1] == [50, 40, 30, 20, 10]
shape = (32, 128)                    # tuples are immutable; shapes are tuples
cfg = {"lr": 3e-4, "batch_size": 32}
cfg["epochs"] = 3                    # dicts keep insertion order
assert list(cfg) == ["lr", "batch_size", "epochs"]
assert 3 in {1, 2, 3}                # set membership is O(1)
print("ok")

# ------------------------------------------------------------ comprehensions
section("comprehensions, zip, enumerate")
preds, labels = [1, 0, 1, 1], [1, 0, 0, 1]
correct = [p == y for p, y in zip(preds, labels)]
acc = sum(correct) / len(labels)
assert acc == 0.75
idx_of = {name: i for i, name in enumerate(["cat", "dog", "bird"])}
assert idx_of["dog"] == 1
print(f"accuracy={acc:.2f}, idx_of={idx_of}")

# ------------------------------------------------------------------- gotchas
section("gotchas that signal experience")


def append_bad(x, acc=[]):           # the mutable default is created ONCE
    acc.append(x)
    return acc


def append_good(x, acc=None):
    acc = [] if acc is None else acc
    acc.append(x)
    return acc


assert append_bad(1) == [1] and append_bad(2) == [1, 2]      # surprise: shared list
assert append_good(1) == [1] and append_good(2) == [2]

a = [[1, 2], [3, 4]]
shallow, deep = copy.copy(a), copy.deepcopy(a)
a[0].append(99)
assert shallow[0] == [1, 2, 99] and deep[0] == [1, 2]       # shallow shares inner lists

assert 7 / 2 == 3.5 and 7 // 2 == 3
assert 0.1 + 0.2 != 0.3 and abs((0.1 + 0.2) - 0.3) < 1e-9   # never compare floats with ==

fns = [lambda: i for i in range(3)]                           # late binding: all see i == 2
assert [f() for f in fns] == [2, 2, 2]
fns = [lambda i=i: i for i in range(3)]                       # bind at definition time
assert [f() for f in fns] == [0, 1, 2]

lst = [3, 1, 2]
assert lst.sort() is None and lst == [1, 2, 3]                # sort() is in-place, returns None
print("ok")

# ----------------------------------------------------------- classes/dunder
section("classes: __call__ is why nn.Module uses forward()")


class Linear:
    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, x):
        return x * self.w + self.b

    def __call__(self, x):           # model(x) -> model.forward(x) (nn.Module adds hooks around it)
        return self.forward(x)


class Affine(Linear):
    def __init__(self, w, b, scale):
        super().__init__(w, b)
        self.scale = scale

    def forward(self, x):
        return super().forward(x) * self.scale


assert Linear(2, 1)(3) == 7 and Affine(2, 1, 10)(3) == 70
print("ok")

# ---------------------------------------------------------------- generators
section("generators: how a DataLoader iterates")


def batches(data, bs):
    for i in range(0, len(data), bs):
        yield data[i:i + bs]          # lazily produces one batch at a time


assert list(batches(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]
gen = batches([1, 2, 3], 2)
assert list(gen) == [[1, 2], [3]] and list(gen) == []          # generators are consumed once
print("ok")

# ---------------------------------------------------------------- collections
section("Counter / defaultdict / heapq (interview staples)")
events = [("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u3", "i1")]
pop = Counter(item for _, item in events)
assert pop.most_common(1) == [("i1", 3)]
by_user = defaultdict(list)
for u, i in events:
    by_user[u].append(i)
assert by_user["u1"] == ["i1", "i2"]
scores = {"a": 0.2, "b": 0.9, "c": 0.5, "d": 0.7}
top2 = heapq.nlargest(2, scores.items(), key=lambda kv: kv[1])   # O(n log k), no full sort
assert [k for k, _ in top2] == ["b", "d"]
print("ok")

# ---------------------------------------------------------------------- numpy
section("NumPy: shapes, axes, broadcasting")
X = np.array([[1., 2., 3.], [4., 5., 6.]])            # (2, 3): 2 examples, 3 features
assert X.shape == (2, 3) and X.T.shape == (3, 2)
assert X.sum(axis=0).shape == (3,)                      # collapse rows -> per-feature
assert X.sum(axis=1).shape == (2,)                      # collapse cols -> per-example
assert X.mean(axis=1, keepdims=True).shape == (2, 1)    # keepdims -> broadcastable
mu = X.mean(axis=0)                                     # (3,)
assert (X - mu).shape == (2, 3)                         # (2,3) - (3,) broadcasts from the right
row_norm = np.linalg.norm(X, axis=1)                    # (2,)
try:
    X - row_norm                                        # (2,3) - (2,) is NOT aligned
    raise AssertionError("should have failed")
except ValueError:
    pass
assert (X - row_norm[:, None]).shape == (2, 3)          # make it (2,1) first
# the silent version of the same bug: (N,1) vs (N,) -> (N,N)
col = np.ones((4, 1))
flat = np.ones(4)
assert (col - flat).shape == (4, 4)                     # silent! this is the MSELoss(pred,target) bug

assert (X @ X.T).shape == (2, 2) and (X * X).shape == (2, 3)   # matmul vs elementwise
mask = X > 2
assert X[mask].tolist() == [3., 4., 5., 6.]
v = X[0]                                                # a view: shares memory
v[0] = 100.
assert X[0, 0] == 100.
print("ok")

# --------------------------------------------------------------- ML snippets
section("softmax / logistic regression step / k-means step")


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # shift by max: exp can't overflow, result unchanged
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


p = softmax(np.array([[1000., 1000., 1000.]]))          # naive exp(1000) would overflow to inf
assert np.allclose(p, [[1 / 3] * 3])

rng = np.random.default_rng(0)
N, D = 200, 3
Xlr = rng.normal(size=(N, D))
w_true = np.array([2., -1., 0.5])
y = (Xlr @ w_true + 0.1 * rng.normal(size=N) > 0).astype(float)
w, b, lr = np.zeros(D), 0.0, 0.5
for _ in range(300):
    prob = 1 / (1 + np.exp(-(Xlr @ w + b)))
    grad_w = Xlr.T @ (prob - y) / N                     # d(mean BCE)/dw
    grad_b = (prob - y).mean()
    w -= lr * grad_w
    b -= lr * grad_b
acc = (((Xlr @ w + b) > 0) == (y == 1)).mean()
assert acc > 0.95, acc
print(f"logistic regression train acc={acc:.3f}, w/|w|={np.round(w / np.linalg.norm(w), 2)}")

pts = np.concatenate([rng.normal(0, 0.5, (50, 2)), rng.normal(5, 0.5, (50, 2))])
C = pts[rng.choice(len(pts), 2, replace=False)]
for _ in range(10):
    d = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (N, K) squared distances
    assign = d.argmin(1)
    C = np.stack([pts[assign == k].mean(0) for k in range(2)])
assert np.allclose(np.sort(C[:, 0]), [0, 5], atol=0.3)
print(f"k-means centroids ~ {np.round(C, 1).tolist()}")

print("\nAll checks passed.")
