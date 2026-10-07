# 01 · Python essentials for ML (75 minutes)

Goal: read any ML code fluently, write a small function under interview pressure, and know
the Python facts that mark someone as having actually used the language.

Run `code/01_python_basics.py` alongside this module.

## 1. The mental model

- Python is interpreted and dynamically typed: you don't declare types, and the program runs
  line by line. Blocks are defined by **indentation** (4 spaces), not braces.
- **Everything is an object**, including functions and classes. Variables are *names bound
  to objects*, not boxes. `a = b` makes two names for the *same* object.
- It is slow per operation (interpreter overhead, ~50–100× slower than C for loops). ML in
  Python works because the heavy lifting happens inside C/CUDA libraries (NumPy, PyTorch);
  Python only orchestrates. The rule: **never loop over numbers in Python; vectorize.**

## 2. Core syntax (what you'll read in every ML codebase)

```python
# scalars & strings
n = 10            # int
lr = 3e-4         # float (scientific notation; the "Adam default" meme)
name = "mlp"      # str
ok = True         # bool
nothing = None    # absence of a value

# f-strings: formatting
print(f"epoch {n}: lr={lr:.5f} model={name}")

# containers
xs = [1, 2, 3]                     # list: ordered, mutable
pt = (1.0, 2.0)                    # tuple: ordered, immutable (used for shapes!)
cfg = {"lr": 3e-4, "bs": 32}       # dict: key -> value, insertion ordered (3.7+)
seen = {1, 2, 3}                   # set: unique elements, O(1) membership

# indexing & slicing (0-based, end-exclusive, negatives count from the end)
xs[0], xs[-1], xs[1:3], xs[::-1]   # first, last, elements 1..2, reversed

# control flow
for i, x in enumerate(xs):         # enumerate gives index + value
    if x % 2 == 0:
        continue
    elif x > 10:
        break
while n > 0:
    n -= 1

# comprehensions (idiomatic; faster than appending in a loop)
squares = [x * x for x in xs if x > 1]
lookup = {k: i for i, k in enumerate(["a", "b"])}

# functions: default args, keyword args, variable args
def train(model, data, lr=1e-3, *args, **kwargs):
    ...
train(m, d, lr=1e-2, verbose=True)

# classes: __init__ runs on construction; self is the instance
class Model:
    def __init__(self, dim):
        self.dim = dim
    def forward(self, x):
        return x * self.dim
    def __call__(self, x):          # makes model(x) work -> this is why nn.Module uses forward()
        return self.forward(x)

class Deep(Model):                  # inheritance; super() calls the parent's __init__
    def __init__(self, dim):
        super().__init__(dim)

# exceptions
try:
    risky()
except (ValueError, KeyError) as e:
    print("failed:", e)
finally:
    cleanup()

# context managers: setup/teardown around a block (files, torch.no_grad(), autocast)
with open("f.txt") as f:
    text = f.read()

# generators: lazily produce values; this is what DataLoader iteration looks like
def batches(data, bs):
    for i in range(0, len(data), bs):
        yield data[i:i + bs]

# type hints: optional, documentation only (not enforced at runtime)
def accuracy(pred: list[int], y: list[int]) -> float:
    return sum(p == t for p, t in zip(pred, y)) / len(y)

# the script entry point idiom
if __name__ == "__main__":
    main()
```

Things to internalize:
- `zip` walks several sequences together; `enumerate` adds an index; `range(a, b)` is `a..b-1`.
- `*args` collects extra positional args into a tuple; `**kwargs` collects extra keyword args
  into a dict. `f(*lst)` / `f(**dct)` unpack them. You see `**kwargs` everywhere in ML libraries.
- Lambda: `key=lambda x: x[1]` is an inline function; `sorted(pairs, key=lambda p: -p[1])`.
- Truthiness: `0`, `0.0`, `""`, `[]`, `{}`, `None` are false. (`if tensor:` on a multi-element
  tensor raises an error — a common PyTorch confusion.)

## 3. NumPy: the actual language of ML

NumPy arrays (`ndarray`) are typed, contiguous blocks of memory with a **shape** and a
**dtype**. PyTorch tensors copy this design exactly, so learning NumPy *is* learning
tensors.

```python
import numpy as np
x = np.array([[1., 2., 3.], [4., 5., 6.]])   # shape (2, 3), dtype float64
x.shape, x.dtype, x.ndim                      # (2, 3), float64, 2
x.T                                           # transpose: (3, 2)
x @ x.T                                       # matrix multiply: (2,3)@(3,2) -> (2,2)
x * x                                         # elementwise (Hadamard) product, NOT matmul
x.sum(), x.sum(axis=0), x.sum(axis=1)         # all, per column (collapse rows), per row
x.mean(axis=1, keepdims=True)                 # (2, 1) instead of (2,) -> keeps it broadcastable
x[x > 2]                                      # boolean mask indexing
x[:, 0]                                       # all rows, column 0 -> shape (2,)
np.argmax(x, axis=1)                          # predicted class per row
np.random.default_rng(0).normal(size=(4, 3))  # modern RNG API
```

**Axis semantics** (interviewers check this): `axis=0` means "collapse along rows, i.e.
operate down each column". For a batch `X` of shape `(N, D)`, feature means are
`X.mean(axis=0)` with shape `(D,)`, per-example norms are `np.linalg.norm(X, axis=1)` with
shape `(N,)`.

**Broadcasting rules** (the source of half of all shape bugs): when two arrays are combined
elementwise, shapes are aligned from the right; a dimension matches if it is equal or if one
of them is 1 (it gets stretched). `(N, D) - (D,)` works (subtract the feature mean).
`(N, D) - (N,)` fails — you need `(N, 1)`. `(N, 1) - (N,)` *silently* produces `(N, N)`;
that exact mistake in a loss function is a famous PyTorch bug (`MSELoss` on `(N,1)` vs `(N,)`).

**Views vs copies:** slicing gives a *view* (shares memory); modifying it modifies the
original. `reshape` returns a view when it can. Fancy/boolean indexing returns a copy.

**Why vectorized code is fast:** the loop runs in compiled C over contiguous memory with
SIMD instructions, no per-element interpreter overhead and no per-element type checks.
A Python loop over a million floats takes ~100 ms; NumPy does it in ~1 ms.

## 4. Pandas in 5 minutes

```python
import pandas as pd
df = pd.read_csv("events.csv")                       # DataFrame: table with named columns
df.head(); df.describe(); df.isna().sum()            # inspect; count missing values
df["ctr"] = df["clicks"] / df["impressions"]         # vectorized column math
agg = df.groupby("user_id")["clicks"].sum()          # split-apply-combine
joined = df.merge(users, on="user_id", how="left")   # SQL-style join
df = df.fillna({"age": df["age"].median()})          # imputation
df.sort_values("ts").drop_duplicates(["user_id", "item_id"], keep="last")
df.to_parquet("out.parquet")                         # columnar format; much faster than csv
```

Experience signal: `merge(how="left")` on a key with duplicates silently *multiplies rows*
(a 1:many join inflating your training set — then your metrics look better because
duplicates leak across train/test). Always check `len()` before and after a join.

## 5. The Python gotchas that signal experience

1. **Mutable default arguments**: `def f(x, acc=[])` shares one list across all calls.
   Use `acc=None` then `acc = acc or []`.
2. **`is` vs `==`**: `is` compares identity, `==` compares value. Use `is None`.
3. **Integer vs float division**: `/` is float division, `//` is floor division. `7 // 2 == 3`.
4. **Float equality**: `0.1 + 0.2 != 0.3`. Use `math.isclose` / `np.allclose` in tests.
5. **Late-binding closures**: lambdas in a loop capture the variable, not its value at the time.
6. **`list.sort()` returns `None`** (in-place); `sorted(lst)` returns a new list.
7. **Shallow vs deep copy**: `copy.copy` copies the container, not nested objects.
8. **The GIL** (Global Interpreter Lock): only one thread executes Python bytecode at a time.
   Threads help for I/O, not CPU work. That is why `DataLoader(num_workers=k)` uses
   *processes*, and why NumPy/PyTorch release the GIL inside their C kernels.
9. **Environments**: `venv`/`uv`/`conda` isolate dependencies; pin versions
   (`torch==2.4.1`) and use a lockfile for reproducibility. "Works on my machine" is
   usually an unpinned dependency.
10. **`pickle` is code execution**: loading an untrusted `.pkl` / old-style `torch.load`
    file can run arbitrary code. Hence `torch.load(..., weights_only=True)` and the
    `safetensors` format.
11. **Iterators are consumed**: you can loop a generator once. `zip(*batch)` transposes a
    list of tuples (the collate idiom).
12. **Big-O you should know**: list append O(1) amortized, `x in list` O(n), `x in set/dict`
    O(1) average, sort O(n log n), dict/set are hash tables. Matmul `(m×k)(k×n)` costs
    O(mkn) — 2·m·k·n FLOPs.

## 6. Coding-screen patterns (ML flavoured)

Typical asks for an MLE coding round, roughly in order of likelihood:

1. **Data wrangling**: group/aggregate/join; compute CTR per item; top-k items per user.
2. **Implement a metric**: precision/recall, AUC, NDCG@K (see `code/07_metrics.py`).
3. **Implement a small algorithm**: k-means, logistic regression with gradient descent,
   k-NN, softmax (with the max-subtraction trick), scaled dot-product attention.
4. **Write a batching iterator / Dataset class.**
5. **Classic easy/medium algorithm**: two-sum (hash map), sliding window, BFS on a grid,
   merge intervals, top-k with a heap.

The protocol that interviewers reward: restate the problem, ask about input sizes and edge
cases, state a brute-force solution, improve it, write it while talking, trace one small
example by hand, state time/space complexity.

Templates to memorize:

```python
# counting with a dict
from collections import Counter, defaultdict
counts = Counter(items)                      # counts.most_common(5)
by_user = defaultdict(list)
for u, i in events: by_user[u].append(i)

# top-k without full sort: O(n log k)
import heapq
topk = heapq.nlargest(k, scores, key=lambda s: s[1])

# numerically stable softmax
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)   # shift by max: exp() can't overflow
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# logistic regression, one gradient step
p = 1 / (1 + np.exp(-(X @ w + b)))          # (N,)
grad_w = X.T @ (p - y) / len(y)             # gradient of mean BCE w.r.t. w
w -= lr * grad_w

# k-means, one iteration
d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (N, K) squared distances
assign = d.argmin(1)
C = np.stack([X[assign == k].mean(0) for k in range(K)])
```

## Follow-up chains you may get

- "Why is NumPy faster than a Python loop?" → "What is the GIL?" → "So how does the
  DataLoader parallelize?" → *processes, not threads; fork copies the dataset object, so
  per-worker RNG state and open file handles are classic bugs.*
- "What does `axis=0` do?" → "Shape of `X.mean(axis=0)` for `(N, D)`?" → "Why `keepdims`?"
  → *to keep the result broadcastable against the original.*
- "What's the difference between a list and a NumPy array?" → *typed contiguous memory,
  vectorized ops, fixed dtype, broadcasting; a list is an array of pointers to objects.*

## Experience signals to drop (only if true for you — and they can be, after today)

- "I check `df.shape` before and after every merge."
- "I'd write the vectorized version, but first the loop version to make sure I have the
  math right, then assert they match on a tiny input."
- "Mutable default args and late-binding closures are the two Python bugs I always look for."
