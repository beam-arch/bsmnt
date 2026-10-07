"""The canonical PyTorch training loop, annotated, plus live demonstrations of the
classic bugs (grad accumulation, softmax-before-CE, the (N,1) vs (N,) broadcast).

Run:  python code/03_torch_training_loop.py
"""
import math
import os
import tempfile
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  torch {torch.__version__}")


# ------------------------------------------------------------------ data
def make_moons(n, noise=0.15, rng=np.random.default_rng(0)):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.stack([np.cos(t), np.sin(t)], 1)
    b = np.stack([1 - np.cos(t), 1 - np.sin(t) - 0.5], 1)
    X = np.concatenate([a, b]) + noise * rng.normal(size=(n, 2))
    y = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    perm = rng.permutation(n)
    return X[perm].astype(np.float32), y[perm]


class MoonsDataset(Dataset):
    """A Dataset is just __len__ + __getitem__; the DataLoader does batching/shuffling/workers."""

    def __init__(self, X, y, mean=None, std=None):
        self.mean = X.mean(0) if mean is None else mean      # normalization stats come from TRAIN
        self.std = X.std(0) if std is None else std
        self.X = torch.from_numpy((X - self.mean) / self.std)   # float32 features
        self.y = torch.from_numpy(y).long()                     # CrossEntropyLoss wants int64 labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


X, y = make_moons(1000)
train_ds = MoonsDataset(X[:800], y[:800])
val_ds = MoonsDataset(X[800:], y[800:], train_ds.mean, train_ds.std)   # reuse train stats: no leakage
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)


# ------------------------------------------------------------------ model
class MLP(nn.Module):
    def __init__(self, d_in=2, d_hidden=32, n_classes=2, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_classes),        # outputs raw logits -- no softmax here
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params}")

# Decoupled weight decay (AdamW); biases are typically excluded from decay.
decay = [p for n, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 1e-2},
                               {"params": no_decay, "weight_decay": 0.0}], lr=3e-3)
epochs = 30
steps = epochs * len(train_loader)
warmup = int(0.05 * steps)
# warmup then cosine decay: the standard transformer-era schedule
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: s / max(1, warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
criterion = nn.CrossEntropyLoss()      # = log_softmax + NLL, numerically fused


@torch.no_grad()                          # no graph building during eval -> less memory, faster
def evaluate(model, loader):
    model.eval()                          # dropout off, BatchNorm uses running stats
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    model.train()                         # back to training mode!
    return total_loss / n, correct / n


# sanity check: loss at init should be ~ ln(num_classes)
init_loss, _ = evaluate(model, val_loader)
print(f"loss at init {init_loss:.3f} vs ln(2) = {math.log(2):.3f}")

# -------------------------------------------------------------- training
model.train()
for epoch in range(epochs):
    running = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)          # 1. clear accumulated grads
        logits = model(xb)                             # 2. forward
        loss = criterion(logits, yb)                   # 3. loss (on logits!)
        loss.backward()                                # 4. backward: grads land in p.grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 5. clip by global norm
        optimizer.step()                               # 6. update weights
        scheduler.step()                               # 7. advance LR schedule
        running += loss.item()                         # .item() syncs with the GPU; fine here
    if epoch % 10 == 0 or epoch == epochs - 1:
        val_loss, val_acc = evaluate(model, val_loader)
        print(f"epoch {epoch:2d}  train loss {running / len(train_loader):.3f}  "
              f"val loss {val_loss:.3f}  val acc {val_acc:.3f}  lr {scheduler.get_last_lr()[0]:.2e}")

val_loss, val_acc = evaluate(model, val_loader)
assert val_acc > 0.9, val_acc

# ---------------------------------------------------------- checkpointing
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "ckpt.pt")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "epoch": epochs,
                "norm": {"mean": train_ds.mean, "std": train_ds.std}}, path)   # save preprocessing too!
    ckpt = torch.load(path, map_location=device, weights_only=False)  # weights_only=True for plain tensors
    model2 = MLP().to(device)
    model2.load_state_dict(ckpt["model"])
    _, acc2 = evaluate(model2, val_loader)
    assert abs(acc2 - val_acc) < 1e-9
    print(f"checkpoint round-trip ok (val acc {acc2:.3f})")

# ===================================================== bug demonstrations
print("\n--- bug 1: forgetting zero_grad() accumulates gradients ---")
m = nn.Linear(2, 2)
xb, yb = next(iter(train_loader))
F.cross_entropy(m(xb), yb).backward()
g1 = m.weight.grad.clone()
F.cross_entropy(m(xb), yb).backward()          # no zero_grad in between
print(f"grad after 2 backward calls == 2x single grad: {torch.allclose(m.weight.grad, 2 * g1)}")

print("\n--- bug 2: softmax before CrossEntropyLoss (double softmax) ---")
logits = torch.tensor([[4.0, -4.0]])
target = torch.tensor([1])                     # the wrong class is confidently predicted
good = F.cross_entropy(logits, target)
bad = F.cross_entropy(F.softmax(logits, 1), target)   # probabilities treated as logits
print(f"correct loss {good.item():.3f}  vs  double-softmax loss {bad.item():.3f} "
      f"(bounded, tiny gradients -> training crawls)")

print("\n--- bug 3: MSE between (N,1) and (N,) broadcasts to (N,N) ---")
pred = torch.randn(5, 1)
target = torch.randn(5)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    wrong = F.mse_loss(pred, target)
    print(f"PyTorch warned: {bool(w)}; wrong loss {wrong.item():.3f} vs right {F.mse_loss(pred.squeeze(1), target).item():.3f}")

print("\n--- bug 4: nn.Module inside a plain list is not registered ---")


class Bad(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 2)]            # should be nn.ModuleList


class Good(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])


print(f"Bad params: {sum(p.numel() for p in Bad().parameters())}, Good params: {sum(p.numel() for p in Good().parameters())}")

print("\n--- bug 5: model.eval() matters when there's dropout ---")
model.train()
xb = val_ds.X[:4].to(device)
with torch.no_grad():
    outs = torch.stack([model(xb) for _ in range(3)])
print(f"train-mode outputs vary across calls: {not torch.allclose(outs[0], outs[1])}")
model.eval()
with torch.no_grad():
    outs = torch.stack([model(xb) for _ in range(3)])
print(f"eval-mode outputs are deterministic: {torch.allclose(outs[0], outs[1])}")
print("\nDone.")
