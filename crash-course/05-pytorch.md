# 05 · PyTorch (75 minutes)

Run `code/03_torch_training_loop.py` (the canonical loop, annotated line by line) and
`code/04_attention_from_scratch.py` (attention + a transformer block, checked against
PyTorch's built-in) with this module.

## 1. What PyTorch is

A tensor library (NumPy on GPUs) + **automatic differentiation** + neural-net building
blocks + data loading + distributed training. It runs **eagerly** (each line executes
immediately, graphs are built dynamically as you compute), which is why it is easy to debug
with print statements and won the research world; `torch.compile` recovers graph-level
performance when you need it. Alternatives: JAX (functional, XLA-compiled, TPU-native),
TensorFlow (legacy in most new projects). Ecosystem: torchvision, torchaudio, Hugging Face
`transformers`/`datasets`/`accelerate`/`peft`, Lightning (training loop boilerplate),
`timm` (vision models).

## 2. Tensors

```python
import torch
x = torch.randn(32, 128)                 # shape (32, 128), dtype float32, device cpu
x = x.to("cuda")                         # move to GPU (copy); x.cuda() too
x.shape, x.dtype, x.device               # torch.Size([32, 128]), torch.float32, cuda:0
x.view(32, 4, 32)                        # reshape without copy (requires contiguous memory)
x.reshape(32, 4, 32)                     # same but copies if needed
x.permute(1, 0)                          # swap axes -> non-contiguous; .contiguous() to fix
x.unsqueeze(0), x.squeeze()              # add / remove size-1 dims
x @ x.T, torch.matmul, torch.einsum("bi,bj->bij", a, b)
x.sum(dim=1, keepdim=True)               # dim is NumPy's axis
x.argmax(dim=-1)
x[mask], x[:, :10]
x.add_(1.0)                              # trailing underscore = in-place (autograd hazard)
x.item()                                 # 1-element tensor -> Python number (GPU sync!)
x.detach().cpu().numpy()                 # to NumPy: must be detached from autograd and on CPU
torch.from_numpy(arr)                    # shares memory with the NumPy array
torch.tensor([1, 2, 3])                  # int64; labels for CrossEntropyLoss must be int64 (long)
```

Broadcasting works as in NumPy, with the same silent-bug risk. Integer tensors do not
track gradients. Default float is float32; `torch.set_default_dtype` exists but don't.

## 3. Autograd

- Tensors with `requires_grad=True` (all `nn.Parameter`s) record the operations applied to
  them in a graph. `loss.backward()` walks the graph in reverse, computing `dloss/dtensor`
  into each leaf's `.grad`.
- **Gradients accumulate** into `.grad` (by design, for gradient accumulation) → you must
  call `optimizer.zero_grad()` every step or gradients from previous batches add up.
- `with torch.no_grad():` disables graph recording — use for evaluation and for manual
  weight updates. `torch.inference_mode()` is a stricter, faster version for pure inference.
- `x.detach()` returns a tensor cut from the graph (same storage). Use when you want to
  stop gradient flow (e.g. targets in bootstrapping, logging).
- The graph is freed after `backward()`; calling it twice needs `retain_graph=True`
  (usually a sign of a bug).
- **In-place ops** on tensors needed for backward raise "modified by an inplace operation".
- Custom gradients: subclass `torch.autograd.Function` with `forward`/`backward`
  (rare; e.g. straight-through estimators, custom kernels).
- Memory: every intermediate kept for backward lives until `backward()` ends. Storing
  `loss` (not `loss.item()`) in a Python list across steps keeps whole graphs alive — a
  classic "memory grows every step" leak.

## 4. `nn.Module`

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, n_classes, p_drop=0.1):
        super().__init__()                               # mandatory
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, n_classes),              # raw logits out; no softmax here
        )
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])  # NOT a Python list
        self.register_buffer("pos", torch.arange(10))    # non-trainable state saved in state_dict

    def forward(self, x):
        return self.net(x)

model = MLP(20, 64, 3)
model.parameters()        # iterator over trainable tensors (what the optimizer gets)
model.state_dict()        # dict of all params + buffers (what you save)
model.to(device); model.train(); model.eval()
```

- Submodules assigned as attributes are *registered* automatically; modules inside a plain
  Python `list`/`dict` are **not** (their params won't train, won't move to GPU, won't
  save) — use `nn.ModuleList` / `nn.ModuleDict`.
- `train()`/`eval()` only toggles mode-dependent layers (Dropout, BatchNorm); it does not
  disable gradients — pair `eval()` with `no_grad()`.
- `nn.Embedding(num, dim)` is a lookup table; `padding_idx` keeps the pad row at zero.
- Losses: `nn.CrossEntropyLoss()` takes **raw logits** `(N, C)` and **int64 class indices**
  `(N,)` (not one-hot). `nn.BCEWithLogitsLoss()` takes logits + float targets; `pos_weight`
  for imbalance. `nn.MSELoss()` — match shapes exactly.
- Optimizers: `torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)`; param
  groups to give different LR/decay to different parts (`no decay for bias/LayerNorm`).
- Schedulers: `scheduler.step()` after `optimizer.step()` (per step or per epoch depending
  on the scheduler); `OneCycleLR`, `CosineAnnealingLR`, `LambdaLR` for warmup.

## 5. Data

```python
from torch.utils.data import Dataset, DataLoader

class MyData(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(MyData(X, y), batch_size=64, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True, persistent_workers=True)
for xb, yb in loader: ...
```

- `num_workers` = subprocesses doing `__getitem__` in parallel (decoding images,
  tokenizing). If the GPU waits on data ("GPU util 30%"), raise workers, move work
  offline (pre-tokenize, pre-resize), or use GPU decoding (DALI).
- `pin_memory=True` puts batches in page-locked host memory so host→GPU copies are
  asynchronous DMA; pair with `.to(device, non_blocking=True)`.
- `collate_fn` turns a list of samples into a batch (padding variable-length sequences).
- `shuffle=True` for train, `False` for eval. `drop_last=True` avoids a tiny final batch
  (BatchNorm hates it; throughput too).
- `IterableDataset` for streaming / huge data; `DistributedSampler` for DDP (call
  `sampler.set_epoch(epoch)` or every epoch has the same order).
- **Worker RNG gotcha**: workers are forked copies; a NumPy `Generator` created in the
  dataset's `__init__` is identical in every worker → identical "random" augmentations
  per worker. Seed per worker with `worker_init_fn` (torch seeds its own and NumPy's
  global RNG per worker since 1.9, but not generator objects you created).

## 6. The canonical training loop (memorize the order)

```python
model.train()
for epoch in range(epochs):
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)     # 1. clear old grads
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # optional AMP
            logits = model(xb)                    # 2. forward
            loss = criterion(logits, yb)          # 3. loss
        loss.backward()                           # 4. backward (grads into .grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5. clip
        optimizer.step()                          # 6. update
        scheduler.step()                          # 7. LR schedule
        running += loss.item()                    # (sync point; fine occasionally)

    model.eval()
    with torch.no_grad():
        for xb, yb in val_loader: ...
    model.train()
```

Checkpointing: save `model.state_dict()` (not the model object), plus optimizer state,
scheduler, epoch, and RNG states if you need exact resumption. Load with
`torch.load(path, map_location=device, weights_only=True)`; `load_state_dict(strict=False)`
when heads differ (fine-tuning).

## 7. Mixed precision

- `torch.autocast` runs matmuls/convs in bf16/fp16 and keeps reductions, softmax, losses,
  norms in fp32. Weights stay fp32 (master copy) — updates are applied in fp32.
- **fp16** needs `torch.amp.GradScaler("cuda")` (scale the loss up so small gradients don't
  underflow to zero, unscale before clipping/stepping, skip steps with inf). **bf16** needs
  no scaler (fp32's range), so on Ampere+ it's the default.
- `torch.backends.cuda.matmul.allow_tf32 = True` (or `set_float32_matmul_precision("high")`)
  makes fp32 matmuls use TF32 Tensor Cores — ~8× faster, slightly less precise; fine for
  training.

## 8. Performance (what the senior people check)

1. **Is the GPU busy?** `nvidia-smi` util is crude; use `torch.profiler` / Nsight Systems.
   Low util → data loading, Python overhead, host–device syncs.
2. **Avoid syncs in the hot loop**: `.item()`, `.cpu()`, `print(tensor)`, `if tensor > 0`,
   `tensor.nonzero()` all block until the GPU catches up. Log every N steps.
3. **Bigger batches, fewer kernels**: GPUs are throughput machines; tiny ops are launch-bound.
4. **`torch.compile(model)`**: captures the Python into a graph (TorchDynamo), fuses
   elementwise ops, generates Triton kernels (Inductor). 1.3–2× typical. Costs compile time;
   **recompiles on new shapes** (pad to fixed shapes or mark dynamic); graph breaks on
   unsupported Python.
5. **Fused optimizers** (`AdamW(..., fused=True)`), `channels_last` memory format for CNNs,
   `cudnn.benchmark=True` (autotunes conv algorithms; non-deterministic), FlashAttention via
   `F.scaled_dot_product_attention`.
6. **Memory**: `torch.cuda.max_memory_allocated()`, `memory_summary()`; activation
   checkpointing (`torch.utils.checkpoint`); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   for fragmentation. The caching allocator holds memory ("reserved") — `nvidia-smi` shows
   reserved, not live; `empty_cache()` rarely fixes a real OOM.
7. **Timing** GPU code needs `torch.cuda.synchronize()` (or CUDA events) around the
   region, plus warmup iterations.

## 9. Distributed training

- **`DataParallel`** (single process, replicates per batch, GIL-bound) — deprecated in
  practice. Use **`DistributedDataParallel` (DDP)**: one process per GPU, each with a full
  model copy and a shard of the data; gradients are **all-reduced** (averaged) across GPUs
  during `backward()`, overlapped with computation in buckets. Launch with `torchrun
  --nproc_per_node=8`. Needs `DistributedSampler`. Effective batch = per-GPU batch × GPUs
  → rescale LR.
- **FSDP / ZeRO** (DeepSpeed): shard parameters, gradients, and optimizer states across
  GPUs instead of replicating; all-gather weights just-in-time per layer. Required when
  the model + optimizer states don't fit on one GPU (≥ ~1–3B params on 80 GB).
- **Tensor parallel** (split individual matmuls across GPUs; needs fast NVLink; used within
  a node), **pipeline parallel** (split layers across GPUs; micro-batches to fill the
  bubble), **expert parallel** (MoE). Large LLM training combines all (3D/4D parallelism).
- Common failures: hangs because ranks call different collectives (e.g. one rank skips a
  step on an uneven batch), `find_unused_parameters=True` needed when some params get no
  gradient, BatchNorm stats per rank (SyncBN), evaluation only on rank 0 but every rank
  must hit the barrier, NCCL timeouts from a straggler, "works on 1 GPU not on 8" = LR
  scaling.

## 10. Reproducibility and deployment

- `torch.manual_seed`, `np.random.seed`, `random.seed`; `torch.use_deterministic_algorithms(True)`
  (some ops error or get slow); `cudnn.deterministic=True, benchmark=False`; seed data
  workers; fix data order. Even then, different GPUs/drivers differ in the last bits.
- Export: `torch.export` / ONNX → TensorRT / ONNX Runtime for latency; TorchScript is legacy.
  Servers: TorchServe, NVIDIA Triton Inference Server, vLLM (LLMs), Ray Serve, BentoML.
- Quantization: post-training dynamic (weights int8), static (needs calibration data), QAT
  (quantization-aware training); for LLMs GPTQ/AWQ (int4 weight-only).

## 11. The bug catalog (this is what "experience" looks like)

1. Forgot `optimizer.zero_grad()` → gradients accumulate, training unstable.
2. Forgot `model.eval()` / `torch.no_grad()` at evaluation → wrong BN/dropout behavior, OOM.
3. Softmax before `CrossEntropyLoss` (double softmax) → trains slowly, poorly.
4. `MSELoss(pred (N,1), target (N,))` → broadcasts to `(N,N)`; silent, wrong gradient.
5. Labels as float / one-hot into `CrossEntropyLoss` → dtype / shape error (or wrong semantics).
6. Modules in a Python list → never trained, never moved to GPU, never saved.
7. `view()` on a permuted tensor → error; use `reshape` or `.contiguous()`.
8. In-place ops on tensors needed for backward.
9. Storing `loss` instead of `loss.item()` → memory leak; `.item()` every step → slowness.
10. Different preprocessing/normalization at train vs eval; tokenizer version mismatch.
11. `scheduler.step()` never called, or called before `optimizer.step()` (warning).
12. Loading a checkpoint without `map_location` on a machine with fewer GPUs.
13. Data augmentation applied to validation data; `shuffle=True` on a sequential eval
    whose ordering you later rely on.
14. `DistributedSampler` without `set_epoch` → same order every epoch.
15. `torch.compile` recompiling every step because batch shapes vary → pad.
16. Non-contiguous `permute` + `.numpy()` surprises; `tensor.T` deprecated for >2D.
17. Mixing devices (`model` on GPU, batch on CPU) → "expected all tensors on the same device".
18. Initializing an embedding table of 50M rows on the CPU in float64 by accident.

## Follow-up chains

- "Walk me through a training step." → "Why `zero_grad`?" → "When would you *not* zero
  the grads?" → *gradient accumulation across micro-batches.*
- "Difference between `model.eval()` and `torch.no_grad()`?" → *mode flags vs. graph
  recording; need both.*
- "What does `torch.compile` do?" → "Why can it be slower?" → *compile time,
  recompilations on dynamic shapes, graph breaks.*
- "DDP vs FSDP?" → "When would DDP OOM but FSDP fit?" → *optimizer states + grads + params
  replicated per GPU in DDP; FSDP shards them: 16 bytes/param with Adam mixed precision.*
- "GPU utilization is 35%. Diagnose." → *profile; check dataloader wait time, number of
  workers, syncs in loop, batch size, small kernels → compile/CUDA graphs.*
