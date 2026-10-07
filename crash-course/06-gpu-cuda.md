# 06 · GPUs and CUDA (60 minutes)

Run `code/09_gpu_arithmetic.py` with this module: it computes the numbers below (roofline,
memory budgets, KV-cache size) and demonstrates memory-bound vs compute-bound behaviour on
your CPU, which follows the same physics.

## 1. CPU vs GPU

- A **CPU** has a few (8–64) big cores built for *latency*: deep pipelines, branch
  prediction, large caches. It runs one thread per core very fast.
- A **GPU** has thousands of small cores built for *throughput*. It runs tens of thousands
  of threads at once and hides memory latency by **switching between threads**, not by
  caching. The programming model is **SIMT** (single instruction, multiple threads): groups
  of 32 threads (a **warp**) execute the same instruction in lockstep on different data.
  `if/else` that splits a warp runs both branches serially (**divergence**).
- Deep learning is mostly dense linear algebra on big arrays — the exact workload GPUs
  were built for. A single H100 does ~1,000 TFLOPS in bf16; a server CPU does ~1–2 TFLOPS.

## 2. NVIDIA hardware model (know the words)

- The chip is a set of **Streaming Multiprocessors (SMs)** — 108 on A100, 132 on H100.
- Each SM has: fp32/int **CUDA cores**, **Tensor Cores** (do a small matrix
  multiply-accumulate, e.g. 4×4 or larger tiles, in one instruction in fp16/bf16/tf32/fp8/int8
  — this is where the TFLOPS come from), a **register file**, and **shared memory / L1**
  (~228 KB usable on H100) that a thread block can use as a programmer-managed cache.
- **Execution hierarchy**: a **kernel** is a function run by a **grid** of **thread
  blocks**; each block (≤1024 threads) is scheduled onto one SM; threads in a block can
  share shared memory and synchronize; blocks are independent.
- **Memory hierarchy** (fastest → slowest, smallest → largest):
  registers (per thread) → shared memory/L1 (per SM, ~TB/s aggregate) → L2 (50 MB on H100)
  → **HBM "global memory"** (80 GB on H100, **~3.35 TB/s**) → NVLink to other GPUs
  (900 GB/s aggregate on H100) → PCIe to the host (~64 GB/s per direction Gen5 x16) →
  network (InfiniBand ~400 Gb/s per link).
- The practical lesson: **moving data costs more than computing on it.** Host↔device
  copies are ~50× slower than HBM; keep data resident on the GPU; use pinned memory and
  async copies; batch small transfers.

## 3. Compute-bound vs memory-bound: the roofline (the key mental model)

For any kernel compute **arithmetic intensity** `I = FLOPs / bytes moved from HBM`. The
GPU can deliver at most `min(peak_FLOPS, I × bandwidth)`. The crossover ("ridge point")
is `peak_FLOPS / bandwidth` ≈ 989e12 / 3.35e12 ≈ **295 FLOP/byte on H100** (~156 on A100).

- A **matmul** `(M×K)(K×N)` does `2MNK` FLOPs and moves `2(MK + KN + MN)` bytes in bf16.
  For M=N=K=4096: 137 GFLOP vs 100 MB → I ≈ 1365 → **compute-bound**. Big GEMMs are the
  only thing that gets near peak.
- **Elementwise ops** (add, GELU, dropout), **normalization**, **softmax**, and **naive
  attention** read and write every element once with ~1–10 FLOPs each → I ≈ 1 →
  **memory-bound**; they run at bandwidth speed regardless of how many TFLOPS you have.
- **Consequences**: (1) fuse memory-bound ops into one kernel so data is read once
  (`torch.compile`, FlashAttention, fused optimizers, fused LayerNorm); (2) reduced precision
  helps memory-bound ops because bytes shrink; (3) a model with many small layers can be
  slow even with tiny FLOPs; (4) **LLM decoding is memory-bound**: each new token re-reads
  all weights (14 GB for a 7B bf16 model) to do only ~14 GFLOP → batching requests
  together is nearly free until you hit the ridge point.
- **Kernel launch overhead** (~5–10 µs each) is a third regime: tiny ops are
  **launch-bound**. Fix with bigger batches, fusion, or **CUDA Graphs** (replay a recorded
  sequence of kernels).

## 4. Precision formats

| Format | Exp bits | Mantissa | Range | Use |
|---|---|---|---|---|
| fp32 | 8 | 23 | ~1e38 | master weights, optimizer states, reductions |
| tf32 | 8 | 10 | ~1e38 | matmul inputs on Tensor Cores (fp32 "for free") |
| fp16 | 5 | 10 | 65,504 max | AMP on older GPUs; needs loss scaling |
| bf16 | 8 | 7 | ~1e38 | default AMP for training on Ampere+ |
| fp8 (E4M3/E5M2) | 4/5 | 3/2 | small | Hopper+ matmuls with per-tensor scaling |
| int8 / int4 | — | — | — | inference quantization (weights, sometimes activations) |

**Mixed precision** = compute matmuls in bf16/fp16 for Tensor Core speed and halved
memory traffic, keep an **fp32 master copy** of weights (so tiny updates don't vanish), and
keep numerically sensitive ops (softmax, loss, norm statistics) in fp32. With fp16, the
**GradScaler** multiplies the loss by a large factor so small gradients don't underflow,
then unscales; it skips the step if it sees inf.

## 5. Memory accounting (interviewers ask you to do this on a whiteboard)

**Training** with Adam in mixed precision, per parameter: bf16 weight (2) + bf16 grad (2) +
fp32 master weight (4) + Adam m (4) + Adam v (4) = **16 bytes/param**.
- 7B model → 112 GB **before activations** → does not fit on one 80 GB GPU → shard with
  FSDP/ZeRO or use LoRA (only adapter params get optimizer states).
- **Activations** scale with `batch × seq_len × hidden × layers` (and `seq²` per head for
  naive attention). They usually dominate for small/medium models and long sequences.
  Reduce with smaller micro-batches + gradient accumulation, **activation (gradient)
  checkpointing** (store only block inputs, recompute in backward: ~30% more compute),
  FlashAttention, shorter sequences.

**Inference**: weights `params × bytes` (7B bf16 = 14 GB; int4 ≈ 3.5 GB) + **KV cache**
`2 × layers × kv_heads × head_dim × bytes` per token — LLaMA-2-7B: 2×32×32×128×2 B =
**0.5 MB/token** → 4k-token context ≈ 2 GB per sequence; a batch of 32 such requests
needs 64 GB just for cache. That is why GQA (fewer KV heads), paged KV cache (vLLM), and
KV quantization exist.

## 6. Kernels, streams, and asynchrony

- A **kernel** is launched by the host and runs asynchronously; Python continues
  immediately. Operations on the same **stream** execute in order; different streams can
  overlap (e.g. copy next batch while computing this one).
- Because launches are async, **errors surface later** at a sync point with confusing
  stack traces; `CUDA_LAUNCH_BLOCKING=1` serializes for debugging; timings need
  `torch.cuda.synchronize()` or CUDA events.
- **"device-side assert triggered"** almost always means an out-of-range index
  (label ≥ num_classes, embedding index ≥ table size). Reproduce on CPU for the real
  message.
- Libraries you call without knowing it: **cuBLAS** (GEMM), **cuDNN** (conv, RNN, some
  attention), **NCCL** (multi-GPU collectives: all-reduce, all-gather, reduce-scatter, ring
  and tree algorithms), **CUTLASS** (templated GEMM building blocks), **Triton** (Python DSL
  for writing fused kernels; what `torch.compile` emits), cuSPARSE, cuRAND.
- **Driver vs toolkit**: PyTorch wheels bundle their CUDA runtime; the machine's **driver**
  must be at least as new as that runtime. "CUDA driver version is insufficient" means
  update the driver (or install the wheel built for an older CUDA).

## 7. FlashAttention (explain it in 30 seconds)

Naive attention computes `S = QKᵀ` (n×n), softmax, then `PV`, writing the n×n matrices to
HBM and reading them back — memory-bound and O(n²) memory. FlashAttention tiles Q, K, V
into blocks that fit in SRAM, computes the softmax incrementally with the online
(running-max) trick, and writes only the final output; backward recomputes the blocks
instead of storing them. Result: **exact** attention, memory linear in n, 2–4× faster. It
is the default behind `F.scaled_dot_product_attention` when shapes/dtypes allow.

## 8. Utilization: how to tell if you are using the hardware

- **MFU (model FLOPs utilization)** = achieved useful FLOP/s ÷ peak. LLM training at 40–55%
  is good; small models often sit at 10–20% because of memory-bound ops and overhead.
- `nvidia-smi` "GPU-Util" = fraction of time *any* kernel was running; a kernel using 1 SM
  counts as 100%. Use `torch.profiler` (kernel timeline, dataloader waits, CPU/GPU gap),
  Nsight Systems/Compute, or DCGM SM-activity metrics.
- Tensor Cores want matrix dims as multiples of 8 (bf16) / 16 (int8) — pad vocab sizes and
  hidden dims (**tile quantization**). The number of thread blocks should divide evenly over
  SMs (**wave quantization**).
- Occupancy = resident warps per SM vs maximum; limited by registers and shared memory per
  block; higher occupancy hides latency better (but isn't everything).

## 9. Diagnosing the common performance problems

| Symptom | Cause | Fix |
|---|---|---|
| GPU util spiky, CPU cores at 100% | data loading / augmentation bound | more workers, pre-process offline, DALI, faster storage, smaller images |
| GPU util low, CPU low | host–device syncs or launch-bound Python | remove `.item()`/`.cpu()` in loop, bigger batch, `torch.compile`, CUDA graphs |
| High util, low MFU | memory-bound ops dominate | fusion, bf16, FlashAttention, bigger hidden dims |
| Step time grows over time | memory fragmentation / leak / `torch.compile` recompiles | check `memory_allocated()`, expandable segments, fix dynamic shapes |
| 8 GPUs barely faster than 1 | communication-bound or dataloader shared | check all-reduce time in profile, overlap, bigger per-GPU batch, NVLink topology, per-rank loaders |
| OOM at step 1 | model + optimizer too big | sharding, mixed precision, LoRA, smaller batch |
| OOM after hours | a rare long sequence / big batch; eval without `no_grad`; leak | cap seq length / sort by length, `no_grad`, `.item()` |

## 10. Multi-GPU communication

- **All-reduce** of G bytes across n GPUs with a ring moves `2(n−1)/n × G` per GPU — it's
  bandwidth-bound, so DDP overlaps gradient all-reduces with the backward pass in
  **buckets** (default 25 MB). Within a node NVLink (~900 GB/s) is fast; across nodes
  InfiniBand/RoCE is 10–20× slower, which is why tensor parallelism stays inside a node
  and data/pipeline parallelism goes across nodes.
- **Collective hangs** happen when ranks disagree on what to call (uneven data, conditional
  branches). Timeouts default to 10–30 minutes — set `NCCL_DEBUG=INFO` and check every rank.

## 11. Inference-specific knowledge

- **Latency vs throughput**: batching increases throughput and per-request latency.
  Metrics for LLMs: **TTFT** (time to first token; prefill, compute-bound),
  **TPOT / inter-token latency** (decode, memory-bound), tokens/s/GPU.
- **Continuous batching** (admit new requests mid-generation), **PagedAttention** (KV cache
  in fixed-size blocks → no fragmentation, ~2–4× more concurrent sequences),
  **speculative decoding** (a small draft model proposes k tokens; the big model verifies
  all k in one forward pass; exact same output distribution; 2–3× faster decode),
  **quantization** (int8/int4 weights help the memory-bound decode phase almost linearly),
  **tensor parallel** across GPUs for lower latency on big models, **prefix caching** for
  shared prompts.
- Frameworks: vLLM, TensorRT-LLM, SGLang, Hugging Face TGI; for non-LLM models NVIDIA
  Triton Inference Server with TensorRT engines, ONNX Runtime, or just TorchServe.

## 12. What a CUDA kernel looks like (conceptual; you won't be asked to write one)

```c
__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index
    if (i < n) c[i] = a[i] + b[i];                   // one thread per element
}
add<<<(n + 255) / 256, 256>>>(a, b, c, n);           // grid of blocks of 256 threads
```

Ideas worth knowing by name: **coalesced memory access** (adjacent threads read adjacent
addresses → one wide transaction), **shared-memory tiling** for matmul (load a tile of A and
B once, reuse from SRAM many times — that is how intensity is raised), **bank conflicts**,
**occupancy**. **Triton** lets you write such kernels in Python at the block level and is
what most ML engineers use when they need a custom fused op.

## 13. Beyond NVIDIA (one line each)

AMD (ROCm, MI300X with 192 GB HBM — software ecosystem is the gap), Google **TPUs** (systolic
arrays, XLA/JAX, pods with fast interconnect), Apple MPS, AWS Trainium/Inferentia, Intel
Gaudi. NVIDIA's moat is CUDA + cuDNN/NCCL + every library being tested on it first.

## Experience signals

- "Before I optimize, I profile — and the first thing I look for is whether the GPU is
  waiting on the dataloader."
- "Decode is memory-bound; that's why int4 weight quantization speeds it up, and why
  batching is nearly free until the ridge point."
- "16 bytes per parameter for Adam in mixed precision — I'd compute whether it fits before
  requesting GPUs."
- "Device-side assert is almost always an index out of range; I'd rerun on CPU to get the
  real error."
- "`nvidia-smi` saying 100% doesn't mean the SMs are busy."

## Follow-up chains

- "How does a GPU speed up training?" → "What's a warp? What's the memory hierarchy?" →
  "Why is LayerNorm slow relative to its FLOPs?" → *memory-bound; fuse.*
- "What's mixed precision?" → "Why keep fp32 master weights?" → "bf16 vs fp16?" →
  "Where do NaNs come from in fp16?" → *overflow in attention logits / loss; ε underflow.*
- "You have a 13B model and 8×A100-40GB; can you fine-tune it?" → *full FT with Adam needs
  ~208 GB + activations: FSDP across 8 GPUs (320 GB total) works with checkpointing; or
  LoRA/QLoRA on far less.*
- "Explain FlashAttention." → "Is it approximate?" → *no, exact; IO-aware.*
- "Why is my multi-GPU job not scaling?" → *communication, dataloader, synchronization
  points, stragglers; measure the all-reduce fraction.*
