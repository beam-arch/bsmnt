"""The whiteboard arithmetic of GPUs: roofline, memory budgets, KV cache, training
cost -- plus a live demonstration that elementwise ops are memory-bound and matmuls
are compute-bound (true on CPU too; the ratios are what matter).

Run:  python code/09_gpu_arithmetic.py
"""
import time

import torch

GB = 1e9   # decimal GB, matching how vendors quote memory

GPUS = {  # dense bf16 TFLOPS (no sparsity), HBM TB/s, HBM GB
    "A100-80GB": (312, 2.0, 80),
    "H100-SXM": (989, 3.35, 80),
}

print("=== 1. roofline ridge points ===")
for name, (tflops, tbs, _) in GPUS.items():
    print(f"{name:<10} ridge = {tflops * 1e12 / (tbs * 1e12):6.0f} FLOP/byte "
          f"-> a kernel needs more than that to be compute-bound")


def gemm_intensity(M, N, K, bytes_per=2):
    flops = 2 * M * N * K
    bytes_moved = bytes_per * (M * K + K * N + M * N)
    return flops / bytes_moved


print("\n=== 2. arithmetic intensity (bf16) ===")
for shape in [(16, 4096, 4096), (512, 4096, 4096), (4096, 4096, 4096)]:
    print(f"GEMM {shape}: {gemm_intensity(*shape):7.1f} FLOP/byte  "
          f"{'compute-bound on H100' if gemm_intensity(*shape) > 295 else 'MEMORY-bound (small M = decode-style)'}")
print(f"elementwise GELU: ~{8 / 4:.0f} FLOP/byte -> hopelessly memory-bound -> fuse it")

print("\n=== 3. training memory per parameter (Adam, mixed precision) ===")
per_param = {"bf16 weights": 2, "bf16 grads": 2, "fp32 master": 4, "Adam m": 4, "Adam v": 4}
print("  " + " + ".join(f"{k} {v}B" for k, v in per_param.items()) + f" = {sum(per_param.values())} bytes/param")
for n_params in (1e9, 7e9, 70e9):
    static = n_params * 16 / GB
    print(f"  {n_params / 1e9:>4.0f}B params: {static:7.0f} GB before activations -> "
          f"{'fits one 80GB GPU' if static < 70 else f'needs sharding over >= {int(static // 70) + 1} x 80GB (FSDP/ZeRO) or LoRA'}")

print("\n=== 4. inference memory: weights + KV cache ===")
layers, heads, kv_heads, head_dim = 32, 32, 32, 128      # LLaMA-2-7B
per_token = 2 * layers * kv_heads * head_dim * 2           # K and V, bf16
print(f"  7B bf16 weights: {7e9 * 2 / GB:.0f} GB; int4: {7e9 * 0.5 / GB:.1f} GB")
print(f"  KV cache: {per_token / 2**20:.2f} MB/token -> 4k context = {per_token * 4096 / GB:.1f} GB/sequence, "
      f"batch of 32 = {per_token * 4096 * 32 / GB:.0f} GB")
print(f"  with GQA (8 kv heads): {2 * layers * 8 * head_dim * 2 / 2**20:.3f} MB/token (4x smaller)")

print("\n=== 5. training cost: FLOPs = 6 * params * tokens ===")
n, d = 7e9, 2e12
flops = 6 * n * d
for name, (tflops, _, _) in GPUS.items():
    for n_gpus, mfu in ((8, 0.4), (1024, 0.4)):
        secs = flops / (n_gpus * tflops * 1e12 * mfu)
        print(f"  7B x 2T tokens = {flops:.1e} FLOPs on {n_gpus:>4} x {name} @ {mfu:.0%} MFU: {secs / 86400:8.1f} days")

print("\n=== 6. live: memory-bound vs compute-bound (on this machine) ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))


def bench(fn, iters=5):
    fn()                                            # warmup (allocations, kernel selection)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()                    # GPU work is async: must sync before reading the clock
    return (time.perf_counter() - t0) / iters


n = 2048
A, B = torch.randn(n, n, device=device), torch.randn(n, n, device=device)
t_mm = bench(lambda: A @ B)
gflops = 2 * n**3 / t_mm / 1e9
x = torch.randn(64 * 1024 * 1024 // 4, device=device)     # 64 MB fp32 tensor
t_add = bench(lambda: x + 1.0)
gbps = 2 * x.numel() * 4 / t_add / 1e9                     # read + write
add_gflops = x.numel() / t_add / 1e9
print(f"  device {device}: matmul {n}x{n}: {gflops:7.1f} GFLOP/s  (intensity {gemm_intensity(n, n, n, 4):.0f} FLOP/byte)")
print(f"  elementwise add on 64MB:  {add_gflops:7.1f} GFLOP/s at {gbps:.1f} GB/s   (intensity 0.125 FLOP/byte)")
print(f"  -> the same hardware delivers {gflops / add_gflops:.0f}x more FLOP/s on the matmul: bandwidth, not compute, limits elementwise ops")

# fusion demo: three elementwise passes vs one
y = torch.randn_like(x)
t_three = bench(lambda: ((x * 2.0) + y) * 0.5)             # 3 kernels, 3 round-trips to memory (eager mode)
fused = torch.compile(lambda a, b: ((a * 2.0) + b) * 0.5) if hasattr(torch, "compile") else None
if fused is not None:
    try:
        t_one = bench(lambda: fused(x, y))
        print(f"  3 separate elementwise ops: {t_three * 1e3:.2f} ms   torch.compile fused: {t_one * 1e3:.2f} ms")
    except Exception as e:  # compile needs a C++ toolchain; not essential here
        print(f"  (torch.compile unavailable here: {type(e).__name__}) unfused 3-op chain: {t_three * 1e3:.2f} ms")
print("Done.")
