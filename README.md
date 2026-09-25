# Lab 3: Optimizing tiled GEMM and a fused linear layer

Learn how tile shape, GPU scheduling, memory reuse, and epilogue fusion affect ML inference. The starting GEMM is already tiled: Triton makes a scalar-versus-tiled exercise a poor optimization baseline. Your task is to improve a sensible blocked implementation and explain when each change helps.

This is an inference-only lab. Run FP16 Triton kernels on a full A100, H100, or H200 GPU using the course CUDA environment. Record the GPU model and software versions, and compare implementations on the same GPU.

## Setup and first run

From this repository's root, in the course CUDA environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
make test
make bench
make demo
```

`make test` checks the provided baseline and helpers. CUDA tests skip on a CPU-only machine: a green CPU run does not validate a GPU kernel. `make test-student` also exercises task C and is expected to fail with `NotImplementedError` until that task is complete. The optimized entry point initially delegates to the baseline and passes output tests; passing tests alone does not complete task B.

Use the same software environment for every comparison.

## Inputs and expected behavior

- A: contiguous FP16 CUDA tensor `(M,K)`. B: FP16 CUDA `(K,N)`, either contiguous or a transposed contiguous view. Both are on the same device.
- Positive dimensions; arbitrary M/N/K tails must work. FP32 accumulation and FP16 GEMM output.
- Fresh output; no input mutation. Rank/shape/device/dtype/layout errors must be explicit.
- Fused linear accepts X `(M,K)`, weight_t `(K,N)` (including `weight.t()`), and optional contiguous FP16 bias `(N,)`. Accumulate, add bias, and optionally apply ReLU in FP32; cast only at the final store.
- Required computation uses Triton. PyTorch may allocate tensors and inspect metadata; no library GEMM fallback in student kernels.
- Zero dimensions, arbitrary strided A, BF16/FP32 inputs, and backward propagation are outside the core task.

## Task A — Understand and test the blocked baseline

Read `src/gemm_lab/kernels/gemm_tiled.py`. Each program computes one `(BM,BN)` output tile and accumulates over K tiles. The default configuration is `(64,64,32)`, four warps, one stage.

Keep this file and the baseline dispatch unchanged as an experimental control. In your report:

1. Derive pointer addresses for A and B and explain broadcasting shapes.
2. Explain separate M/N/K masks, including why masking only output stores is insufficient.
3. Add one public test with simultaneous tails and a transposed B view. Run correctness on your GPU before timing.
4. Derive useful FLOPs and minimum input bytes for one output tile. Explain why increasing BK alone does not improve the ideal input-reuse ratio.

## Task B — Optimize across shapes

Edit `src/gemm_lab/kernels/gemm_optimized.py`. Start from the blocked algorithm, build your own kernel with grouped program ordering, and investigate at most 12 launch configurations (tile sizes, warps, stages). Handle the final incomplete group correctly. Dispatch/autotuning may depend on dimensions, strides, dtype, and GPU metadata, never input values or benchmark identity.

Use the following bounded experiment set; tuples are `(M,N,K)`:

| Family | Shapes |
|---|---|
| Square | `(256,256,256)`, `(1024,1024,1024)` |
| Tall or wide | `(2048,128,512)`, `(128,2048,512)` |
| Small-M inference | `(1,1024,1024)`, `(32,1024,1024)` |
| Irregular | `(65,97,33)`, `(511,769,257)` |

Compare at least three valid configurations and isolate program ordering in a controlled ablation. Discuss a neutral or negative result; do not assume grouping or larger tiles always help. Explain how your chosen configurations balance tail work, parallelism, reuse, and resource pressure. Keep first-call compilation/tuning cost separate from warmed timing.

```bash
python scripts/bench_gemm.py --m 511 --n 769 --k 257 --transpose-b
```

The default run compares torch, baseline, and optimized GEMM on the same tensors and checks outputs before timing. No speedup requirement is implied by these shapes.

## Task C — Fuse linear, bias, and optional ReLU

Implement `fused_linear_relu` in `src/gemm_lab/kernels/gemm_fused.py` with one Triton compute kernel. Handle bias present/absent and ReLU on/off. Read a transposed weight view directly; do not copy weights every forward call. Do not store a separate GEMM output or call PyTorch bias/ReLU on the result.

```bash
make test-student
python scripts/bench_gemm.py --workload linear --m 32 --n 1024 --k 1024 --transpose-b
python scripts/bench_gemm.py --workload linear --m 512 --n 512 --k 512 --no-relu
python scripts/run_mlp_demo.py --fused
```

The unfused baseline stores a FP32 GEMM intermediate and then performs the epilogue so it has the same rounding behavior as fusion. PyTorch linear is a practical comparator with slightly different rounding. The public FP16 checks use bounded inputs and `atol=rtol=0.02`; these are development checks, not a guarantee for arbitrary input magnitudes.

The MLP demo uses fixed synthetic inputs and shared weights, verifies output agreement, and measures a full batch every iteration. It does not test trained-model accuracy.

## Timing and profiling

`bench_gemm.py` writes append-only JSONL rows with device metadata, inputs, repetitions, and raw repeated batch timings. Its metric is **synchronized wrapper latency**, including dispatch and allocations, not isolated kernel duration. Compile and correctness-check before warmup. Read medians together with the min/max spread; rerun if the GPU was shared or unstable. Compare implementations within the same GPU allocation. Do not divide a baseline from one GPU by a candidate from another.

Use Nsight Compute for diagnostic evidence on the course GPU cluster:

```bash
python scripts/profile_gemm.py --case optimized --m 1024 --n 1024 --k 1024
```

Profile results must not replace normal timing runs. The short capture includes only a few launches; select your GEMM kernel rather than tensor-initialization kernels.

## Deliverables

Submit changed student kernel files, your added tests, `report.md`, and raw JSONL results. In the report include environment/allocation details, configuration choices, one tile-size ablation, one ordering ablation, fusion results, and a hardware explanation of a negative result. Distinguish measurements from hypotheses. Cite adapted implementations and tools under the course policy. Grading policy will be supplied separately.
