"""Repeated synchronized wrapper timing; includes Python dispatch/allocation."""

import statistics
import time
import torch


def tflops(m, n, k, seconds):
    return 2.0 * m * n * k / (seconds * 1e12)


def measure(fn, *, device, warmup=10, iters=20, repeats=5):
    if warmup < 0 or iters < 1 or repeats < 2:
        raise ValueError("Need warmup >= 0, iters >= 1, repeats >= 2")
    device = torch.device(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        samples = []
        for _ in range(repeats):
            sync()
            start = time.perf_counter()
            for _ in range(iters):
                fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000 / iters)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
        "timing_scope": "synchronized_wrapper",
        "warmup": warmup,
        "iters": iters,
        "repeats": repeats,
    }


def bench_once(fn, a, b, warmup, iters):
    return (
        measure(lambda: fn(a, b), device=a.device, warmup=warmup, iters=iters)[
            "median_ms"
        ]
        / 1000
    )


def environment(device):
    import triton

    p = torch.cuda.get_device_properties(device)
    return {
        "gpu": p.name,
        "device_uuid": str(getattr(p, "uuid", "unavailable")),
        "memory_bytes": p.total_memory,
        "compute_capability": [p.major, p.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
    }
