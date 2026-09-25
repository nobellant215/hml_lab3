"""Profile one implementation on one shape; timings from ncu are diagnostic."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case", choices=["baseline", "optimized", "fused"], default="baseline"
    )
    p.add_argument("--m", type=int, default=512)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--out", default="profiles/gemm")
    args = p.parse_args()
    ncu = shutil.which("ncu")
    if not ncu:
        p.error("Nsight Compute (ncu) is not in PATH")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ncu,
            "--set",
            "full",
            "--launch-count",
            "3",
            "--kernel-name",
            "regex:.*gemm.*|.*fused.*",
            "--export",
            args.out,
            sys.executable,
            "scripts/bench_gemm.py",
            "--cases",
            args.case,
            "--workload",
            "linear" if args.case == "fused" else "gemm",
            "--m",
            str(args.m),
            "--n",
            str(args.n),
            "--k",
            str(args.k),
            "--warmup",
            "1",
            "--iters",
            "1",
            "--repeats",
            "2",
            "--out",
            args.out + ".diagnostic.jsonl",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
