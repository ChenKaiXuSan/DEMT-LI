"""Model-only forward-pass latency (Table VII).

FP32, batch size 1, one jointly generated three-phase sequence per forward pass;
10 warm-up runs, then 100 runs with CUDA synchronization before and after each pass.
Data loading, host-device transfer, pre/post-processing and I/O are excluded. The
input includes the full three-phase CT sequence, as in the reported evaluation.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .network import build_model, deterministic, load_checkpoint, predict, tensors
from .protocol import load_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--patient", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    deterministic()
    device = torch.device(args.device)
    model = build_model("image_2d")
    model.load_state_dict(load_checkpoint(args.checkpoint)["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    case = load_cases(args.data, [args.patient])[args.patient]
    ct, surrogate, rpm, _ = tensors(case, device, "image_2d")
    times = []
    with torch.inference_mode():
        for _ in range(10):
            predict(model, "image_2d", ct, surrogate, rpm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            predict(model, "image_2d", ct, surrogate, rpm)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    median = float(np.median(times))
    result = dict(median_ms=median, q25_ms=float(np.quantile(times, .25)),
                  q75_ms=float(np.quantile(times, .75)), p95_ms=float(np.quantile(times, .95)),
                  sequences_per_second=1000 / median, joint_volumes_per_second=3000 / median,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024 ** 3,
                  parameters=sum(p.numel() for p in model.parameters()),
                  python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(device), ct_input_shape=list(ct.shape),
                  warmup=10, repetitions=100, precision="FP32", samples_ms=times)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"median {median:.2f} ms per three-phase sequence on {result['gpu']}")


if __name__ == "__main__":
    main()
