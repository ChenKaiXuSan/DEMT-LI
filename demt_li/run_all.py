"""Train all 3 methods x 6 folds, one job per GPU at a time, then summarize."""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from .protocol import METHODS, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    (args.output / "logs").mkdir(parents=True, exist_ok=True)
    jobs = queue.Queue()
    for split in folds():
        for method in METHODS:
            jobs.put((method, split["fold"]))
    failures = []

    def worker(gpu):
        while True:
            try:
                method, fold = jobs.get_nowait()
            except queue.Empty:
                return
            command = [sys.executable, "-m", "demt_li.train", "--method", method, "--fold", str(fold),
                       "--data", str(args.data), "--output", str(args.output),
                       "--epochs", str(args.epochs)]
            log = args.output / "logs" / f"{method}_fold_{fold:02d}.log"
            print(f"GPU {gpu}: {method} fold {fold}", flush=True)
            with log.open("w") as stream:
                code = subprocess.call(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu))
            if code:
                failures.append((method, fold))
                print(f"FAILED: {method} fold {fold} (see {log})", flush=True)

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} job(s) failed: {failures}")
    subprocess.check_call([sys.executable, "-m", "demt_li.summarize", "--run", str(args.output),
                           "--data", str(args.data)])


if __name__ == "__main__":
    main()
