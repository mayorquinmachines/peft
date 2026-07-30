"""Run the repo's own benchmark (method_comparison/MetaMathQA) for the kappa-LoRA
experiment and emit its result as validation metrics.

This is the repo-native rung of the evidence ladder: the harness, protocol, and
metrics are the ones peft maintainers already publish results for under
method_comparison/MetaMathQA/results/ — so the numbers are directly comparable
to the repo's own corpus. The declared published baseline in
.remyx/validation.yaml is lora--llama-3.2-3B-rank32.json from that corpus
(standard LoRA, same rank, same protocol).
"""
import glob
import json
import os
import subprocess
import sys

HARNESS_DIR = os.path.join("method_comparison", "MetaMathQA")
EXPERIMENT = os.path.join("experiments", "lora", "llama-3.2-3B-rank32-kappa")


def _result_files(harness_dir):
    return {p: os.path.getmtime(p)
            for p in glob.glob(os.path.join(harness_dir, "*results", "*.json"))}


def main() -> int:
    repo_root = os.getcwd()
    harness_dir = os.path.join(repo_root, HARNESS_DIR)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=harness_dir, check=True)
    # The eval pool's GPU driver speaks CUDA 12.8; a default-resolved torch wheel
    # can target a newer CUDA and refuse to initialize. Pin the cu126 build,
    # AFTER the requirements install so nothing re-resolves it away.
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torch==2.7.1",
         "--index-url", "https://download.pytorch.org/whl/cu126"],
        cwd=harness_dir, check=True)

    before = _result_files(harness_dir)
    proc = subprocess.run([sys.executable, "run.py", "-v", EXPERIMENT], cwd=harness_dir)

    after = _result_files(harness_dir)
    new_or_updated = [p for p, mt in after.items() if mt > before.get(p, 0)]
    if proc.returncode != 0 or not new_or_updated:
        print(f"harness exited {proc.returncode}; new result files: {new_or_updated}",
              file=sys.stderr)
        return 1

    result_path = max(new_or_updated, key=after.get)
    print(f"reading harness result: {result_path}", file=sys.stderr)
    with open(result_path) as f:
        info = json.load(f)

    train_info = info.get("train_info") or {}
    rows = train_info.get("metrics") or []
    last = rows[-1] if rows else {}

    metrics = {
        "test_accuracy": last.get("test accuracy"),
        "num_trainable_params": train_info.get("num_trainable_params"),
        "peak_vram_gb": round((train_info.get("accelerator_memory_max") or 0) / 1e9, 2),
        "train_time_s": round(train_info.get("train_time") or 0, 1),
    }
    if metrics["test_accuracy"] is None or metrics["num_trainable_params"] is None:
        print(f"harness result missing declared metrics: {metrics}", file=sys.stderr)
        return 1

    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    sys.argv = sys.argv[:1]  # accept and ignore --variant/--ref/--seed
    sys.exit(main())
