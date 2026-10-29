#!/usr/bin/env python3
"""Record the current interpreter, allocation, CUDA and data readiness. Never train."""
import argparse
import datetime as dt
import glob
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88"


def command(argv, cwd=None, timeout=8):
    try:
        p = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return {"argv": argv, "returncode": p.returncode, "output": p.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "error": str(exc)}


def collect():
    report = {
        "checked_at": dt.datetime.now().astimezone().isoformat(),
        "root": str(ROOT), "hostname": socket.gethostname(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(),
        "allocation": {key: os.getenv(key) for key in
                       ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST",
                        "CUDA_VISIBLE_DEVICES")},
        "gpu_device_nodes": glob.glob("/dev/nvidia*"),
        "nvidia_smi": command(["nvidia-smi"]),
        "dependencies": {}, "blockers": [],
    }
    driver = Path("/proc/driver/nvidia/version")
    report["kernel_driver"] = driver.read_text() if driver.is_file() else None
    modules = {"torch": "torch", "torchmetrics": "torchmetrics", "hydra-core": "hydra",
               "omegaconf": "omegaconf", "numpy": "numpy", "scipy": "scipy",
               "scikit-learn": "sklearn", "tqdm": "tqdm", "betacal": "betacal"}
    for package, module in modules.items():
        try:
            report["dependencies"][package] = {
                "version": metadata.version(package),
                "import_spec_found": importlib.util.find_spec(module) is not None}
            if not report["dependencies"][package]["import_spec_found"]:
                report["blockers"].append("Dependency has metadata but cannot be found: " + package)
        except (metadata.PackageNotFoundError, ImportError) as exc:
            report["dependencies"][package] = {"missing": True, "error": str(exc)}
            report["blockers"].append("Missing dependency: " + package)
    if not os.getenv("SLURM_JOB_ID"):
        report["blockers"].append("No Slurm allocation; do not run on a login node")
    else:
        report["slurm_job"] = command(["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]])
    try:
        import torch
        report["cuda"] = {"torch_version": torch.__version__, "build_version": torch.version.cuda,
                          "available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()}
        if not os.getenv("SLURM_JOB_ID"):
            raise RuntimeError("No Slurm allocation; CUDA allocation/backward diagnostic skipped")
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        x = torch.tensor([1., 2.], device="cuda:0", requires_grad=True)
        x.square().sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(x.grad).all().item()
        report["cuda"].update(device="cuda:0", name=torch.cuda.get_device_name(0),
                              allocation_and_backward="passed", diagnostic_gradient=x.grad.cpu().tolist())
    except Exception as exc:
        report["cuda_error"] = repr(exc)
        report["blockers"].append("CUDA allocation/backward diagnostic did not pass")
    upstream = ROOT / "code/upstream"
    if (upstream / ".git").exists():
        report["upstream_head"] = command(["git", "rev-parse", "HEAD"], cwd=upstream)
        report["upstream_status"] = command(["git", "status", "--porcelain"], cwd=upstream)
        if report["upstream_head"].get("output", "").strip() != COMMIT:
            report["blockers"].append("Upstream commit differs from required baseline")
        if report["upstream_status"].get("returncode") != 0 or report["upstream_status"].get("output", "").strip():
            report["blockers"].append("Upstream worktree is dirty or could not be checked")
        report["candidate_data_files"] = [str(p.relative_to(ROOT)) for p in upstream.glob("datasets/*cifar10*resnet152*.pkl")]
        report["data_note"] = "Files listed only; run.py must validate actual config, hashes, shapes and splits."
    else:
        report["blockers"].append("Pinned upstream checkout and dataset caches are absent")
    report["status"] = "blocked" if report["blockers"] else "preflight_passed_not_experiment_success"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = args.output or ROOT / "outputs/preflight" / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    if out.exists():
        parser.error("Refusing to overwrite an existing diagnostic record")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = collect()
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "record": str(out), "blockers": report["blockers"]}, indent=2))
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
