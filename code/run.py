#!/usr/bin/env python3
"""Run the pinned calibration code, with paired splits and auditable attempts.

The pinned upstream checkout and an allocated CUDA GPU are required.
The four-condition protocol reproduces the reported fixed-classifier study.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "1fb9c6bb7b98e3960b3a9281ef25e8fcedfb1a88"
METHODS = ("uncalibrated", "ts", "gc_ts", "gc_ts_shuffle")
sys.dont_write_bytecode = True  # Importing upstream must not dirty its checkout.


class Blocked(RuntimeError):
    """A prerequisite was not met; no replacement experiment is permitted."""


class InvalidRun(RuntimeError):
    """A scientific or numerical validity check failed."""


class InterruptedRun(RuntimeError):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Interrupted by signal {signum} ({signal.Signals(signum).name})")


@contextlib.contextmanager
def recordable_signals():
    """SIGKILL and node loss cannot be caught; SIGTERM leaves a final record."""
    def interrupt(signum, frame):
        raise InterruptedRun(signum)

    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_write(path, value):
    """Write a complete JSON record atomically within its reserved run directory."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")
    temporary.replace(path)


def numeric(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Blocked(f"Cannot import module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_level_plan(path):
    """Read this project's intentionally small, flat YAML configuration subset.

    This scheduling read uses only stdlib, so missing ML dependencies can still
    produce a blocked record. OmegaConf independently loads and checks the same
    file before any experiment; arbitrary YAML syntax is deliberately rejected.
    """
    result = {}
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config syntax: {path}:{line_number}")
        key, value = (part.strip() for part in line.split(":", 1))
        if not re.fullmatch(r"[a-z_]+", key) or key in result:
            raise ValueError(f"Invalid or duplicate config key at {path}:{line_number}")
        if value.startswith("[") and value.endswith("]"):
            value = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            if key == "seeds":
                value = [int(item) for item in value]
        elif key in {"num_groups", "num_partitions", "group_optimizer_steps"}:
            value = int(value)
        elif key == "weight_decay":
            value = float(value)
        result[key] = value
    required = {"level", "methods", "seeds", "num_groups", "num_partitions",
                "group_optimizer_steps", "weight_decay", "device"}
    if required - result.keys():
        raise ValueError(f"Missing level config keys: {sorted(required - result.keys())}")
    return result


def reserve_run(level, method, seed):
    parent = ROOT / "outputs/runs"
    parent.mkdir(parents=True, exist_ok=True)
    tag = f"_{level}_{method}_s{seed}_a"
    attempts = []
    for existing in parent.glob(f"*{tag}*"):
        match = re.search(r"_a(\d+)$", existing.name)
        if match:
            attempts.append(int(match.group(1)))
    attempt = max(attempts, default=0) + 1
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = parent / f"{stamp}{tag}{attempt:04d}"
    directory.mkdir(exist_ok=False)
    return directory, attempt


def git_output(upstream, *arguments):
    command = subprocess.run(["git", "-C", str(upstream), *arguments],
                             capture_output=True, text=True, timeout=30)
    if command.returncode:
        raise Blocked(f"Git validation failed: {command.stderr.strip()}")
    return command.stdout.strip()


def validate_checkout(upstream):
    if not (upstream / ".git").exists():
        raise Blocked("Real code/upstream Git checkout is absent; browser text is not a baseline")
    if Path(git_output(upstream, "rev-parse", "--show-toplevel")).resolve() != upstream.resolve():
        raise Blocked("Upstream path is not its own Git checkout root")
    revision = git_output(upstream, "rev-parse", "HEAD")
    if revision != COMMIT:
        raise Blocked(f"Upstream HEAD {revision} differs from required {COMMIT}")
    if git_output(upstream, "status", "--porcelain", "--untracked-files=all"):
        raise Blocked("Upstream worktree is dirty; preserve the fixed baseline")
    required = ["data.py", "utils.py", "evaluate.py", "methods/group_calibration.py",
                "methods/temp_scaling.py", "conf/main.yaml",
                "conf/data/cifar10_resnet152.yaml",
                "conf/method/group_calibration_combine_ts.yaml"]
    for relative in required:
        if not (upstream / relative).is_file():
            raise Blocked(f"Required upstream file absent: {relative}")
    return {"upstream/" + relative: sha256(upstream / relative) for relative in required}


def tensor_hash(tensors):
    """Hash shapes, dtypes and unchanged parameter bytes without drawing RNG."""
    digest = hashlib.sha256()
    for tensor in tensors:
        array = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        import torch
        digest.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_data(parts, torch):
    description = {}
    common_classes, common_features = None, None
    for name, part in parts.items():
        for key in ("features", "logits", "labels"):
            if key not in part or not isinstance(part[key], torch.Tensor):
                raise InvalidRun(f"{name}.{key} is missing or is not a Torch tensor")
            if part[key].device.type != "cpu":
                raise InvalidRun(f"{name}.{key} is not stored on CPU as expected by upstream stage two")
        features, logits, labels = (part[key] for key in ("features", "logits", "labels"))
        if features.ndim != 2 or logits.ndim != 2 or labels.ndim != 1:
            raise InvalidRun(f"Unexpected dimensions in {name}")
        if not (len(features) == len(logits) == len(labels) and len(labels) > 0):
            raise InvalidRun(f"Unequal or zero sample counts in {name}")
        if not features.is_floating_point() or not logits.is_floating_point():
            raise InvalidRun(f"Non-floating features/logits in {name}")
        if labels.dtype != torch.int64:
            raise InvalidRun(f"Labels in {name} must retain upstream-compatible int64 dtype")
        if not torch.isfinite(features).all().item() or not torch.isfinite(logits).all().item():
            raise InvalidRun(f"Non-finite input in {name}")
        if labels.min().item() < 0 or labels.max().item() >= logits.shape[1]:
            raise InvalidRun(f"Label outside class range in {name}")
        if common_classes is None:
            common_classes, common_features = logits.shape[1], features.shape[1]
        if logits.shape[1] != common_classes or features.shape[1] != common_features:
            raise InvalidRun("Class or feature dimensions differ across splits")
        description[name] = {key: {"shape": list(part[key].shape), "dtype": str(part[key].dtype),
                                   "device": str(part[key].device)}
                             for key in ("features", "logits", "labels")}
    return description


class Observer:
    """Observe original optimizers; never replace loss, gradients or parameters."""

    def __init__(self, torch, group_module, ts_module):
        self.torch, self.gc, self.ts = torch, group_module, ts_module
        self.scope = {"kind": "idle"}
        self.partitions = 0
        self.base_calls = 0
        self.initialization_hashes = []
        self.steps = []
        self.groups = []
        self.temperatures = []
        self.invalid_reasons = []
        self.warnings = []
        self.optimizer_parameter_refs = []

    def export(self):
        return {"initialization_hashes": self.initialization_hashes,
                "optimizer_steps": self.steps, "group_counts": self.groups,
                "temperatures": self.temperatures,
                "invalid_reasons": self.invalid_reasons, "warnings": self.warnings}

    def check_temperature(self, value, scope, fallback=False):
        tensor = self.torch.as_tensor(value).detach().cpu()
        finite = bool(self.torch.isfinite(tensor).all().item())
        positive = finite and bool((tensor > 0).all().item())
        self.temperatures.append({**scope, "values": [numeric(x) for x in tensor.reshape(-1).tolist()],
                                  "finite": finite, "positive": positive, "fallback": bool(fallback)})
        if not finite:
            self.invalid_reasons.append(f"Non-finite temperature: {scope}")
        if not positive:
            self.warnings.append(f"Temperature positivity did not pass: {scope}")
        if fallback:
            self.invalid_reasons.append(f"Upstream NaN-temperature fallback was used: {scope}")

    @contextlib.contextmanager
    def instrument(self):
        torch, observer = self.torch, self
        original_step = torch.optim.LBFGS.step
        original_init = self.gc.WNet.__init__
        original_optimize = self.gc.optimize_group_fn

        def observed_init(model, *args, **kwargs):
            original_init(model, *args, **kwargs)
            observer.scope["model"] = model
            observer.initialization_hashes.append(tensor_hash(list(model.parameters())))

        def observed_optimize(*args, **kwargs):
            previous = observer.scope
            index = observer.partitions
            observer.partitions += 1
            observer.scope = {"kind": "group_training", "partition": index}
            try:
                result = original_optimize(*args, **kwargs)
                observer.check_temperature(result[0], {"kind": "temporary", "partition": index})
                if observer.invalid_reasons:
                    raise InvalidRun("; ".join(observer.invalid_reasons))
                return result
            finally:
                observer.scope = previous

        def observed_step(optimizer, closure=None):
            if closure is None:
                raise InvalidRun("Expected LBFGS closure is missing")
            parameters = list(optimizer._params)
            observer.optimizer_parameter_refs.append(parameters)
            before = [parameter.detach().cpu().clone() for parameter in parameters]
            model = observer.scope.get("model")
            scope = {key: value for key, value in observer.scope.items() if key != "model"}
            record = {**scope, "closure_calls": 0, "finite_losses": True,
                      "finite_gradients": True, "missing_gradient": False,
                      "max_gradient_norm": 0.0,
                      "optimizer_defaults": {key: value for key, value in optimizer.defaults.items()
                                             if isinstance(value, (int, float, str, bool, type(None)))},
                      "parameter_devices": [str(parameter.device) for parameter in parameters],
                      "parameter_shapes": [list(parameter.shape) for parameter in parameters],
                      "temperature_before": [numeric(x) for x in before[0].reshape(-1).tolist()]}
            if model is not None:
                model_parameters = list(model.parameters())
                record["model_devices"] = [str(parameter.device) for parameter in model_parameters]
                record["optimizer_has_all_model_parameters"] = (
                    {id(parameter) for parameter in model_parameters}
                    <= {id(parameter) for parameter in parameters})
                record["model_hash_before_step"] = tensor_hash(model_parameters)
                if not record["optimizer_has_all_model_parameters"]:
                    observer.invalid_reasons.append("Runtime optimizer/model parameter identities do not match")
            observer.steps.append(record)

            def observed_closure():
                loss = closure()
                record["closure_calls"] += 1
                loss_finite = bool(torch.isfinite(loss.detach()).all().item())
                record["finite_losses"] = record["finite_losses"] and loss_finite
                record["last_loss"] = numeric(loss.detach().item())
                norm_square = 0.0
                for parameter in parameters:
                    if parameter.grad is None:
                        record["missing_gradient"] = True
                        continue
                    finite = bool(torch.isfinite(parameter.grad).all().item())
                    record["finite_gradients"] = record["finite_gradients"] and finite
                    if finite:
                        norm_square += parameter.grad.detach().double().square().sum().item()
                if math.isfinite(norm_square):
                    record["max_gradient_norm"] = max(record["max_gradient_norm"], math.sqrt(norm_square))
                return loss

            try:
                returned = original_step(optimizer, closure=observed_closure)
                record["status"] = "returned"
                return returned
            except BaseException as exc:
                record["status"] = "raised"
                record["exception"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                after = [parameter.detach().cpu().clone() for parameter in parameters]
                record["temperature_after"] = [numeric(x) for x in after[0].reshape(-1).tolist()]
                record["parameter_delta_l2"] = [numeric((new.double() - old.double()).norm().item())
                                                 for old, new in zip(before, after)]
                record["finite_parameters_after"] = all(bool(torch.isfinite(value).all().item())
                                                        for value in after)
                if model is not None:
                    record["model_hash_after_step"] = tensor_hash(list(model.parameters()))
                    record["model_parameters_changed"] = (record["model_hash_before_step"]
                                                            != record["model_hash_after_step"])
                    if not record["model_parameters_changed"]:
                        observer.invalid_reasons.append("Grouping parameters did not update in this optimization")
                if not record["finite_losses"] or not record["finite_gradients"]:
                    observer.invalid_reasons.append(f"Non-finite loss or gradient: {scope}")
                if not record["finite_parameters_after"]:
                    observer.invalid_reasons.append(f"Non-finite optimized parameter: {scope}")
                if record["missing_gradient"]:
                    observer.invalid_reasons.append(f"Optimizer parameter had no gradient: {scope}")

        torch.optim.LBFGS.step = observed_step
        self.gc.WNet.__init__ = observed_init
        self.gc.optimize_group_fn = observed_optimize
        try:
            yield self
        finally:
            torch.optim.LBFGS.step = original_step
            self.gc.WNet.__init__ = original_init
            self.gc.optimize_group_fn = original_optimize

    def calibrate_ts(self, train_logits, train_labels, test_logits, *, scope):
        previous = self.scope
        self.scope = scope
        step_start = len(self.steps)
        try:
            result = self.ts.calibrate(train_logits=train_logits, train_labels=train_labels,
                                       test_logits=test_logits)
            records = self.steps[step_start:]
            raw = records[-1]["temperature_after"] if records else []
            # The original TS implementation calls its closure once more after
            # LBFGS.step; inspect those final gradients without running it again.
            if records:
                final_gradients = [parameter.grad for parameter in self.optimizer_parameter_refs[-1]]
                records[-1]["finite_final_ts_gradients"] = all(
                    gradient is not None and bool(self.torch.isfinite(gradient).all().item())
                    for gradient in final_gradients)
                if not records[-1]["finite_final_ts_gradients"]:
                    self.invalid_reasons.append(f"Missing/non-finite final TS gradient: {scope}")
            fallback = any(value == "nan" for value in raw) and result["tau"] == 1
            self.check_temperature(result["tau"], scope, fallback=fallback)
            if not math.isfinite(float(result["loss"])):
                self.invalid_reasons.append(f"Non-finite final TS loss: {scope}")
            if self.invalid_reasons:
                raise InvalidRun("; ".join(self.invalid_reasons))
            return result
        finally:
            self.scope = previous

    def base_calibrator(self, method_name, train_logits, train_labels, test_logits):
        if method_name != "temp_scaling":
            raise Blocked(f"Unsupported base method in fixed configuration: {method_name}")
        partition, group = divmod(self.base_calls, 2)
        self.base_calls += 1
        scope = {"kind": "group_temperature", "partition": partition, "group": group}
        count = {**scope, "calibration": int(len(train_labels)), "evaluation": int(len(test_logits))}
        self.groups.append(count)
        if count["calibration"] == 0 and count["evaluation"] > 0:
            self.invalid_reasons.append(f"Empty calibration group with evaluation samples: {scope}")
            raise InvalidRun(self.invalid_reasons[-1])
        return self.calibrate_ts(train_logits, train_labels, test_logits, scope=scope)


def execute(metadata, directory, plan, observer_holder):
    preflight = load_module("_project_preflight", ROOT / "code/preflight.py").collect()
    metadata["preflight"] = preflight
    metadata["versions"] = preflight.get("dependencies", {})
    json_write(directory / "preflight.json", preflight)
    if preflight["blockers"]:
        raise Blocked("; ".join(preflight["blockers"]))

    # These imports and all fitting happen only after the allocated-GPU checks.
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        import torchmetrics  # noqa: F401 -- fail before starting a fit if its imports break.
    except (ImportError, OSError, RuntimeError) as exc:
        raise Blocked(f"Runtime dependency import failed: {exc}") from exc
    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available():
        raise Blocked("Allocated CUDA became unavailable after preflight")
    torch.cuda.set_device(0)
    check = torch.ones(1, device="cuda:0")
    torch.cuda.synchronize()
    del check
    metadata["device"] = "cuda:0"
    metadata["gpu_name"] = torch.cuda.get_device_name(0)
    upstream = ROOT / "code/upstream"
    metadata["code_sha256"].update(validate_checkout(upstream))
    metadata["upstream_commit"] = COMMIT
    loaded_plan = OmegaConf.to_container(OmegaConf.load(ROOT / f"code/configs/{plan['level']}.yaml"),
                                         resolve=True)
    if loaded_plan != plan:
        raise Blocked("Project level configuration does not match its scheduling read")
    expected = {"num_groups": 2, "num_partitions": 2 if plan["level"] == "smoke" else 20,
                "group_optimizer_steps": 5 if plan["level"] == "smoke" else 100,
                "weight_decay": 0.1, "device": "cuda:0"}
    for key, value in expected.items():
        if plan[key] != value:
            raise Blocked(f"Level setting {key} deviates from the authorized protocol: {plan[key]}")
    method_config = OmegaConf.load(upstream / "conf/method/group_calibration_combine_ts.yaml")
    method_config.num_groups = plan["num_groups"]
    method_config.num_partitions = plan["num_partitions"]
    method_config.optimizer.steps = plan["group_optimizer_steps"]
    method_config.w_net.weight_decay = plan["weight_decay"]
    if method_config.w_net.model != "linear" or method_config.base_calibrator.name != "temp_scaling":
        raise Blocked("Unexpected upstream group model or base calibrator")
    data_config = OmegaConf.load(upstream / "conf/data/cifar10_resnet152.yaml")
    metadata["data_sha256"] = {}
    metadata["data_paths"] = {}
    for short, key in (("val", "val_path"), ("test", "test_path")):
        path = Path(str(data_config[key]))
        path = path if path.is_absolute() else upstream / path
        if not path.is_file():
            raise Blocked(f"Official {short} cache is absent: {path}")
        metadata["data_paths"][short] = str(path.resolve())
        metadata["data_sha256"][short] = sha256(path)
        data_config[key] = str(path.resolve())
    cfg = OmegaConf.load(upstream / "conf/main.yaml")
    cfg.method = method_config
    cfg.data = data_config
    cfg.seeds = [metadata["seed"]]
    metadata["configuration"] = {"project_level": plan,
                                  "effective_upstream": OmegaConf.to_container(cfg, resolve=True),
                                  "wrapper_method": metadata["method"],
                                  "wrapper_seed": metadata["seed"],
                                  "module_loading": "direct source modules; methods/__init__.py bypassed",
                                  "upstream_optimizer_lr_config_is_not_passed": True,
                                  "probability_nll_epsilon": 1e-10,
                                  "ece_bins": 15, "brier_class_reduction": "sum"}
    json_write(directory / "configuration.json", metadata["configuration"])
    utils = load_module("utils", upstream / "utils.py")
    data = load_module("_upstream_data", upstream / "data.py")
    evaluate = load_module("_upstream_evaluate", upstream / "evaluate.py")
    group = load_module("_upstream_gc", upstream / "methods/group_calibration.py")
    temperature = load_module("_upstream_ts", upstream / "methods/temp_scaling.py")
    captured = []
    original_splitter = data.RandomSplitter

    def recorded_splitter(*args, **kwargs):
        splitter = original_splitter(*args, **kwargs)
        captured.append(splitter)
        return splitter

    data.RandomSplitter = recorded_splitter
    try:
        values = data.load_data(data_config=data_config, seed=metadata["seed"])
    finally:
        data.RandomSplitter = original_splitter
    parts = dict(zip(("D_val", "D_cal", "D_eval"), values))
    metadata["data_description"] = validate_data(parts, torch)
    if len(captured) != 1 or len(captured[0].idx_splits) != 2:
        raise InvalidRun("Unexpected upstream RandomSplitter behavior")
    cal_index, eval_index = [np.asarray(index, dtype="<i8") for index in captured[0].idx_splits]
    combined = np.concatenate((cal_index, eval_index))
    if not np.array_equal(np.sort(combined), np.arange(len(combined))):
        raise InvalidRun("Split indices are not a disjoint complete partition")
    digest = hashlib.sha256()
    for name, indices in (("D_cal", cal_index), ("D_eval", eval_index)):
        digest.update(name.encode() + str(len(indices)).encode() + b"\0" + indices.tobytes())
    metadata["split_sha256"] = digest.hexdigest()
    np.savez_compressed(directory / "indices.npz", D_cal=cal_index, D_eval=eval_index,
                        D_val=np.arange(len(parts["D_val"]["labels"]), dtype=np.int64))
    metadata["initialization_hashes"] = []
    if metadata["method"] == "gc_ts_shuffle":
        children = np.random.SeedSequence(metadata["seed"]).spawn(3)
        permutations = {}
        for (name, part), child in zip(parts.items(), children):
            permutation = np.random.default_rng(child).permutation(len(part["features"]))
            permutations[name] = permutation
            parts[name] = {**part, "features": part["features"][torch.as_tensor(permutation)]}
        np.savez_compressed(directory / "permutations.npz", **permutations)
        metadata["permutation_rng"] = {"type": "NumPy SeedSequence(seed).spawn(3) + default_rng",
                                        "order": ["D_val", "D_cal", "D_eval"],
                                        "seed": metadata["seed"]}
    # The upstream seed function is called after loading/splitting, as in main.py.
    # NumPy generators above do not touch global NumPy or Torch initialization RNG.
    utils.set_seed(metadata["seed"])
    observer = Observer(torch, group, temperature)
    observer_holder.append(observer)
    validation, calibration, evaluation = (parts[name] for name in ("D_val", "D_cal", "D_eval"))
    metadata["fit_started_at_utc"] = now()
    fit_start = time.perf_counter()
    with observer.instrument():
        if metadata["method"] == "uncalibrated":
            result = {"logits": evaluation["logits"]}
        elif metadata["method"] == "ts":
            result = observer.calibrate_ts(calibration["logits"], calibration["labels"],
                                            evaluation["logits"], scope={"kind": "global_temperature"})
        else:
            result = group.calibrate_combine(
                val_features=validation["features"], val_logits=validation["logits"],
                val_labels=validation["labels"], test_train_features=calibration["features"],
                test_train_logits=calibration["logits"], test_train_labels=calibration["labels"],
                test_test_features=evaluation["features"], test_test_logits=evaluation["logits"],
                method_config=method_config, base_calibrate_fn=observer.base_calibrator,
                seed=metadata["seed"], cfg=cfg)
    torch.cuda.synchronize()
    metadata["fit_elapsed_seconds"] = time.perf_counter() - fit_start
    metadata["initialization_hashes"] = observer.initialization_hashes
    if observer.invalid_reasons:
        raise InvalidRun("; ".join(observer.invalid_reasons))
    if metadata["method"].startswith("gc_ts"):
        if len(observer.initialization_hashes) != plan["num_partitions"]:
            raise InvalidRun("Unexpected number of initialized group partitions")
        if observer.base_calls != plan["num_partitions"] * plan["num_groups"]:
            raise InvalidRun("Unexpected number of group calibration calls")
    probabilities = result.get("prob")
    if probabilities is None:
        probabilities = torch.softmax(result["logits"], dim=1)
    probabilities = probabilities.detach().cpu()
    if probabilities.shape != evaluation["logits"].shape:
        raise InvalidRun("Prediction shape differs from evaluation logits")
    if not torch.isfinite(probabilities).all().item():
        raise InvalidRun("Non-finite evaluation probabilities")
    if not ((probabilities >= 0).all().item() and (probabilities <= 1).all().item()):
        raise InvalidRun("Evaluation probabilities lie outside [0, 1]")
    if not torch.allclose(probabilities.sum(1), torch.ones(len(probabilities), dtype=probabilities.dtype),
                          atol=1e-5, rtol=1e-5):
        raise InvalidRun("Evaluation probabilities do not sum to one")
    original_predictions = evaluation["logits"].argmax(1)
    predictions = probabilities.argmax(1)
    labels = evaluation["labels"]
    metrics = evaluate.evaluate(y=labels, num_classes=probabilities.shape[1], n_bins=15,
                                pred_logits=result.get("logits"), pred_prob=result.get("prob"))
    one_hot = torch.nn.functional.one_hot(labels, num_classes=probabilities.shape[1])
    metrics = {"ece": float(metrics["ece_m"]), "nll": float(metrics["nll"]),
               "accuracy": float(metrics["acc"]),
               "brier": float((probabilities - one_hot).square().sum(1).mean().item()),
               "prediction_changes": int((predictions != original_predictions).sum().item())}
    if not all(math.isfinite(value) for value in metrics.values()):
        raise InvalidRun("Non-finite evaluation metric")
    metadata["metrics"] = metrics
    metadata["accuracy_preserving_observed"] = metrics["prediction_changes"] == 0
    metadata["warnings"] = observer.warnings
    if metrics["prediction_changes"]:
        metadata["warnings"].append("Predicted classes changed; accuracy preservation was not observed")
    np.savez_compressed(directory / "predictions.npz", probabilities=probabilities.numpy(),
                        labels=labels.numpy(), predictions=predictions.numpy(),
                        original_predictions=original_predictions.numpy(), evaluation_indices=eval_index)
    json_write(directory / "metrics.json", metrics)
    json_write(directory / "observations.json", observer.export())
    metadata["status"] = "completed"


def run_one(level, method, seed, plan, manifest=None):
    directory, attempt = reserve_run(level, method, seed)
    snapshot = directory / "source_snapshot"
    snapshot.mkdir()
    code_hashes, snapshot_paths = {}, {}
    for relative in ("code/run.py", "code/preflight.py", f"code/configs/{level}.yaml"):
        content = (ROOT / relative).read_bytes()
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        code_hashes[relative] = hashlib.sha256(content).hexdigest()
        snapshot_paths[relative] = str(destination.relative_to(directory))
    metadata = {"schema_version": 1, "run_id": directory.name, "attempt": attempt,
                "level": level, "method": method, "seed": seed, "status": "running",
                "started_at_utc": now(), "project_root": str(ROOT),
                "python_executable": sys.executable, "python_version": platform.python_version(),
                "upstream_commit_required": COMMIT, "upstream_commit": None,
                "device": None, "metrics": {}, "split_sha256": None,
                "initialization_hashes": [], "data_sha256": {},
                "code_sha256": code_hashes, "local_source_snapshots": snapshot_paths,
                "requested_configuration": {"project_level": plan, "method": method, "seed": seed},
                "protocol": "four-condition paired feature-correspondence study"}
    observer_holder = []
    json_write(directory / "metadata.json", metadata)
    if manifest is not None:
        selected = json.loads(manifest.read_text())
        selected.append(str((directory / "metadata.json").relative_to(ROOT)))
        json_write(manifest, selected)
    started = time.perf_counter()
    print(f"{directory.name}: starting", flush=True)
    with recordable_signals(), (directory / "run.log").open("x", buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                print(json.dumps(metadata["requested_configuration"], ensure_ascii=False, indent=2))
                execute(metadata, directory, plan, observer_holder)
            except Blocked as exc:
                metadata["status"] = "blocked"
                metadata["reason"] = str(exc)
                print(f"BLOCKED: {exc}")
            except InvalidRun as exc:
                metadata["status"] = "invalid"
                metadata["reason"] = str(exc)
                traceback.print_exc()
            except InterruptedRun as exc:
                metadata["status"] = "failed"
                metadata["reason"] = str(exc)
                metadata["interrupted_signal"] = exc.signum
                traceback.print_exc()
            except KeyboardInterrupt:
                metadata["status"] = "failed"
                metadata["reason"] = "Interrupted by keyboard (SIGINT)"
                metadata["interrupted_signal"] = int(signal.SIGINT)
                traceback.print_exc()
            except Exception as exc:
                metadata["status"] = "failed"
                metadata["reason"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            finally:
                if observer_holder:
                    observer = observer_holder[0]
                    metadata["initialization_hashes"] = observer.initialization_hashes
                    json_write(directory / "observations.json", observer.export())
                if metadata["status"] != "completed":
                    metadata["metrics"] = {}
                metadata["ended_at_utc"] = now()
                metadata["elapsed_seconds"] = time.perf_counter() - started
                metadata["exit_code"] = (128 + metadata["interrupted_signal"]
                                          if "interrupted_signal" in metadata else
                                          0 if metadata["status"] == "completed" else
                                          2 if metadata["status"] == "blocked" else 1)
                json_write(directory / "metadata.json", metadata)
                (directory / "exit_code.txt").write_text(str(metadata["exit_code"]) + "\n")
    print(json.dumps({"run_id": directory.name, "status": metadata["status"],
                      "reason": metadata.get("reason"), "metadata": str(directory / "metadata.json")},
                     ensure_ascii=False), flush=True)
    return metadata["exit_code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("smoke", "pilot", "main"), default="smoke")
    parser.add_argument("--method", choices=(*METHODS, "all"), default=None,
                        help="Override the level YAML; no arguments defaults to smoke/gc_ts/seed 0")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Explicit fixed seeds 0..4; main example: --seeds 0 1 2 3 4")
    parser.add_argument("--manifest-out", type=Path,
                        help="New JSON list of this command's metadata paths, relative to the repository root")
    args = parser.parse_args()
    try:
        plan = read_level_plan(ROOT / f"code/configs/{args.level}.yaml")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    methods = list(METHODS) if args.method == "all" else ([args.method] if args.method else plan["methods"])
    seeds = args.seeds if args.seeds is not None else plan["seeds"]
    if not methods or any(method not in METHODS for method in methods):
        parser.error("Level configuration contains an unsupported method")
    if not seeds or any(type(seed) is not int or seed not in range(5) for seed in seeds):
        parser.error("Only seeds 0, 1, 2, 3, 4 are permitted")
    if len(set(seeds)) != len(seeds):
        parser.error("Duplicate seeds are not permitted")
    if args.level in ("smoke", "pilot") and seeds != [0]:
        parser.error("Smoke and pilot use seed 0; use main for seeds 0..4")
    if plan["level"] != args.level:
        parser.error("Level config does not match its filename")
    manifest = None
    if args.manifest_out is not None:
        manifest = args.manifest_out if args.manifest_out.is_absolute() else ROOT / args.manifest_out
        manifest.parent.mkdir(parents=True, exist_ok=True)
        try:
            with manifest.open("x") as stream:
                stream.write("[]\n")
        except FileExistsError:
            parser.error("Refusing to overwrite manifest: " + str(manifest))
    results = []
    for seed in seeds:
        for method in methods:
            results.append(run_one(args.level, method, seed, plan, manifest=manifest))
            # Missing prerequisites apply to the entire command. Keep the first
            # blocked attempt; do not manufacture 20 identical blocked records.
            if results[-1] == 2:
                print("Remaining requested runs were not started because prerequisites are blocked.")
                return 2
            if results[-1] >= 128:
                return results[-1]
    return 1 if any(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
