#!/usr/bin/env python3
"""Official CUDA A sequential single-email passes. CUDA libraries are imported only at execution time."""
from __future__ import annotations

import hashlib
import json
import os
import select
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from cuda_A_single_contract import KERNEL_PROOF_SHA256, MODEL_SHA256, PROTOCOL, ROLE, TRAINING_ID, VECTORIZER_SHA256, validate_clean_provenance, validate_single_arm
from cuda_logreg import GOLD_LOCKS, ROOT, check_gpu_evidence, decoded_predictions, stamp
from eval_scoring import SEMANTICS, score_records

TIMING_BOUNDARY = "Loaded-model sequential single-email wall: immediately before request to resident ORIGINAL-sklearn CPU worker through original From/Subject/Body preprocessing, frozen TF-IDF transform, per-email NPZ serialization/hash/IPC, NPZ hash/load/CSR, H2D, native one-row predict+predict_proba, confidence/label D2H and decoding, final device synchronization. Startup/import/model/vectorizer/data loading and pre-call synchronization excluded; scoring/log writes after timer. NOT batch-wall/300; local IPC rather than B/C HTTP."
COLD_WARM_POLICY = "One loaded fixed-model process, sequential actual single-email calls in frozen order as N full 300-row passes. Zero inference warmups, no discarded calls. First-email lazy initialization retained. Later calls naturally warm; existing disk compilation cache allowed, not cache-cold."
WORKER = ROOT / "experiments/scripts/cuda_A_feature_worker.py"
EXP_PYTHON = ROOT / "experiments/.venv/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker_message(worker, timeout=60):
    readable, _, _ = select.select([worker.stdout], [], [], timeout)
    if not readable:
        raise RuntimeError("Own CPU feature worker timed out; stop and preserve partial run set")
    line = worker.stdout.readline()
    if not line:
        raise RuntimeError("Own CPU feature worker exited; inspect its stderr, never fall back")
    return json.loads(line)


def stop_own_worker(worker):
    if worker.poll() is not None:
        return
    try:
        worker.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
        worker.stdin.flush()
    except (BrokenPipeError, OSError, ValueError):
        pass
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=5)


def profiler_check():
    evidence = {"detected": False, "injection_env": {}, "parent_chain": [], "injected_libraries": []}
    for key in ("CUDA_INJECTION64_PATH", "CUDA_INJECTION32_PATH", "NVTX_INJECTION64_PATH", "NVPROF_ID", "NSYS_PROFILING_SESSION_ID"):
        if os.environ.get(key):
            evidence["injection_env"][key] = os.environ[key]
    if any(token in os.environ.get("LD_PRELOAD", "").lower() for token in ("cupti", "toolsinjection", "nvprof", "nsight")):
        evidence["injection_env"]["LD_PRELOAD"] = "profiling-related preload present"
    pid = os.getpid()
    for _ in range(12):
        base = Path("/proc") / str(pid)
        try:
            comm = (base / "comm").read_text().strip()
            evidence["parent_chain"].append({"pid": pid, "comm": comm})
            if comm.lower() in {"nsys", "nsys-launcher", "nvprof", "ncu", "ncu-ui"}:
                evidence["detected"] = True
            status = (base / "status").read_text().splitlines()
            pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
            if pid <= 1:
                break
        except OSError:
            break
    maps = Path("/proc/self/maps").read_text()
    evidence["injected_libraries"] = [line.split()[-1] for line in maps.splitlines() if any(token in line.lower() for token in ("libtoolsinjection", "libcuinj", "libnvperf_host"))]
    evidence["detected"] = evidence["detected"] or bool(evidence["injection_env"] or evidence["injected_libraries"])
    if evidence["detected"]:
        raise RuntimeError("Profiler detected; no official A run can be accepted")
    return evidence


def run_cuda_a_session(times: int) -> list[dict]:
    if not isinstance(times, int) or isinstance(times, bool) or times < 1:
        raise ValueError("times must be a positive integer")
    tmp_root = ROOT / "experiments/tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cuda_A_times_", dir=tmp_root))
    try:
        return _execute_session(times, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _execute_session(times: int, work: Path) -> list[dict]:
    import cupy as cp
    import cupyx.scipy.sparse as sparse
    import cuml
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator
    import joblib
    from scipy.sparse import load_npz
    from eval_three_arms import load_data

    model_path = ROOT / "experiments/models/cuda_A_20261115/cuml_logreg.joblib"
    proof_path = ROOT / "experiments/models/cuda_A_20261115/cuda_A_kernel_proof.json"
    assert sha(model_path) == MODEL_SHA256
    assert sha(proof_path) == KERNEL_PROOF_SHA256
    proof = json.loads(proof_path.read_text())
    train = json.loads((ROOT / "experiments/models/cuda_A_20261115/training.json").read_text())
    assert train["training_id"] == proof["training_id"] == TRAINING_ID and proof["model_sha256"] == MODEL_SHA256
    for rel, digest in GOLD_LOCKS.items():
        assert sha(ROOT / rel) == digest
    assert cuml.__version__ in {"25.10.0", "25.10.00"}
    assert cp.cuda.runtime.getDeviceCount() > 0 and cp.cuda.runtime.memGetInfo()[0] >= 8 * 1024 ** 3
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    assert "GB10" in name
    rmm.reinitialize(pool_allocator=True, initial_pool_size=256 * 1024 ** 2, maximum_pool_size=4 * 1024 ** 3)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    model = joblib.load(model_path)
    assert type(model).__module__.startswith("cuml.") and isinstance(model.coef_, cp.ndarray)
    assert model.coef_.device.id == 0 and model.coef_.dtype == cp.float64
    df, split = load_data()
    assert len(split["test"]) == len(set(split["test"])) == 300
    classes = [int(x) for x in cp.asnumpy(cp.asarray(model.classes_)).ravel()]
    assert sorted(classes) == list(range(6))
    feature_dir = work / "feature_bridge"
    worker = subprocess.Popen(
        [str(EXP_PYTHON), str(WORKER), "--out-dir", str(feature_dir), "--times", str(times)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=(work / "feature_worker.stderr").open("w"),
        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    run_set_id = str(uuid.uuid4())
    runs = []
    try:
        ready = worker_message(worker)
        assert ready["status"] == "ready" and ready["task_ids"] == split["test"]
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        for offset in range(times):
            run = "run%d" % (offset + 1)
            rows = []
            started_at = stamp()
            before = profiler_check()
            gpu_event_times = []
            for index, task_id in enumerate(split["test"]):
                cp.cuda.runtime.deviceSynchronize()
                start = time.perf_counter()
                worker.stdin.write(json.dumps({"run_id": run, "index": index}) + "\n")
                worker.stdin.flush()
                feature = worker_message(worker)
                assert feature["status"] == "ok" and feature["task_id"] == task_id and feature["index"] == index and feature["run_id"] == run
                path = Path(feature["path"])
                assert path.parent == feature_dir and sha(path) == feature["npz_sha256"]
                host = load_npz(path)
                assert host.shape == (1, 20000)
                gpu = sparse.csr_matrix(host)
                assert gpu.data.device.id == gpu.indices.device.id == gpu.indptr.device.id == 0
                begin.record()
                predictions = model.predict(gpu)
                probabilities = model.predict_proba(gpu)
                end.record()
                end.synchronize()
                assert isinstance(predictions, cp.ndarray) and isinstance(probabilities, cp.ndarray)
                assert predictions.device.id == probabilities.device.id == 0 and probabilities.shape == (1, 6)
                assert bool(cp.isfinite(probabilities).all()) and bool(cp.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-6))
                decoded = decoded_predictions(cp.asnumpy(predictions).ravel().tolist(), classes)
                confidence = cp.asnumpy(probabilities.max(axis=1)).tolist()
                cp.cuda.runtime.deviceSynchronize()
                wall_ms = (time.perf_counter() - start) * 1000
                assert len(decoded) == len(confidence) == 1
                gpu_event_times.append(float(cp.cuda.get_elapsed_time(begin, end)))
                rows.append({
                    "task_id": int(task_id),
                    "gold": df.loc[task_id, "label"],
                    "prediction": decoded[0],
                    "confidence": float(confidence[0]),
                    "latency_ms": wall_ms,
                    "raw_reply": None,
                    "error": None,
                    "inference_batch_size": 1,
                    "inference_sequence": index,
                    "prediction_source": "single-row-native-cuda",
                    "feature_metadata": feature,
                })
                os.remove(path)
                del gpu, host, predictions, probabilities
            after = profiler_check()
            evidence = {
                "backend": "cuml-native",
                "device": "cuda:0",
                "gpu": name,
                "estimator_class": type(model).__module__ + "." + type(model).__name__,
                "fit_input_cuda": True,
                "coefficients_cuda": True,
                "predictions_cuda": True,
                "probabilities_cuda": True,
                "cuda_synchronized": True,
                "fit_cuda_event_ms": train["fit_cuda_event_ms"],
                "predict_cuda_event_ms": sum(gpu_event_times),
                "fit_evidence_is_prior_training": True,
                "fit_evidence_origin": "Unchanged model and prior FIT kernel proof; NO NEW FIT",
                "profiling": "none",
            }
            check_gpu_evidence(evidence)
            provenance = {
                "run_id": run,
                "run_set_id": run_set_id,
                "invocation_id": str(uuid.uuid4()),
                "started_at": started_at,
                "completed_at": stamp(),
                "inference_performed": True,
                "actual_inference_rows": 300,
                "evaluation_role": ROLE,
                "profiled": False,
                "timing_protocol": PROTOCOL,
                "model_sha256": MODEL_SHA256,
                "training_id": TRAINING_ID,
                "kernel_proof_sha256": KERNEL_PROOF_SHA256,
                "gold_hashes": GOLD_LOCKS,
                "device_evidence": evidence,
                "profiler_check": before,
                "post_profiler_check": after,
                "warmup_inferences": 0,
                "retained_first_use": True,
                "method_calls": ["predict", "predict_proba"],
                "startup_excluded": True,
                "bridge_included": True,
                "transfers_included": True,
                "synchronized": True,
                "no_inference_discarded": True,
                "timing_boundary": TIMING_BOUNDARY,
                "cold_warm_policy": COLD_WARM_POLICY,
                "vectorizer_sha256": ready["vectorizer_sha256"],
                "evaluation_unit": "single_email",
                "inference_batch_size": 1,
                "timed_samples": 300,
                "sequential": True,
                "per_email_wall_timing": True,
                "worker_code_sha256": sha(WORKER),
                "clean_harness_sha256": sha(Path(__file__)),
                "original_worker_environment": {k: ready[k] for k in ready if k != "task_ids"},
            }
            validate_clean_provenance(provenance)
            arm = {
                "arm": "A_cuda_logreg",
                "backend_id": "cuml_native",
                "status": "ok",
                "items": rows,
                "task_ids": split["test"],
                "gold_labels": [row["gold"] for row in rows],
                "predictions": [row["prediction"] for row in rows],
                "confidences": [row["confidence"] for row in rows],
                "latency_ms": statistics.mean(row["latency_ms"] for row in rows),
                "device_evidence": evidence,
                **score_records(rows),
            }
            validate_single_arm(arm)
            runs.append({"arm": arm, "provenance": provenance})
        return runs
    finally:
        stop_own_worker(worker)
