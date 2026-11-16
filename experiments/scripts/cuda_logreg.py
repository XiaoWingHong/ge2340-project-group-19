#!/usr/bin/env python3
"""Native cuML CUDA training plus three fresh evals. Execution requires scoped GO."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from eval_scoring import LABELS, SEMANTICS, score_records

ROOT = Path(__file__).resolve().parents[2]
GOLD_LOCKS = {
 "experiments/data/emails_labeled_gold_masked-2026-11-14-v2.jsonl": "2868b95fa79fe7d3e1f90f0cb8d0b9d3707cdff9542aa7b7801b8a5530258779",
 "experiments/data/split_gold-2026-11-14.json": "b2e067b9995202df22bb30e5255e575ca0191e57b03a1340df858460493ef835",
}
PARAMS = dict(penalty="l2", C=1.0, fit_intercept=True, class_weight="balanced", max_iter=1000, tol=1e-4, solver="qn", linesearch_max_iter=50, output_type="cupy")


def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
 return datetime.now(ZoneInfo("Asia/Hong_Kong")).isoformat()


def memory_bound(train=1400, test=300, features=20000, classes=6, dtype_bytes=8):
 parameters = classes * (features + 1)
 return {"train_csr_worst_case_bytes": train * features * (dtype_bytes + 4) + (train + 1) * 4,
         "test_csr_worst_case_bytes": test * features * (dtype_bytes + 4) + (test + 1) * 4,
         "parameters_bytes": parameters * dtype_bytes,
         "lbfgs_s_y_history_bytes": 2 * 5 * parameters * dtype_bytes,
         "rmm_pool_limit_bytes": 4 * 1024**3,
         "requested_incremental_unified_memory_headroom_bytes": 8 * 1024**3,
         "caveat": "Matrix/history arithmetic only, not measured process peak. CUDA context, kernels, copies and solver workspace add overhead. 8 GiB is requested safety headroom, not a measured estimate; cap RMM pool at 4 GiB and fail on OOM, never fall back."}


def check_gpu_evidence(evidence):
 if evidence.get("backend") != "cuml-native" or not str(evidence.get("device", "")).startswith("cuda"):
  raise ValueError("Native cuML CUDA evidence required")
 for field in ("fit_input_cuda", "coefficients_cuda", "predictions_cuda", "probabilities_cuda", "cuda_synchronized"):
  if evidence.get(field) is not True:
   raise ValueError("Missing real CUDA execution evidence: " + field)
 for field in ("fit_cuda_event_ms", "predict_cuda_event_ms"):
  if not isinstance(evidence.get(field), (float, int)) or evidence[field] <= 0:
   raise ValueError("Missing CUDA event timing")
 if not str(evidence.get("estimator_class", "")).startswith("cuml."):
  raise ValueError("CPU estimator forbidden")


def decoded_predictions(predictions, class_codes):
 codes = [int(x) for x in predictions]
 if any(float(a) != b for a,b in zip(predictions,codes)) or any(c not in class_codes for c in codes):
  raise ValueError("Invalid CUDA class code")
 return [LABELS[c] for c in codes]


def main():
 ap = argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--execute-scoped-go", action="store_true", help="Only after manager grants scoped handoff")
 ap.add_argument("--out-dir", type=Path, default=ROOT / "experiments/models/cuda_A_20261115")
 args = ap.parse_args()
 if not args.execute_scoped_go:
  print(json.dumps({"execution": "HOLD", "params": PARAMS, "memory": memory_bound()}, indent=2))
  return
 if platform.machine() != "aarch64":
  raise RuntimeError("Planned ARM64 runtime only")
 if args.out_dir.exists():
  raise RuntimeError("Refusing to overwrite training or any evaluation")
 for rel,digest in GOLD_LOCKS.items():
  assert sha(ROOT/rel) == digest, "Frozen gold changed"
 # GPU libraries are deliberately imported only after the explicit execution gate.
 import cupy as cp
 import cuml
 import cupyx.scipy.sparse as cpx_sparse
 import rmm
 from rmm.allocators.cupy import rmm_cupy_allocator
 import joblib
 import numpy as np
 import sklearn
 from scipy.sparse import load_npz
 from importlib.metadata import version
 from eval_three_arms import load_data
 if cuml.__version__ != "25.10.00" and cuml.__version__ != "25.10.0":
  raise RuntimeError("Expected pinned cuML 25.10.0")
 assert cp.cuda.runtime.getDeviceCount() > 0
 assert cp.cuda.runtime.driverGetVersion() >= 13000
 properties = cp.cuda.runtime.getDeviceProperties(0)
 name = properties["name"].decode() if isinstance(properties["name"],bytes) else properties["name"]
 assert "GB10" in name, name
 free,total = cp.cuda.runtime.memGetInfo()
 assert free >= memory_bound()["requested_incremental_unified_memory_headroom_bytes"], "Recheck scoped memory budget"
 rmm.reinitialize(pool_allocator=True, initial_pool_size=256 * 1024**2, maximum_pool_size=4 * 1024**3)
 cp.cuda.set_allocator(rmm_cupy_allocator)
 from cuml.linear_model import LogisticRegression
 df,split = load_data()
 assert (len(split["train"]),len(split["test"])) == (1400,300)
 assert len(set(split["test"])) == 300
 # Reuse frozen train-fitted TF-IDF exactly; only the classifier is retrained.
 vectorizer_path = ROOT / "experiments/models/tfidf.joblib"
 vectorizer_hash = sha(vectorizer_path)
 feature_dir = ROOT / "experiments/models/cuda_A_features_20261115" / str(uuid.uuid4())
 feature_dir.mkdir(parents=True)
 def frozen_features(which, name):
  path = feature_dir / (name + ".npz")
  subprocess.run([str(ROOT/"experiments/.venv/bin/python"),str(ROOT/"experiments/scripts/cuda_logreg_features.py"),"--split",which,"--out",str(path)],check=True)
  meta = json.loads(path.with_suffix(".json").read_text())
  assert meta["task_ids"] == split[which] and meta["vectorizer_sha256"] == vectorizer_hash
  assert sha(path) == meta["npz_sha256"]
  return load_npz(path),meta
 X_host,train_features = frozen_features("train", "train")
 assert X_host.shape == (1400,20000)
 y_host = np.asarray([LABELS.index(lab) for lab in df.loc[split["train"],"label"]],dtype=np.int32)
 X_gpu,y_gpu = cpx_sparse.csr_matrix(X_host),cp.asarray(y_host)
 assert X_gpu.data.device.id == X_gpu.indices.device.id == X_gpu.indptr.device.id == y_gpu.device.id == 0
 estimator = LogisticRegression(**PARAMS)
 assert estimator.solver_model.lbfgs_memory == 5
 assert estimator.solver_model.penalty_normalized is True
 fit_start,fit_end = cp.cuda.Event(),cp.cuda.Event()
 cp.cuda.runtime.deviceSynchronize()
 fit_wall = time.perf_counter()
 cp.cuda.nvtx.RangePush("GE2340_CUDA_LOGREG_FIT")
 fit_start.record()
 estimator.fit(X_gpu,y_gpu)
 fit_end.record();fit_end.synchronize()
 cp.cuda.nvtx.RangePop()
 fit_wall_ms = (time.perf_counter()-fit_wall)*1000
 fit_event_ms = float(cp.cuda.get_elapsed_time(fit_start,fit_end))
 assert isinstance(estimator.coef_,cp.ndarray) and estimator.coef_.device.id == 0
 assert estimator.coef_.dtype == cp.float64, "Float64 sparse path must not silently downcast"
 assert type(estimator).__module__.startswith("cuml.")
 classes = [int(x) for x in cp.asnumpy(cp.asarray(estimator.classes_)).ravel()]
 assert sorted(classes) == list(range(6))
 assert bool(cp.isfinite(estimator.coef_).all())
 args.out_dir.mkdir(parents=True)
 training_id = str(uuid.uuid4())
 artifact = args.out_dir / "cuml_logreg.joblib"
 joblib.dump(estimator,artifact)
 metadata = {"trained_at":stamp(), "training_id":training_id, "backend":"cuml-native", "device":"cuda:0", "gpu":name, "driver_cuda_api":cp.cuda.runtime.driverGetVersion(), "cuda_runtime":cp.cuda.runtime.runtimeGetVersion(), "cuml":cuml.__version__, "cupy":cp.__version__, "distribution_versions":{package:version(package) for package in ("cuml-cu13","libcuml-cu13","cudf-cu13","rmm-cu13","cuda-python","cuda-toolkit","numba-cuda","numba","numpy","scipy","scikit-learn")}, "numpy":np.__version__, "sklearn_preprocessing":train_features["sklearn"], "cuml_compat_sklearn":sklearn.__version__, "feature_bridge":"Frozen TF-IDF executes in original sklearn1.9.1/NumPy2.5.3 venv, CSR serialized without refitting; native cuML uses isolated compatible sklearn1.7.2. No CPU classifier is loaded.", "training_features":train_features, "feature_directory":str(feature_dir), "params":PARAMS, "qn_internal_defaults":{"lbfgs_memory":5,"penalty_normalized":True}, "seed_note":"Historical random_state=2340 has no parameter in native cuML QN; deterministic full-batch solver, no SGD sampling; split seed stays2340. No numerical identity claimed.", "solver_differences":"sklearn lbfgs -> native cuML qn (L2 selects L-BFGS); cuML default history5 and line-search50, stopping/reduction/regularization normalization differ; penalty_normalized=True and lbfgs_memory=5 are verified internal QN defaults in25.10, not constructor keywords in this release. Balanced class weights, C1, L2, intercept, tol1e-4 and max_iter1000 preserved.", "n_train":1400, "n_features":20000, "dtype":"float64 GPU CSR (sparse path avoids the dense convert_dtype=True float32 default)", "fit_wall_ms":fit_wall_ms, "fit_cuda_event_ms":fit_event_ms, "n_iter":cp.asnumpy(cp.asarray(estimator.n_iter_)).tolist(), "vectorizer_sha256":vectorizer_hash, "model_sha256":sha(artifact), "gold_hashes":GOLD_LOCKS, "memory_plan":memory_bound(), "script_sha256":sha(__file__), "nsys_note":"Launch under planned nsys profile and inspect FIT/PREDICT CUDA kernel ranges before accepting outputs."}
 (args.out_dir/"training.json").write_text(json.dumps(metadata,indent=2)+"\n")
 del X_gpu,y_gpu,X_host
 test = df.loc[split["test"]]
 for run in ("run1","run2","run3"):
  started_at = stamp()
  cp.cuda.runtime.deviceSynchronize()
  start = time.perf_counter()
  host,test_features = frozen_features("test",run)
  gpu = cpx_sparse.csr_matrix(host)
  begin,end = cp.cuda.Event(),cp.cuda.Event()
  cp.cuda.nvtx.RangePush("GE2340_CUDA_LOGREG_PREDICT_"+run)
  begin.record()
  predictions = estimator.predict(gpu)
  probabilities = estimator.predict_proba(gpu)
  end.record();end.synchronize()
  cp.cuda.nvtx.RangePop()
  assert isinstance(predictions,cp.ndarray) and isinstance(probabilities,cp.ndarray)
  assert predictions.device.id == probabilities.device.id == gpu.data.device.id == gpu.indices.device.id == gpu.indptr.device.id == 0
  assert probabilities.shape == (300,6) and bool(cp.isfinite(probabilities).all())
  assert bool(cp.allclose(probabilities.sum(axis=1),1,rtol=1e-6,atol=1e-6))
  decoded = decoded_predictions(cp.asnumpy(predictions).ravel().tolist(),classes)
  confidence = cp.asnumpy(probabilities.max(axis=1)).tolist()
  cp.cuda.runtime.deviceSynchronize()
  wall_ms = (time.perf_counter()-start)*1000
  evidence = {"backend":"cuml-native", "device":"cuda:0", "gpu":name, "estimator_class":type(estimator).__module__+"."+type(estimator).__name__, "fit_input_cuda":True, "coefficients_cuda":True, "predictions_cuda":True, "probabilities_cuda":True, "cuda_synchronized":True, "fit_cuda_event_ms":fit_event_ms, "predict_cuda_event_ms":float(cp.cuda.get_elapsed_time(begin,end)), "profiling":"NVTX FIT/PREDICT ranges, CUDA kernel trace recorded by scoped nsys command; require trace inspection"}
  check_gpu_evidence(evidence)
  items = [{"task_id":int(task), "gold":gold, "prediction":pred, "confidence":conf, "latency_ms":wall_ms/300, "raw_reply":None, "error":None} for task,gold,pred,conf in zip(split["test"],test["label"],decoded,confidence)]
  arm = {"arm":"A_cuda_logreg", "backend_id":"cuml_native", "status":"ok", "items":items, "task_ids":[r["task_id"] for r in items], "gold_labels":[r["gold"] for r in items], "predictions":decoded, "confidences":confidence, "raw_replies":[None]*300, "row_errors":[None]*300, "latency_ms_per_email":[wall_ms/300]*300, "latency_ms":wall_ms/300, "latency_ms_batch_total":wall_ms, "device_evidence":evidence, **score_records(items)}
  provenance = {"run_id":run, "inference_performed":True, "invocation_id":str(uuid.uuid4()), "started_at":started_at, "completed_at":stamp(), "training_id":training_id, "training_metadata_sha256":sha(args.out_dir/"training.json"), "model_sha256":sha(artifact), "gold_hashes":GOLD_LOCKS, "vectorizer_sha256":vectorizer_hash, "feature_metadata":test_features, "feature_bridge_script_sha256":sha(ROOT/"experiments/scripts/cuda_logreg_features.py"), "script_sha256":sha(__file__), "device_evidence":evidence, "latency_note":"Batch wall time includes original-venv CPU TF-IDF subprocess startup and NPZ serialization/loading (compatibility bridge), CSR conversion, H2D transfer, both native CUDA predict/predict_proba and D2H, with explicit synchronization; model training/loading excluded. Equal batch-time share per email, not individual measured requests. nsys profiling overhead applies."}
  payload = {"scoring_semantics":SEMANTICS, "taxonomy":LABELS, "provenance":provenance, "summary":[{k:v for k,v in arm.items() if k in ("arm","n_scored","n_correct","acc_hard","acc_conf","macro_f1","errors","error_rate","latency_ms")}], "arms":[arm]}
  path = args.out_dir/f"eval_gold_masked_v2_A_cuda_{run}.json"
  with path.open("x") as f:json.dump(payload,f,indent=2);f.write("\n")
  path.with_suffix(".md").write_text(f"# Fresh CUDA A {run}\n\nNative cuML; n=300. Acc_hard={arm['acc_hard']:.9f}; Acc_conf={arm['acc_conf']:.9f}; macro-F1={arm['macro_f1']:.9f}; failures={arm['errors']}; batch latency share={arm['latency_ms']:.6f} ms.\n\nSolver and timing caveats: see training.json and JSON provenance. Require CUDA trace verification.\n")
  del gpu,host,predictions,probabilities
 assert sha(vectorizer_path) == vectorizer_hash
 for rel,digest in GOLD_LOCKS.items():assert sha(ROOT/rel) == digest
 print("Training and all three genuine CUDA evaluations saved; inspect CUDA trace before accepting.")

if __name__ == "__main__":main()
