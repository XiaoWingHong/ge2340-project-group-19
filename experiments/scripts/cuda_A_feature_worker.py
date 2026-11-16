#!/usr/bin/env python3
"""Resident ORIGINAL-sklearn text bridge: one frozen email per request; no classifier."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import joblib, numpy as np, sklearn
from scipy.sparse import save_npz
from eval_three_arms import load_data, build_text_arm_a
from cuda_logreg import ROOT, GOLD_LOCKS
from cuda_A_single_contract import VECTORIZER_SHA256

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ordered_request(request, count, ids, times=1):
    total = 300 * times
    if count >= total:
        raise ValueError("Exactly %s requests only; no warmups or extras" % total)
    run = "run%d" % (count // 300 + 1)
    index = count % 300
    if request != {"run_id": run, "index": index}:
        raise ValueError("Sequential frozen300 passes required in order")
    return run, index, ids[index]

def transform_one(vectorizer, df, task_id):
    return vectorizer.transform(build_text_arm_a(df.loc[[task_id]])).astype(np.float64).tocsr()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--times", type=int, default=1)
    args = ap.parse_args()
    if args.times < 1:
        raise SystemExit("--times must be a positive integer")
    out = args.out_dir.resolve()
    if not str(out).startswith(str((ROOT / "experiments/tmp").resolve())):
        raise SystemExit("feature bridge must live under experiments/tmp and is removed after the run")
    out.mkdir(parents=True, exist_ok=False)
    for rel, digest in GOLD_LOCKS.items():
        assert sha(ROOT / rel) == digest
    assert sklearn.__version__ == "1.9.1"
    df, split = load_data()
    ids = split["test"]
    assert len(ids) == 300
    vector = ROOT / "experiments/models/tfidf.joblib"
    assert sha(vector) == VECTORIZER_SHA256
    vectorizer = joblib.load(vector)
    assert type(vectorizer).__module__ == "sklearn.feature_extraction.text"
    print(json.dumps({"status": "ready", "task_ids": ids, "vectorizer_sha256": sha(vector), "sklearn": sklearn.__version__, "numpy": np.__version__, "classifier_loaded_or_run": False, "times": args.times}), flush=True)
    count = 0
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("command") == "shutdown":
            break
        run, index, task_id = ordered_request(req, count, ids, args.times)
        started = time.perf_counter()
        X = transform_one(vectorizer, df, task_id)
        assert X.shape == (1, 20000)
        path = out / ("%s_%03d.npz" % (run, index))
        assert not path.exists()
        save_npz(path, X, compressed=False)
        print(json.dumps({"status": "ok", "run_id": run, "index": index, "task_id": task_id, "path": str(path), "npz_sha256": sha(path), "shape": list(X.shape), "nnz": int(X.nnz), "worker_call_ms": (time.perf_counter() - started) * 1000}), flush=True)
        count += 1

if __name__ == "__main__":
    main()
