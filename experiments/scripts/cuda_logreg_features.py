#!/usr/bin/env python3
"""CPU-only frozen TF-IDF transform in original environment; never load classifier."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import joblib
import numpy as np
import scipy
from scipy.sparse import save_npz
import sklearn
from eval_three_arms import load_data, build_text_arm_a
from cuda_logreg import GOLD_LOCKS, ROOT

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--split",choices=("train","test"),required=True)
 ap.add_argument("--out",type=Path,required=True)
 a=ap.parse_args()
 assert sklearn.__version__=="1.9.1", "Frozen TF-IDF requires the original environment"
 assert not a.out.exists()
 for rel,digest in GOLD_LOCKS.items():assert sha(ROOT/rel)==digest
 df,split=load_data()
 path=ROOT/"experiments/models/tfidf.joblib"
 vectorizer=joblib.load(path)
 assert type(vectorizer).__module__=="sklearn.feature_extraction.text"
 t=time.perf_counter()
 X=vectorizer.transform(build_text_arm_a(df.loc[split[a.split]])).astype(np.float64).tocsr()
 ms=(time.perf_counter()-t)*1000
 assert X.shape==(1400 if a.split=="train" else 300,20000)
 a.out.parent.mkdir(parents=True,exist_ok=True)
 save_npz(a.out,X,compressed=False)
 metadata={"split":a.split,"task_ids":split[a.split],"vectorizer_sha256":sha(path),"sklearn":sklearn.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"shape":list(X.shape),"nnz":X.nnz,"csr_bytes":X.data.nbytes+X.indices.nbytes+X.indptr.nbytes,"transform_wall_ms":ms,"npz_sha256":sha(a.out),"classifier_loaded_or_run":False,"gold_hashes":GOLD_LOCKS}
 a.out.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
 print("FROZEN_TFIDF",a.split,"shape",X.shape,"nnz",X.nnz,"bytes",metadata["csr_bytes"],flush=True)

if __name__=="__main__":main()
