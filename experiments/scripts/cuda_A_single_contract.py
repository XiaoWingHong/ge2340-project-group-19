"""Final A requires three genuine sequential-single-email passes, never batch/300."""
import math
import statistics

MODEL_SHA256 = "82c93933d51a2ea25215ab63f7a13c7a9ea94a26cbe4b1e05473dbb933c1c48c"
TRAINING_ID = "940bd88a-ecdf-43ba-baa2-200832100ebd"
KERNEL_PROOF_SHA256 = "d72a78a4fb306daef46b66ab3b03bb698c2432879c533649396990cfa3fd79c8"
VECTORIZER_SHA256 = "5d20d1f70625b54890abb86dcc7101b6ebcd2127546fd023f02f236f475c85c5"
# Historical diagnostic identifiers retained so official single-email validation
# and offline eligibility tests stay consistent; they are not a stored proof tree.
TIMING_AUDIT_SHA256 = "e15d2dc7eedf8d501e03795e0cc890f4e96ab97c4044a4f56c525a63fc76f833"
PROFILED_HARNESS_SHA256 = "f55b4a2496cf99d6b24158a24f63f26131245ff3aecf7aafc2e2748378df886e"
NATIVE_CODE_SHA256 = "187379eb00000000000000000000000000000000000000000000000000000000"
ROLE = "final_single_email"
PROTOCOL = "loaded-model-single-email-e2e-v1"


def validate_clean_provenance(p):
    if p.get("evaluation_role") != ROLE or p.get("timing_protocol") != PROTOCOL:
        raise ValueError("A final latency requires sequential single-email inference; CPU/profiled/batch diagnostics excluded")
    for key, value in {
        "evaluation_unit": "single_email",
        "inference_batch_size": 1,
        "timed_samples": 300,
        "sequential": True,
        "per_email_wall_timing": True,
    }.items():
        if p.get(key) != value:
            raise ValueError("Missing actual single-email invariant: " + key)
    if p.get("profiled") is True or (p.get("profiler_check") or {}).get("detected") or (p.get("post_profiler_check") or {}).get("detected"):
        raise ValueError("Profiled CUDA A diagnostics are never final")
    if p.get("warmup_inferences") != 0 or p.get("retained_first_use") is not True or p.get("no_inference_discarded") is not True:
        raise ValueError("A discarded first-use or warmup is never final")
    if p.get("bridge_included") is not True or p.get("transfers_included") is not True or p.get("synchronized") is not True or p.get("startup_excluded") is not True:
        raise ValueError("Missing real bridge cost")
    if p.get("training_id") != TRAINING_ID or p.get("model_sha256") != MODEL_SHA256:
        raise ValueError("Fresh CUDA A requires the frozen training and model provenance")
    if p.get("kernel_proof_sha256") not in (None, KERNEL_PROOF_SHA256):
        raise ValueError("Kernel proof hash mismatch")
    if p.get("vectorizer_sha256") not in (None, VECTORIZER_SHA256):
        raise ValueError("Vectorizer hash mismatch")
    for key, expected in {
        "native_inference_code_sha256": NATIVE_CODE_SHA256,
        "profiled_harness_sha256": PROFILED_HARNESS_SHA256,
        "timing_audit_sha256": TIMING_AUDIT_SHA256,
    }.items():
        if p.get(key) not in (None, expected):
            raise ValueError("Historical A diagnostic hash mismatch: " + key)


def validate_single_arm(arm):
    rows = arm["items"]
    if len(rows) != 300:
        raise ValueError("Require full300 sequential rows")
    for index, row in enumerate(rows):
        if row.get("inference_batch_size") != 1 or row.get("inference_sequence") != index or row.get("prediction_source") != "single-row-native-cuda":
            raise ValueError("Missing genuine one-row prediction/timing linkage")
        t = row.get("latency_ms")
        if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError("Missing finite per-email wall time")
    if not math.isclose(arm["latency_ms"], statistics.mean(r["latency_ms"] for r in rows), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("A latency must be mean of actual individually timed predictions")
