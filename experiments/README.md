# Experiments

Email-type classification on a locked, name-masked human gold set
(`promo | admin | course | personal | news | other`).

## Official results

Official scores are the 11 per-model `ge2340-eval-times-v1` files under `results/`:

- `eval_times_A_cuda_logreg.json`
- `eval_times_B_clef.json`
- `eval_times_B_kev.json`
- `eval_times_B_laya.json`
- `eval_times_B_openjev.json`
- `eval_times_B_semif.json`
- `eval_times_C_gemma4_26b_a4b.json`
- `eval_times_C_lfm25_24b_a2b.json`
- `eval_times_C_muse_glimmer_30b.json`
- `eval_times_C_nemotron35_lightning.json`
- `eval_times_C_qwen38_27b.json`

Each file is one model evaluated with `--times 3`: three genuine full passes over the locked 300-row test set, 900 raw rows, and the arithmetic means of the three per-run metrics (`n_correct` and `errors` are mean counts; n remains 300 per run).

- **Acc_hard** (`acc_hard`) = number of correct predictions / 300.
- **Acc_conf** (`acc_conf`) = sum of confidence on correct, non-error, valid-prediction rows / 300. Failed, null, and incorrect rows contribute 0.
- **Macro-F1** (`macro_f1`) = six-label macro F1 (abstentions are false negatives, not a seventh class).
- `mean_top_choice_confidence` is a separate field, not Acc_conf.

`models/` holds only the official Arm A estimator `models/cuda_A_20261115/cuml_logreg.joblib` and the frozen vectorizer `models/tfidf.joblib`.

Print the official mean table (and per-run rows) without rewriting any scores:

```bash
python scripts/summarize_eval_times.py
```

## Data

- `data/emails_labeled_gold_masked-2026-11-14-v2.jsonl` — gold labels (name-masked v2). Gitignored. Keep a local copy to run the evaluator.
- `data/split_gold-2026-11-14.json` — stratified 70/15/15 split, seed 2340
- `data/label_map.json` — label order and ids

Do not overwrite these files with a newly sampled set.

## Environment

Arm A requires a CUDA GPU and `../venvs/cuml25_10_cuda13`. Arm B and Arm C official scoring also require CUDA-backed serves; start the matching script from `../bin/` and confirm the runtime device. Do not fall back to CPU.

- `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Copy `.env.eval.example` to `.env.eval` and fill in backend URLs and keys. Do not commit or publish the real `.env.eval`.

`--help` works with the host Python. Evaluation and tests use this `.venv`. Official Arm A (`--arms A`) must be launched with the cuML environment, which still uses `.venv` only for the frozen sklearn TF-IDF worker.

## Train and eval

The official evaluation entry is `scripts/eval_three_arms.py --times N` (`N` is a positive integer, default 1). Each invocation evaluates **exactly one** model: `--arms A`, or `--arms B` with one `--decision-backends` id, or `--arms C` with one `--llm-backends` id. If the model is omitted or more than one target is named, the script exits with an error and writes no official JSON. Environment backend lists cannot select the model.

`--times N` runs N genuine full passes over the locked 300-row test set and writes **one JSON** with 300×N raw rows plus arithmetic means of the N per-run metrics. It does **not** write result Markdown. `--out-md` is still accepted by argparse but is rejected at runtime and is not an official output path. Omitting `--out-json` writes `results/eval_times_<target>.json` (dry-run: `results/eval_times_<target>_dryrun.json`).

```bash
# Usage (host Python is enough for --help)
python scripts/eval_three_arms.py --help

# Read-only official table
python scripts/summarize_eval_times.py

# Experiment virtualenv: B/C evaluation and tests
source .venv/bin/activate
python scripts/eval_three_arms.py --arms C --llm-backends qwen38_27b --dry-run
python scripts/eval_three_arms.py --arms C --llm-backends qwen38_27b --times 3 \
  --require-frozen-300
python scripts/eval_three_arms.py --arms B --decision-backends clef --times 3 \
  --require-frozen-300

# Official Arm A (cuML environment)
../venvs/cuml25_10_cuda13/bin/python scripts/eval_three_arms.py --arms A --times 3 \
  --require-frozen-300
```

`eval_three_arms.py` reads the locked gold labels and split. Official scores are already in the 11 `results/eval_times_*.json` files; only run the evaluator if you intend to produce a new JSON.

Offline tests (no inference):

```bash
cd scripts
../.venv/bin/python -m unittest test_eval_times test_eval_completion test_eval_semantics test_chat_http_raw_capture
```
