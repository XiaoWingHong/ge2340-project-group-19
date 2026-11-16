# Email-type classification

This project classifies emails into six labels:

`promo | admin | course | personal | news | other`

Human gold labels (name-masked) are compared against three systems:

- **Arm A** — frozen sklearn TF-IDF plus official native CUDA cuML logistic regression, evaluated as sequential single-email passes
- **Arm B** — SystemOne decision backends (clef, laya, kev, semif, openjev)
- **Arm C** — local LLMs (Qwen 3.8 27B, Gemma 4 26B-A4B, Nemotron 3.5 Lightning, Muse Glimmer 30B, LFM 2.5 24B-A2B)

## Official results

Official scores are the 11 per-model `ge2340-eval-times-v1` files under `experiments/results/`:

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

Each file is the official result of one model evaluated with `--times 3`: three genuine full passes over the locked 300-row test set, 900 raw rows, and the arithmetic means of the three per-run metrics (`n_correct` and `errors` are mean counts; n remains 300 per run).

- **Acc_hard** (`acc_hard`) = number of correct predictions / 300.
- **Acc_conf** (`acc_conf`) = sum of confidence on correct, non-error, valid-prediction rows / 300. Failed, null, and incorrect rows contribute 0.
- **Macro-F1** (`macro_f1`) = six-label macro F1 (abstentions are false negatives, not a seventh class).
- `mean_top_choice_confidence` is a separate field, not Acc_conf.

Failed or unparseable replies keep a null prediction, retain the raw reply, error, and latency, and are always incorrect (including when gold is `other`).

Print the official mean table (and per-run rows) without rewriting any scores:

```bash
python experiments/scripts/summarize_eval_times.py
```

The official Arm A estimator is `experiments/models/cuda_A_20261115/cuml_logreg.joblib`, used with the frozen vectorizer `experiments/models/tfidf.joblib`. `experiments/models/` holds only those estimators; official scores live under `experiments/results/`.

## Environment

Arm A requires a CUDA GPU and the cuML environment at `venvs/cuml25_10_cuda13`. Arm B and Arm C official scoring also require CUDA-backed serves; start the matching script under `bin/` and confirm the runtime device. Do not fall back to CPU.

- Python 3, `experiments/.venv`, and the packages in `experiments/requirements.txt` (used for `--help`, tests, Arm B/C evaluation, and Arm A’s frozen sklearn TF-IDF worker).
- Copy `experiments/.env.eval.example` to `experiments/.env.eval` and fill in backend URLs, keys, and model ids. Do not commit or publish the real `.env.eval`.
- Start scripts under `bin/` and wrappers under `serve/` assume model weights are already on disk (those weight trees are gitignored).

## How to run the evaluation

The official evaluation entry is `experiments/scripts/eval_three_arms.py --times N` (`N` is a positive integer, default 1). Each invocation evaluates **exactly one** model: `--arms A`, or `--arms B` with one `--decision-backends` id, or `--arms C` with one `--llm-backends` id. If the model is omitted or more than one target is named, the script exits with an error and writes no official JSON. Environment backend lists cannot select the model.

`--times N` runs N genuine full passes over the locked 300-row test set and writes **one JSON** with 300×N raw rows plus arithmetic means of the N per-run metrics. It does **not** write result Markdown. `--out-md` is still accepted by argparse but is rejected at runtime and is not an official output path. Omitting `--out-json` writes `results/eval_times_<target>.json` (dry-run: `results/eval_times_<target>_dryrun.json`).

`--help` works with the host Python (`python experiments/scripts/eval_three_arms.py --help`). Running an evaluation or the unit tests requires `experiments/.venv`. Official Arm A (`--arms A`) must be launched with `venvs/cuml25_10_cuda13`, which provides cuML; that path still uses `experiments/.venv` only for the frozen TF-IDF worker.

From `experiments/`, with `.env.eval` in place:

```bash
# Usage (host Python is enough for --help)
python scripts/eval_three_arms.py --help

# Read-only official table
python scripts/summarize_eval_times.py

# Activate the experiment environment for B/C evaluation and tests
source .venv/bin/activate

# Check configuration without sending requests
python scripts/eval_three_arms.py --arms C --llm-backends qwen38_27b --dry-run

# One Arm C backend, three genuine repeats; default out-json is
# results/eval_times_C_qwen38_27b.json
python scripts/eval_three_arms.py --arms C --llm-backends qwen38_27b --times 3 \
  --require-frozen-300

# One Arm B backend
python scripts/eval_three_arms.py --arms B --decision-backends clef --times 3 \
  --require-frozen-300

# Official Arm A (cuML environment); default out-json is
# results/eval_times_A_cuda_logreg.json
../venvs/cuml25_10_cuda13/bin/python scripts/eval_three_arms.py --arms A --times 3 \
  --require-frozen-300
```

`eval_three_arms.py` reads the locked gold labels and split (seed 2340, stratified 70/15/15). Official scores are already in the 11 `results/eval_times_*.json` files; only run the evaluator if you intend to produce a new JSON.

Offline tests (from `experiments/scripts/`, experiment virtualenv, no inference):

```bash
../.venv/bin/python -m unittest test_eval_times test_eval_completion test_eval_semantics test_chat_http_raw_capture
```

### Preparing new mail (optional)

The labeled gold set is already frozen. To turn new Outlook `.pst` mail into Label Studio tasks (this does **not** overwrite gold):

```bash
cd converter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m converter --input path/to/mailbox.pst --output /tmp/labelstudio_tasks.json \
  --sample 2000 --seed 2340
```

See `converter/README.md` and `python -m converter --help`. Raw input must be a single `.pst` file or a folder of `*.pst` files (nested allowed). `.eml`, `.olm`, and MBOX are not accepted. After parse → anonymize → English-filter → empty-drop → dedupe, `--sample 2000` randomly selects 2000 tasks with seed 2340. If fewer than 2000 remain, all are kept and a warning is printed.

## Layout

```
converter/                     PST mail → Label Studio JSON (.pst only)
experiments/
  results/                     Official eval_times_*.json scores (11 files)
  models/                      Official CUDA A estimator and frozen TF-IDF only
    cuda_A_20261115/           cuml_logreg.joblib, training.json, kernel proof
    tfidf.joblib
  data/                        Split and label map (gold jsonl is gitignored)
  scripts/                     eval_three_arms.py, summarize_eval_times.py,
                               scoring helpers, and offline tests
  .env.eval.example            Key names only; copy to .env.eval locally
  requirements.txt
  .venv/                       Experiment virtualenv
label-studio/                  labeling_config.xml and docker-compose.yml.example
bin/                           CUDA start scripts for each backend
serve/                         Health wrappers and SystemOne serve code
venvs/cuml25_10_cuda13/        cuML environment for official Arm A
data/                          Raw mail archives (gitignored; not required to read results)
```

Raw mail archives and model weight trees may exist on disk for serving; they are gitignored and are not needed just to read the official scores.

## License / privacy

Do not publish raw mail, the gold jsonl, `.env.eval`, API tokens, or model-provider secrets. Copy `label-studio/docker-compose.yml.example` to `docker-compose.yml` locally. The gold file `experiments/data/emails_labeled_gold_masked-2026-11-14-v2.jsonl` stays on disk for scoring and is not in git.
