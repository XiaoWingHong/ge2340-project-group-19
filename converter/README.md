# Mail → Label Studio JSON converter

Convert Outlook `.pst` mail into Label Studio import tasks for email-type
classification (`promo | admin | course | personal | news | other`).

The converter does **not** import into Label Studio. Output is a JSON file
you upload manually. It does **not** overwrite any labeled gold set; gold
labels live under `experiments/data/` and are maintained separately.

Raw input must be a **single `.pst` file** or a **folder of `*.pst` files**
(nested `*.pst` are included). `.eml`, `.olm`, and MBOX are not accepted as
user input.

## Output schema

A JSON array of tasks:

```json
{
  "id": "<stable-id>",
  "data": {
    "from": "...",
    "subject": "...",
    "body": "..."
  }
}
```

Stable id: SHA-256 of `Message-ID` (first 16 hex chars); if missing, hash of
`(from + subject + date)`.

## Setup

```bash
cd converter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python dependencies: the standard library for parsing, HTML stripping, and
dedupe, plus optional `langdetect` for English filtering (a heuristic
fallback is used if it is not installed).

### Outlook `.pst` (LibPST / `readpst`)

`.pst` files are **not** parsed in pure Python. The converter shells out to
**`readpst`** from [LibPST](https://www.five-ten-sg.com/libpst/):

```bash
# Debian / Ubuntu
sudo apt install pst-utils
```

That package provides the `readpst` binary. The converter runs:

```text
readpst -r -S -b -8 -o <tempdir> <file.pst>
```

(`-r` recursive folders, `-S` one file per message, `-b` skip RTF, `-8`
UTF-8.) Extracted files are then parsed internally as `.eml` (or MBOX). Set
the `READPST` environment variable if the binary is not on `PATH`.

`python -m converter --pst-help` reprints these notes.

## How to run

```bash
# Package module form
python -m converter --input path/to/mailbox.pst --output /tmp/labelstudio_tasks.json \
  --sample 2000 --seed 2340

# Or the thin script
python convert.py --input path/to/pst-folder --output /tmp/labelstudio_tasks.json \
  --sample 2000 --seed 2340
```

| Flag | Meaning |
|------|---------|
| `--input` / `-i` | A **single `.pst`**, or a **folder of `*.pst`** (nested `*.pst` are included) |
| `--output` / `-o` | Destination JSON path |
| `--sample N` | After parse → anonymize → English-filter → empty-drop → dedupe, randomly select N tasks (for example `--sample 2000`) |
| `--seed` | RNG seed for `--sample` (default: **2340**) |
| `--no-english-filter` | Keep non-English bodies |
| `--no-anonymize` | Skip PII redaction (not recommended) |
| `--pst-help` | Print `readpst` / LibPST notes |

If `--sample 2000` is set and fewer than 2000 tasks remain after the
pipeline, **all remaining tasks are kept** and a warning is printed.
`--sample` is optional; when omitted, every remaining task is written.

### Examples

```bash
# Single Outlook .pst (requires readpst)
python -m converter -i path/to/mailbox.pst -o /tmp/from_pst.json --sample 2000 --seed 2340

# Folder of .pst files (all *.pst, including nested)
python -m converter -i path/to/pst-folder -o /tmp/from_psts.json --sample 2000 --seed 2340
```

## Pipeline steps

1. **Parse** — `from`, `subject`, `body` (prefer `text/plain`; HTML stripped via stdlib `HTMLParser`). `.pst` goes through `readpst` first.
2. **Anonymize** — emails → `[EMAIL]`; phones → `[PHONE]`; 8-digit student-like IDs (`5–7xxxxxxx`) → `[STUDENT_ID]`; light greeting/sign-off names → `[NAME]`
3. **English-only** — drop non-English bodies when detectable (`langdetect`, else Latin/CJK heuristic). Bodies shorter than 40 characters are kept (too little signal). Use `--no-english-filter` to disable.
4. **Drop** empty bodies
5. **Dedupe** — by stable id (Message-ID hash or from+subject+date hash)
6. **Sample** (optional) — randomly select N tasks with `--sample` and `--seed` (default seed 2340)

## Tests

From this directory, with the venv active:

```bash
python -m unittest discover -s tests -v
```

The unittest file covers `--sample` / seed 2340, undersized-pool warnings,
rejection of `.eml` / `.olm` / MBOX at the public input boundary, `.pst`
folder discovery, and a mock `readpst` extract. It does not write large
dumps into `out/` and does not modify labeled gold data.

## Layout

```
converter/
  README.md
  requirements.txt
  convert.py
  converter/
    __main__.py
    cli.py
    pipeline.py
    parse_pst.py
    parse_eml.py          # internal, after readpst extracts a .pst
    parse_mbox.py         # internal, after readpst extracts a .pst
    anonymize.py
    english.py
    html_text.py
    message.py
  fixtures/               # internal parser fixtures; not public inputs
  tests/
  out/
  .venv/
```

## Privacy notes

- Prefer running on already-exported mail you are allowed to process.
- Keep raw `.pst` / mail off shared disks when possible; store anonymized JSON for labeling.
- Do not commit real mail, tokens, or `.env` files to git.
