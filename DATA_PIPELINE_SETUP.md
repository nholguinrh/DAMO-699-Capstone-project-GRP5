# Data Pipeline Setup — from clean clone to a running `01_eda` notebook

This is the prerequisite guide for anyone who pulls this repo fresh and wants to run
[`notebooks/01_eda/eda.ipynb`](./notebooks/01_eda/eda.ipynb) (or any other notebook that reads
from `data/processed/`). As of the Round 1 merges (#35, #36), **`data/raw/` and `data/processed/`
are gitignored** — they're pipeline output, regenerated locally, not committed to git. A fresh
clone has empty `data/raw/` and `data/processed/` folders (just `.gitkeep` placeholders) until you
run one of the two pipelines below.

If you only need environment/credential setup for **Path B** specifically, [`docs/SETUP.md`](./docs/SETUP.md)
already covers that in more detail. This document's job is to explain both paths, be explicit
about which one you need for the EDA notebook, and cover running the notebook itself — locally or
in Colab.

## 0. Which path do I actually need?

There are two independent data-collection implementations in `src/`, both merged, both valid —
see `M2_CHECKLIST.md` §"Parallel-track model" for why two exist:

| | **Path A** (`src/run_data_collection.py`) | **Path B** (`src/orchestrator.py`) |
|---|---|---|
| Merged via | PR #35 | PR #36 |
| Style | Sequential, one source at a time | Concurrent (`ThreadPoolExecutor`) |
| Produces `data/raw/*.json` (raw API cache) | ✅ | ✅ |
| Produces `data/processed/*.csv` (cleaned) | ✅ | ❌ — Path B intentionally stops at raw caching (Bronze layer); cleaning/processing isn't implemented for it yet |

**`notebooks/01_eda/eda.ipynb` reads `data/processed/*.csv` directly, so you need Path A's
pipeline to have run at least once.** Path B is documented below too, in case you're working on
something that only needs the raw cache, or on extending Path B's cleaning step — but it will not
by itself produce what the EDA notebook needs.

## 1. Common setup (do this once, either path)

**Clone and enter the repo:**

```bash
git clone https://github.com/nholguinrh/DAMO-699-Capstone-project-GRP5.git
cd DAMO-699-Capstone-project-GRP5
```

**Python:** 3.11+ recommended (matches the CI workflow's version). A virtual environment is
recommended but not required:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**Get a free FRED API key** (both paths need it, ~2 minutes):

1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Request a key (account required, free)
3. Copy the 32-character key

**Create your `.env` file** in the project root (same folder as this file):

```bash
cp .env.example .env
```

Then edit `.env` and paste your key:

```
FRED_API_KEY=your-actual-32-character-key-here
```

`.env` is gitignored — it never gets committed, and each teammate keeps their own. Only
`.env.example` (the placeholder template) is tracked in git.

## 2. Running Path A (produces `data/processed/*.csv` — needed for `01_eda`)

**Install dependencies:**

```bash
pip install pandas requests beautifulsoup4 pdfplumber python-dotenv
```

(`beautifulsoup4` and `pdfplumber` are only used by `src/build_cpi_release_dates.py`, the CPI
release-date scraper — not by the core pull/clean pipeline. Install them anyway; it's one line.)

**Run the full pipeline** (live API calls to BoC, FRED, and StatCan, in sequence):

```bash
python src/run_data_collection.py
```

This writes:
- `data/raw/bank_of_canada/`, `data/raw/fred/`, `data/raw/statcan/` — timestamped raw JSON
- `data/processed/bank_of_canada_data.csv`
- `data/processed/fred_rates.csv`
- `data/processed/statcan_cpi.csv`

**Rebuild from a previous run without hitting the APIs again:**

```bash
python src/run_data_collection.py --from-cache
```

Full details, including the `--refresh-cpi-dates` flag and troubleshooting, are in
[`src/README.md`](./src/README.md).

## 3. Running Path B (raw cache only — does not feed `01_eda` today)

**Install dependencies:**

```bash
pip install python-dotenv requests
```

**Run it** — either via the notebook:

```bash
cd notebooks
jupyter notebook 02_cleaning_path_b.ipynb
```

or directly from Python, from the project root:

```python
from src.orchestrator import run_full_collection
import src.config as cfg

results = run_full_collection(cfg)
```

This writes timestamped raw JSON to `data/raw/` (three sources fetched concurrently) but does
**not** produce `data/processed/*.csv`. Full details in [`docs/SETUP.md`](./docs/SETUP.md).

## 4. Running `notebooks/01_eda/eda.ipynb`

The notebook resolves its own project root relative to its own location
(`Path.cwd().resolve().parents[1]`), which assumes the notebook's working directory is
`notebooks/01_eda/` — true by default in classic Jupyter/JupyterLab, since they set the kernel's
cwd to the notebook's folder.

### Option A — locally

After running Path A (§2) at least once so `data/processed/*.csv` exists:

```bash
pip install jupyter matplotlib numpy pandas seaborn statsmodels
cd notebooks/01_eda
jupyter notebook eda.ipynb
```

Run all cells top-to-bottom from a fresh kernel (Kernel → Restart & Run All). No further
configuration needed — the `.env` file is only read by the data-collection step, not by the
notebook itself.

### Option B — Google Colab (colab.google.com)

Colab doesn't have the repo or your `.env` locally, and its default working directory (`/content`)
doesn't match what the notebook expects, so there are two extra steps versus running locally.

1. **Open a blank Colab notebook** and clone the repo. This is a **private** repo, so you'll need
   a [GitHub personal access token](https://github.com/settings/tokens) (classic token, `repo`
   scope is enough) — generate one, then in a Colab cell:

   ```python
   import getpass
   token = getpass.getpass("GitHub token: ")
   !git clone https://{token}@github.com/nholguinrh/DAMO-699-Capstone-project-GRP5.git
   %cd DAMO-699-Capstone-project-GRP5
   ```

   (Using `getpass` keeps the token out of the notebook's saved output/history — don't paste it
   directly into a cell.)

2. **Install dependencies** (Colab preinstalls numpy/pandas/matplotlib/seaborn/statsmodels, but
   pin them explicitly to be safe, plus the pipeline-only packages):

   ```python
   !pip install -q pandas requests beautifulsoup4 pdfplumber python-dotenv seaborn statsmodels
   ```

3. **Set your FRED API key** for this session — simplest to set it directly as an environment
   variable rather than writing a `.env` file, since `python-dotenv`'s loader falls back to
   whatever's already in `os.environ`:

   ```python
   import os, getpass
   os.environ["FRED_API_KEY"] = getpass.getpass("FRED API key: ")
   ```

4. **Run Path A** to populate `data/processed/`:

   ```python
   !python src/run_data_collection.py
   ```

5. **Point the notebook at the right working directory.** Either open `eda.ipynb` from the
   cloned repo via Colab's file browser (left sidebar → files → navigate into
   `DAMO-699-Capstone-project-GRP5/notebooks/01_eda/eda.ipynb` → double-click), which Colab will
   run in the same runtime/filesystem as the steps above — or, if you opened the notebook a
   different way (e.g. Colab's GitHub import dialog), add a cell **before** the notebook's first
   code cell:

   ```python
   %cd /content/DAMO-699-Capstone-project-GRP5/notebooks/01_eda
   ```

   so `Path.cwd().resolve().parents[1]` resolves to the repo root, matching what the notebook
   expects locally.

6. Run all cells top-to-bottom.

## Troubleshooting

| Problem | Fix |
|:--------|:----|
| `FileNotFoundError: Could not locate the project root` (Path A) | You're not inside a full clone (missing `.git`). Confirm you cloned rather than downloaded a zip. |
| `EnvironmentError` / `ValueError`: FRED API key not found | `.env` is missing/empty, or (Colab) you skipped the `os.environ["FRED_API_KEY"]` step. |
| `ModuleNotFoundError: No module named 'dotenv'` | `pip install python-dotenv`. |
| `FileNotFoundError` on `data/processed/*.csv` when opening `eda.ipynb` | You ran Path B instead of Path A, or haven't run either yet — see §0. |
| Notebook's `PROJECT_ROOT` resolves to the wrong folder | Working directory isn't `notebooks/01_eda/`. Locally: reopen via Jupyter's file browser rather than a bare `jupyter notebook` from elsewhere. Colab: see step 5 above. |
| `HTTP 429` / rate-limit warnings during collection | Normal — both pipelines auto-retry with exponential back-off. Let it run. |
