# Environment Setup — Data-Collection Pipeline

This guide explains how to set up the API credentials and dependencies
needed to run the data-collection pipeline (`src/orchestrator.py` and
`notebooks/02_cleaning_path_b.ipynb`).

## 1. Install Python dependencies

```bash
pip install python-dotenv requests
```

> `requests` is likely already installed. `python-dotenv` loads API keys
> from a local `.env` file so they never appear in code or git history.

## 2. Get a FRED API key (free, ~2 minutes)

The U.S. Federal Reserve FRED API requires a free personal key:

1. Go to **https://fred.stlouisfed.org/docs/api/api_key.html**
2. Click **"Request API Key"** (create an account if needed)
3. Copy the key (a 32-character hex string)

## 3. Create your `.env` file

In the **project root** (same folder as `README.md`), create a file
called `.env` by copying the template:

```bash
# Linux / macOS / Git Bash
cp .env.example .env

# PowerShell
Copy-Item .env.example .env
```

Then open `.env` and replace the placeholder with your actual key:

```
FRED_API_KEY=your-actual-32-character-key-here
```

> **⚠️ Important:**
> - `.env` is listed in `.gitignore` — it will **never** be committed.
> - Each teammate creates their own `.env` locally.
> - `.env.example` (the template) **is** committed, so everyone knows
>   the format.

## 4. Run the pipeline

```bash
# From the project root
cd notebooks
jupyter notebook 02_cleaning_path_b.ipynb
```

Or run the orchestrator directly from Python:

```python
import sys
sys.path.insert(0, ".")  # if running from project root

from src.orchestrator import run_full_collection
import src.config as cfg

results = run_full_collection(cfg)
```

The pipeline will:
1. Fetch all series from **Bank of Canada**, **FRED**, and **Statistics
   Canada** concurrently (3 threads)
2. Cache each raw API response as a timestamped JSON file in `data/raw/`
3. Print a summary of what was pulled

## 5. Verify it worked

Check that JSON files appeared in `data/raw/`:

```bash
ls data/raw/*.json     # Linux / macOS
dir data\raw\*.json    # PowerShell
```

You should see files named like:
```
boc_overnight_rate__CBC20210_20260804_140500.json
fred_us_treasury_10y__DGS10_20260804_140502.json
statcan_cpi_all_items__41690973_20260804_140501.json
...
```

## Troubleshooting

| Problem | Fix |
|:--------|:----|
| `EnvironmentError: FRED API key not found` | Your `.env` file is missing or `FRED_API_KEY` is not set in it. Follow steps 2–3 above. |
| `ModuleNotFoundError: No module named 'dotenv'` | Run `pip install python-dotenv` |
| `ModuleNotFoundError: No module named 'src'` | Make sure you run from the **project root**, or add it to `sys.path` as shown above. |
| `HTTP 429` / rate-limit warnings in the log | Normal — the client auto-retries with exponential back-off. Just wait. |
