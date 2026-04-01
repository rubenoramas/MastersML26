# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MastersML26** predicts 2026 Masters Tournament outcomes for every player in the field. End-to-end sports data science pipeline using the DataGolf API.

## Setup

```bash
pip install -r requirements.txt
# Add your real API key to .env:
# DATAGOLF_API_KEY=your_key_here
```

## Running notebooks

```bash
jupyter notebook notebooks/
```

Run a single notebook non-interactively:
```bash
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb
```

## Architecture

```
src/api_client.py          # DataGolf API wrapper (caching, rate-limiting)
data/raw/                  # Auto-cached JSON responses (git-ignored)
data/processed/            # Cleaned DataFrames
data/features/             # Feature matrices ready for modeling
notebooks/                 # Numbered, sequential workflow notebooks
outputs/csv/               # Exported prediction tables
outputs/figures/           # Saved plots
```

### API client (`src/api_client.py`)

`pull_endpoint(endpoint, params, force=False)` is the single entry point for all DataGolf calls:
- Loads `DATAGOLF_API_KEY` from `.env` via `python-dotenv`
- Checks `data/raw/<endpoint>__<params>.json` before hitting the network
- Sleeps 1.5 s between live requests
- Pass `force=True` to bypass cache and refresh

### Notebook workflow

Notebooks are numbered to reflect the pipeline order:
- `01_data_collection.ipynb` — fetch raw data from three DataGolf endpoints and inspect structure
