import os
import json
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://feeds.datagolf.com"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _cache_filename(endpoint: str, params: dict) -> Path:
    """Build a deterministic cache filename from endpoint + params."""
    safe_endpoint = endpoint.replace("/", "_")
    param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()) if k != "key")
    name = f"{safe_endpoint}__{param_str}.json" if param_str else f"{safe_endpoint}.json"
    return RAW_DIR / name


def pull_endpoint(endpoint: str, params: dict | None = None, force: bool = False) -> dict | list:
    """
    Fetch a DataGolf API endpoint, using a local cache when available.

    Parameters
    ----------
    endpoint : str
        Path after the base URL, e.g. "historical-raw-data/rounds".
    params : dict, optional
        Query parameters (API key is injected automatically).
    force : bool
        If True, skip the cache and always hit the API.

    Returns
    -------
    dict or list
        Parsed JSON response.
    """
    api_key = os.getenv("DATAGOLF_API_KEY")
    if not api_key or api_key == "placeholder":
        raise EnvironmentError(
            "DATAGOLF_API_KEY is not set. Add your real key to .env."
        )

    params = dict(params or {})
    cache_path = _cache_filename(endpoint, params)

    if cache_path.exists() and not force:
        print(f"[cache] {cache_path.name}")
        with open(cache_path, "r") as f:
            return json.load(f)

    params["key"] = api_key
    url = f"{BASE_URL}/{endpoint}"
    print(f"[fetch] GET {url}  params={params}")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    with open(cache_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[saved] {cache_path.name}")

    time.sleep(1.5)
    return data
