import os
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://feeds.datagolf.com"
API_KEY = os.getenv("DATAGOLF_API_KEY")

ENDPOINTS = [
    ("preds/skill-ratings",                 {}),
    ("preds/pre-tournament",                {}),
    ("field-updates",                       {"tour": "pga"}),
    ("preds/player-decompositions",         {}),
    ("historical-raw-data/rounds",          {"tour": "pga", "event_id": "14", "year": "2024"}),
    ("historical-event-stats/finishes",     {"tour": "pga", "event_id": "14", "year": "2024"}),
    ("historical-odds/outrights",           {"tour": "pga", "event_id": "14", "year": "2024", "market": "win"}),
    ("betting-tools/outrights",             {"tour": "pga", "market": "win", "odds_format": "percent"}),
]

for endpoint, params in ENDPOINTS:
    p = {**params, "key": API_KEY}
    url = f"{BASE_URL}/{endpoint}"
    try:
        r = requests.get(url, params=p, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                keys = list(data.keys())
            elif isinstance(data, list):
                keys = f"list[{len(data)}]"
            else:
                keys = type(data).__name__
            print(f"SUCCESS  {endpoint}  —  keys: {keys}")
        elif r.status_code == 403:
            print(f"BLOCKED  {endpoint}  —  403 Forbidden")
        else:
            print(f"ERROR    {endpoint}  —  HTTP {r.status_code}")
    except Exception as e:
        print(f"ERROR    {endpoint}  —  {e}")
