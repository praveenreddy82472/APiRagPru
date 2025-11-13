# src/fetch_messages.py
import os
import json
import uuid
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ----------- Paths -----------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # go one level up to root
OUT_DIR = os.path.join(BASE_DIR, "output")
OUT_FILE = os.path.join(OUT_DIR, "messages_raw.json")
os.makedirs(OUT_DIR, exist_ok=True)

# ----------- API Setup -----------
API_URL = os.getenv(
    "MESSAGES_BASE_URL",
    "https://november7-730026606190.europe-west1.run.app/messages"
)

print(">>> Fetching member messages from API...")
resp = requests.get(API_URL, timeout=30)

if resp.status_code != 200:
    raise RuntimeError(f"Failed to fetch data: {resp.status_code} {resp.text[:200]}")

data = resp.json()

# Handle structure like { "total": ..., "items": [...] }
if isinstance(data, dict) and "items" in data:
    data = data["items"]

# Add unique IDs
for d in data:
    if not d.get("id"):
        d["id"] = str(uuid.uuid4())

print(f"✅ Retrieved {len(data)} records")

# Save to output folder at root
with open(OUT_FILE, "w", encoding="utf8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved messages to {OUT_FILE}")
