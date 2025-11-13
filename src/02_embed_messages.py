"""
Fast Azure OpenAI embedding generator for large message datasets.
Optimized with batching and parallel requests.
"""

import os
import json
import requests
import math
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# Environment setup
# -------------------------------
load_dotenv()

AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_EMB_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_EMB_API_VERSION", "2024-02-01")

if not all([AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT]):
    raise ValueError("❌ Missing Azure OpenAI environment variables in .env")

# Embedding API URL
EMBED_URL = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/embeddings?api-version={AZURE_OPENAI_API_VERSION}"
HEADERS = {"api-key": AZURE_OPENAI_KEY, "Content-Type": "application/json"}

# -------------------------------
# File paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "output", "messages_raw.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "output", "messages_embedded.json")

# -------------------------------
# Load input data
# -------------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf8") as f:
    messages = json.load(f)

print(f"✅ Loaded {len(messages)} messages for embedding")

# -------------------------------
# Parameters
# -------------------------------
BATCH_SIZE = 16            # send 16 texts per request
MAX_WORKERS = 5            # concurrent requests (4–8 is ideal)
CHECKPOINT_EVERY = 500     # save progress every 500 records

# -------------------------------
# Helper: Embed a single batch
# -------------------------------
def embed_batch(batch):
    """Send a batch of up to BATCH_SIZE messages for embedding."""
    try:
        inputs = [m["message"].strip() for m in batch if m.get("message")]
        if not inputs:
            return []

        payload = {"input": inputs}
        r = requests.post(EMBED_URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json().get("data", [])
        results = []
        for msg, emb_obj in zip(batch, data):
            results.append({
                "id": msg["id"],
                "user_id": msg.get("user_id"),
                "user_name": msg.get("user_name"),
                "timestamp": msg.get("timestamp"),
                "content": msg.get("message"),
                "content_vector": emb_obj["embedding"]
            })
        return results
    except Exception as e:
        print(f"❌ Batch failed ({len(batch)} items): {e}")
        return []

# -------------------------------
# Main processing loop
# -------------------------------
embedded = []

batches = math.ceil(len(messages) / BATCH_SIZE)
print(f"⚙️  Processing {batches} batches of size {BATCH_SIZE}...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for i in range(0, len(messages), BATCH_SIZE):
        batch = messages[i:i + BATCH_SIZE]
        futures.append(executor.submit(embed_batch, batch))

    for idx, f in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Embedding")):
        res = f.result()
        embedded.extend(res)

        # Save checkpoint periodically
        if len(embedded) % CHECKPOINT_EVERY < BATCH_SIZE:
            with open(OUTPUT_FILE, "w", encoding="utf8") as f:
                json.dump(embedded, f, ensure_ascii=False, indent=2)
            print(f"💾 Checkpoint saved ({len(embedded)} embeddings)")

# -------------------------------
# Save final output
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf8") as f:
    json.dump(embedded, f, ensure_ascii=False, indent=2)

print(f"✅ Completed {len(embedded)} embeddings → {OUTPUT_FILE}")
