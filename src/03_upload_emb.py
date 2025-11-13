import os
import json
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_SEARCH_INDEX")
key = os.getenv("AZURE_SEARCH_KEY") or os.getenv("AZURE_SEARCH_API_KEY")

if not all([endpoint, index_name, key]):
    raise ValueError("❌ Please set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY (or API_KEY), and AZURE_SEARCH_INDEX in the .env file!")

# -------------------------------
# Connect to Azure AI Search
# -------------------------------
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(key)
)

# -------------------------------
# Load embedded messages
# -------------------------------
INPUT_FILE = os.path.join("..", "output", "messages_embedded.json")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)

print(f"✅ Loaded {len(messages)} embedded messages")

# -------------------------------
# Upload to Azure AI Search
# -------------------------------
for msg in messages:
    try:
        # Match index schema
        doc = {
            "id": msg["id"],
            "user_id": msg.get("user_id"),
            "user_name": msg.get("user_name"),
            "timestamp": msg.get("timestamp"),
            "content": msg.get("content"),
            "content_vector": [float(x) for x in msg["content_vector"]],
        }

        # Upload document
        search_client.upload_documents(documents=[doc])
        print(f"✅ Uploaded: {msg['id']}")

    except Exception as e:
        print(f"❌ Error uploading {msg.get('id')}: {e}")

print("\n🎯 All documents uploaded successfully!")
