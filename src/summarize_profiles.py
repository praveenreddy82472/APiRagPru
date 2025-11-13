import os
import json
from collections import defaultdict
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

# -------------------------------
# Load environment variables
# -------------------------------
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
)

MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o-mini")

# -------------------------------
# Load messages
# -------------------------------
INPUT_FILE = os.path.join("..", "output", "messages_raw.json")
OUTPUT_FILE = os.path.join("..", "output", "user_profiles.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)

print(f"✅ Loaded {len(messages)} messages")

# -------------------------------
# Group by user_name
# -------------------------------
user_messages = defaultdict(list)
for msg in messages:
    user = msg.get("user_name", "Unknown")
    user_messages[user].append(msg["message"])

print(f"📊 Found {len(user_messages)} unique users")

# -------------------------------
# Summarize each user
# -------------------------------
summaries = []

for user, msgs in tqdm(user_messages.items(), desc="Summarizing users"):
    joined_text = "\n".join(msgs[:50])  # limit to first 50 messages per user for speed
    prompt = f"""
    You are an analyst summarizing member behavior.
    Below are messages from {user}.
    Summarize this person’s key interests, habits, and requests.
    Be concise and objective.

    Messages:
    {joined_text}

    Summary:
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250
        )

        summary_text = response.choices[0].message.content.strip()

        summaries.append({
            "id": f"profile-{user.replace(' ', '_')}",
            "user_name": user,
            "content": summary_text,
        })

    except Exception as e:
        print(f"❌ Error summarizing {user}: {e}")

# -------------------------------
# Save to file
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summaries, f, ensure_ascii=False, indent=2)

print(f"✅ Saved {len(summaries)} user summaries to {OUTPUT_FILE}")
