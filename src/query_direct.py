import os
import requests
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings

# ==========================================================
# Load Environment Variables
# ==========================================================
load_dotenv()

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

AZURE_CHAT_MODEL = os.getenv("AZURE_OPENAI_CHAT_MODEL")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

AZURE_EMB_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT")
AZURE_EMB_KEY = os.getenv("AZURE_OPENAI_EMB_KEY")
AZURE_EMB_ENDPOINT = os.getenv("AZURE_OPENAI_EMB_ENDPOINT")
AZURE_EMB_VERSION = os.getenv("AZURE_OPENAI_EMB_API_VERSION")

# ==========================================================
# Embedding Model
# ==========================================================
emb_model = AzureOpenAIEmbeddings(
    deployment=AZURE_EMB_DEPLOYMENT,
    model="text-embedding-3-small",
    api_key=AZURE_EMB_KEY,
    azure_endpoint=AZURE_EMB_ENDPOINT,
    api_version=AZURE_EMB_VERSION,
)

# ==========================================================
# Azure OpenAI Chat Client
# ==========================================================
chat_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_VERSION,
)

# ==========================================================
# FULL HYBRID RAG FUNCTION (Production Ready)
# ==========================================================
def run_rag(question: str) -> str:

    if not question.strip():
        return "Please enter a question."

    # ------------------------------------------------------
    # 1) Embed the Question -> Vector for Azure Search
    # ------------------------------------------------------
    try:
        query_vector = emb_model.embed_query(question)
    except Exception as e:
        return f"❌ Embedding Error: {e}"

    # ------------------------------------------------------
    # 2) Azure Cognitive Search Request (Hybrid)
    # ------------------------------------------------------
    search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2023-11-01"

    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_SEARCH_API_KEY,
    }

    payload = {
        "search": question,
        "queryType": "semantic",
        "semanticConfiguration": "default",
        "top": 8,

        # vector search
        "vector": query_vector,
        "vectorFields": "content_vector",

        # select fields to retrieve
        "select": "content,user_name,message_time",
    }

    try:
        res = requests.post(search_url, headers=headers, json=payload)
        res.raise_for_status()
    except Exception as e:
        return f"❌ Azure Search Error: {e}"

    docs = res.json().get("value", [])

    # ------------------------------------------------------
    # 3) Build Proper Context for LLM
    # ------------------------------------------------------
    if not docs:
        context = ""
    else:
        parts = []
        for d in docs:
            content = d.get("content", "")
            user = d.get("user_name", "Unknown")
            ts = d.get("message_time", "")

            chunk = f"[{user} @ {ts}]\n{content}"
            parts.append(chunk)

        context = "\n\n----\n\n".join(parts)

    # ------------------------------------------------------
    # 4) Build Prompt (same as your working LangChain version)
    # ------------------------------------------------------
    system_prompt = (
        "You are a professional assistant who analyzes member messages.\n"
        "Use ONLY the given context.\n"
        "Infer logical answers if clues exist.\n"
        "If no information is available, say: 'I don't know based on the available info.'\n"
        "Answer clearly and concisely."
    )

    user_prompt = f"Messages:\n{context}\n\nQuestion: {question}\n\nAnswer:"

    # ------------------------------------------------------
    # 5) Azure OpenAI Chat Completion
    # ------------------------------------------------------
    try:
        completion = chat_client.chat.completions.create(
            model=AZURE_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = completion.choices[0].message.content
        return answer.strip()

    except Exception as e:
        return f"❌ OpenAI Error: {e}"


# ==========================================================
# Run Directly (Local testing)
# ==========================================================
if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("\n💬 Answer:\n")
        print(run_rag(q))
