import openai, sys
print("Python version:", sys.version)
print("OpenAI version:", openai.__version__)
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
load_dotenv()

client = SearchIndexClient(
    os.getenv("AZURE_SEARCH_ENDPOINT"),
    AzureKeyCredential(os.getenv("AZURE_SEARCH_API_KEY"))
)

index = client.get_index(os.getenv("AZURE_SEARCH_INDEX"))
print(index)
