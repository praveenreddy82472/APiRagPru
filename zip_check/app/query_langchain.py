import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

embeddings = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
    model="text-embedding-3-small",
    api_key=os.getenv("AZURE_OPENAI_EMB_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_EMB_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_EMB_API_VERSION")
)

vector_store = AzureSearch(
    azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    azure_search_key=os.getenv("AZURE_SEARCH_API_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embeddings.embed_query,
    content_field="content",
    vector_field="content_vector",
    hybrid_fields=["content"],
    hybrid_weight=0.5
)

retriever = vector_store.as_retriever(search_type="hybrid")
retriever.search_kwargs = {}
retriever.search_kwargs["filters"] = None
retriever.k = 50

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_CHAT_MODEL"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.2,
)

prompt_template = """
You are a professional assistant who analyzes member messages.

Question: {question}
Messages:
{context}

Answer with reasoning:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)
