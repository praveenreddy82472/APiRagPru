from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

# load environment
load_dotenv()

# import chain
try:
    from app.query_langchain import qa_chain
except Exception as e:
    print("Could not load RAG chain:", e)
    qa_chain = None

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

@app.get("/")
async def home():
    return {"message": "API running"}

@app.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    if not qa_chain:
        return QueryResponse(answer="❌ QA chain not loaded")

    result = qa_chain.invoke({"query": req.question})
    return QueryResponse(answer=result.get("result"))
