import os
import pickle
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool

# === Load env vars ===
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# === Configure Gemini ===
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === FastAPI app ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load FAISS + chunks once ===
try:
    index = faiss.read_index("ipc_faiss.index")
    with open("my_chunks.pkl", "rb") as f:
        ipc_chunks = pickle.load(f)
    print("✅ Loaded FAISS index and IPC chunks.")
except Exception as e:
    print(f"❌ Failed to load index/chunks: {e}")
    index = None
    ipc_chunks = []

# === Embedding model ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# === FAISS search ===
def search_ipc(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

# === Gemini analysis ===
def ask_gemini_ipc_advice(query, relevant_chunks):
    prompt = f"""
You are a legal assistant.

**Query:** "{query}"

Based on these legal IPC chunks:
{chr(10).join(f"- {chunk.strip()}" for chunk in relevant_chunks)}

---

Give two outputs:

**Legal Advice**: List all IPCs mentioned, and explain them simply.
**Non-Legal Advice**: Real-world steps the user can take immediately.
"""
    return gemini_model.generate_content(prompt).text

# === LangChain Tavily ===
def setup_langchain_tavily():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY
    )
    tavily_tool = Tool(
        name="tavily_search",
        description="Deep search tool for IndianKanoon precedents",
        func=TavilySearch(api_key=TAVILY_API_KEY, max_results=5).run,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional legal research assistant.
Use the `tavily_search` tool to search only on IndianKanoon.org.
Return structured case details (Court, Facts, Issues, Judgment, etc.).
Do not make up information.
"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm=llm, tools=[tavily_tool], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[tavily_tool], verbose=False)

agent_executor = setup_langchain_tavily()

def get_case_precedents(query, agent_executor):
    return agent_executor.invoke({"input": query})["output"]

# === Request Schema ===
class QueryRequest(BaseModel):
    query: str

# === API endpoint ===
@app.post("/legal-assistant")
async def legal_assistant(request: QueryRequest):
    if not index or not ipc_chunks:
        raise HTTPException(status_code=500, detail="Index or chunks not loaded")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    ipc_results = search_ipc(query, embed_model, index, ipc_chunks)
    ipc_analysis = ask_gemini_ipc_advice(query, ipc_results)
    precedents = get_case_precedents(query, agent_executor)
    return {"ipc_analysis": ipc_analysis, "precedents": precedents}
