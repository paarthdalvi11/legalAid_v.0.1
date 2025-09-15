import os
import pickle
import faiss
import cv2
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File
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
GOOGLE_KEYS = os.getenv("GOOGLE_KEYS", "").split(",")  # multiple keys
current_key_index = 0

# === Gemini key rotation setup ===
def set_gemini_key():
    global current_key_index
    if current_key_index >= len(GOOGLE_KEYS):
        raise RuntimeError("❌ All API keys exhausted.")
    key = GOOGLE_KEYS[current_key_index].strip()
    genai.configure(api_key=key)
    return genai.GenerativeModel("gemini-1.5-flash")

def rotate_key():
    global current_key_index
    current_key_index += 1
    return set_gemini_key()

gemini_model = set_gemini_key()

# Retry wrapper
def generate_with_retry(prompt_or_inputs):
    global gemini_model
    while True:
        try:
            print("💡 Sending request to Gemini...")  # message for console
            return gemini_model.generate_content(prompt_or_inputs).text
        except Exception as e:
            print(f"⚠️ Error with key {current_key_index}: {e}")
            gemini_model = rotate_key()
            print(f"🔄 Switched to API key {current_key_index}")

# === FastAPI app ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_ipc(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

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
    return generate_with_retry(prompt)

# === LangChain Tavily setup (unchanged) ===
def setup_langchain_tavily():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_KEYS[0])
    tavily_tool = Tool(
        name="tavily_search",
        description="Deep search tool for IndianKanoon precedents",
        func=TavilySearch(api_key=TAVILY_API_KEY, max_results=5).run,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a professional legal research assistant.
Use the `tavily_search` tool to search only on IndianKanoon.org.
Return structured case details (Court, Facts, Issues, Judgment, etc.)."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm=llm, tools=[tavily_tool], prompt=prompt)
    return AgentExecutor(agent=agent, tools=[tavily_tool], verbose=False)

agent_executor = setup_langchain_tavily()

def get_case_precedents(query, agent_executor):
    return agent_executor.invoke({"input": query})["output"]

# === Schemas ===
class QueryRequest(BaseModel):
    query: str

# === Endpoints ===
@app.post("/legal-assistant")
async def legal_assistant(request: QueryRequest):
    print("📝 Received text query")
    if not index or not ipc_chunks:
        raise HTTPException(status_code=500, detail="Index or chunks not loaded")
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")
    ipc_results = search_ipc(query, embed_model, index, ipc_chunks)
    ipc_analysis = ask_gemini_ipc_advice(query, ipc_results)
    precedents = get_case_precedents(query, agent_executor)
    print("✅ Finished processing text query")
    return {"ipc_analysis": ipc_analysis, "precedents": precedents}

@app.post("/legal-assistant-image")
async def legal_assistant_image(file: UploadFile = File(...)):
    print("🖼 Received image file")
    img_bytes = await file.read()
    prompt = """
    You are an image captioner for crime & suspicious activity detection. 
    Provide a short, precise caption describing exactly what is happening.
    - Mention weapons if present.
    - Mention suspicious or violent actions.
    - Be factual, not speculative.
    """
    caption = generate_with_retry([prompt, {"mime_type": file.content_type, "data": img_bytes}])
    print(f"🖼 Caption generated: {caption}")

    ipc_results = search_ipc(caption, embed_model, index, ipc_chunks)
    ipc_analysis = ask_gemini_ipc_advice(caption, ipc_results)
    print("✅ Finished processing image")
    return {"caption": caption, "ipc_analysis": ipc_analysis}

def process_video_and_analyze(video_path):
    print("🎬 Starting video processing...")
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * 3
    frame_count, saved_frames = 0, []

    with tempfile.TemporaryDirectory() as tmpdir:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_file = os.path.join(tmpdir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_file, frame)
                saved_frames.append(frame_file)
            frame_count += 1
        cap.release()

        print(f"🎬 Extracted {len(saved_frames)} frames from video")
        frame_captions = []
        for idx, frame_file in enumerate(saved_frames):
            print(f"🎥 Analyzing frame {idx+1}/{len(saved_frames)}")
            with open(frame_file, "rb") as f:
                image_data = f.read()
            frame_prompt = """
            You are analyzing a video frame showing possible crime or suspicious activity.
            - Short precise caption of what is happening.
            - Mention weapons if visible.
            - Confidence score (0.0 to 1.0).
            Format strictly as:
            Caption: <your caption>
            Confidence: <score>
            """
            frame_caption = generate_with_retry([frame_prompt, {"mime_type": "image/jpeg", "data": image_data}])
            frame_captions.append(frame_caption)

        summary_prompt = f"""
        You are given frame-wise captions from a video.
        Aggregate them into ONE final caption describing the video.
        Include any weapons mentioned even once.
        Frame Captions:
        {frame_captions}
        """
        final_caption = generate_with_retry(summary_prompt)

    ipc_results = search_ipc(final_caption, embed_model, index, ipc_chunks)
    ipc_analysis = ask_gemini_ipc_advice(final_caption, ipc_results)
    print("✅ Finished video analysis")
    return ipc_analysis

@app.post("/legal-assistant-video")
async def analyze_video(file: UploadFile = File(...)):
    print("🎬 Received video file")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    result = process_video_and_analyze(video_path)
    os.remove(video_path)
    return {"ipc_analysis": result}
