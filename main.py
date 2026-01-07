import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import uuid
import base64
import hashlib
from io import BytesIO
from langchain_ollama import ChatOllama
from fastapi.concurrency import run_in_threadpool

SUMMARY_CACHE: Dict[str, dict] = {}

def _cache_key(data: dict) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()
# ----------------------------
# SHARED SESSION STORE (file-based)
# ----------------------------
SESSIONS_FILE = os.path.join(os.getcwd(), "sessions.json")

def save_session_to_file(session_id: str, data: dict):
    """Save session data to a shared JSON file."""
    try:
        sessions = {}
        if os.path.exists(SESSIONS_FILE):
            with open(SESSIONS_FILE, "r", encoding="utf-8") as fh:
                sessions = json.load(fh)
    except Exception:
        sessions = {}
    sessions[session_id] = data
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as fh:
            json.dump(sessions, fh)
    except Exception as e:
        print(f"Failed to save session file: {e}")

def load_session_from_file(session_id: str):
    """Load session data saved by another service."""
    if not os.path.exists(SESSIONS_FILE):
        return None
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as fh:
            sessions = json.load(fh)
        return sessions.get(session_id)
    except Exception as e:
        print(f"Failed to load session file: {e}")
        return None


# Local imports
from parsers.pdf_parser import parse_pdf
from utils.file_utils import read_csv, read_excel
from summarizer.summary_agent import generate_summary
from models.predictor import predict_kpis
from visualizer.chart_generator import generate_charts
from recommender.suggestor import get_recommendations
from qa_agent.qa_bot import get_answer

# ----------------------------
# APP INIT
# ----------------------------
app = FastAPI(title="AI Report Insight Agent", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 

# ----------------------------
# GLOBAL CONTEXT STORE
# ----------------------------
# Dict[session_id, {"context": str, "chart": BytesIO, "summary": str}]
context_store: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# REQUEST MODELS
# ----------------------------
class ChatRequest(BaseModel):
    session_id: str
    content: str


# ----------------------------
# FILE UPLOAD ENDPOINT
# ----------------------------
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    data = {}

    if ext == "pdf":
        file_bytes = await file.read()
        data = parse_pdf(file_bytes)
    elif ext == "csv":
        df = read_csv(file.file)
        data = {"df": df, "raw_text": df.to_string()}
    elif ext in ["xlsx", "xls"]:
        df = read_excel(file.file)
        data = {"df": df, "raw_text": df.to_string()}
    else:
        return JSONResponse({"error": "Unsupported file format"}, status_code=400)

    if not data:
        return JSONResponse({"error": "Parsing failed"}, status_code=500)

    return await run_in_threadpool(process_summary, data)


# ----------------------------
# JSON UPLOAD ENDPOINT
# ----------------------------
@app.post("/upload-json")
async def upload_json(request: Request):
    try:
        payload = await request.json()
    except json.JSONDecodeError as e:
        return JSONResponse(
            {"error": f"Invalid JSON format: {e.msg} (line {e.lineno}, column {e.colno})"},
            status_code=422
        )

    if isinstance(payload, list):
        data = {"json_data": payload}
    elif isinstance(payload, dict):
        if "json_data" in payload:
            data = {"json_data": payload["json_data"]}
        else:
            data = {"json_data": [payload]}
    else:
        return JSONResponse(
            {"error": "Invalid JSON format. Must be an object or an array"},
            status_code=400
        )

    return await run_in_threadpool(process_summary, data)


# ----------------------------
# COMMON PROCESS FUNCTION
# ----------------------------
def process_summary(data: dict):
    """
    Generate summary, charts, predictions and store session context.
    Optimized for GCP performance and safety.
    """

    # ----------------------------
    # CACHE CHECK (HUGE SPEED-UP)
    # ----------------------------
    cache_key = _cache_key(data)
    if cache_key in SUMMARY_CACHE:
        cached = SUMMARY_CACHE[cache_key]

        # new session id, same result
        session_id = str(uuid.uuid4())
        context_store[session_id] = cached["context_store"]

        save_session_to_file(session_id, cached["session_file"])

        response = cached["response"].copy()
        response["session_id"] = session_id
        return response

    # ----------------------------
    # GENERATE SUMMARY (HEAVY)
    # ----------------------------
    full_response = generate_summary(data)

    # ----------------------------
    # CONTEXT (LIMIT SIZE)
    # ----------------------------
    if "raw_text" in data:
        context_text = data["raw_text"][:8000]  # limit context
    elif "json_data" in data:
        context_text = json.dumps(data["json_data"], indent=2)[:8000]
    else:
        context_text = ""

    session_id = str(uuid.uuid4())

    # ----------------------------
    # CLEAN SUMMARY EXTRACTION
    # ----------------------------
    summary_part = full_response
    if "#### Chart Data:" in full_response:
        summary_part = full_response.split("#### Chart Data:")[0].strip()

    # ----------------------------
    # AUX PROCESSING (SAFE)
    # ----------------------------
    try:
        prediction = predict_kpis(data)
    except Exception as e:
        prediction = {"error": str(e)}

    try:
        suggestions = get_recommendations(data)
    except Exception as e:
        suggestions = {"error": str(e)}

    # ----------------------------
    # CHART GENERATION (OPTIONAL)
    # ----------------------------
    chart_obj = None
    chart_base64 = None
    chart_error = None

    try:
        chart_result = generate_charts(full_response)
        if chart_result and chart_result.get("chart"):
            chart_obj = chart_result["chart"]
            chart_base64 = base64.b64encode(
                chart_obj.getvalue()
            ).decode("utf-8")
        elif chart_result and chart_result.get("error"):
            chart_error = chart_result["error"]
    except Exception as e:
        chart_error = str(e)

    # ----------------------------
    # CONTEXT STORE
    # ----------------------------
    context_payload = {
        "context": context_text + "\n\n" + summary_part,
        "chart": chart_obj,
        "summary": summary_part,
    }

    context_store[session_id] = context_payload

    session_payload = {
        "context": context_text + "\n\n" + summary_part,
        "summary": summary_part,
        "chart": chart_base64,
    }

    save_session_to_file(session_id, session_payload)

    # ----------------------------
    # FINAL RESPONSE
    # ----------------------------
    response = {
        "session_id": session_id,
        "summary": summary_part,
        "predictions": prediction,
        "suggestions": suggestions,
    }

    if chart_base64:
        response["chart"] = chart_base64
        response["chart_url"] = f"/chart/{session_id}"
    elif chart_error:
        response["chart_error"] = chart_error

    # ----------------------------
    # CACHE RESULT
    # ----------------------------
    SUMMARY_CACHE[cache_key] = {
        "response": response,
        "context_store": context_payload,
        "session_file": session_payload,
    }

    return response
# ----------------------------
# CHART ENDPOINT
# ----------------------------
@app.get("/chart/{session_id}")
async def get_chart(session_id: str):
    """
    Retrieve chart image for a specific session.
    """
    session = context_store.get(session_id)
    if not session or not session.get("chart"):
        return JSONResponse({"error": "No chart found for this session"}, status_code=404)
    
    chart_bytes = session["chart"].getvalue()
    return StreamingResponse(BytesIO(chart_bytes), media_type="image/png")


# ----------------------------
# CHAT ENDPOINT
# ----------------------------
@app.post("/chat")
async def chat_with_agent(request: Request):
    """
    Chat with the AI agent about uploaded data.
    Requires valid session_id from upload response.
    """

    # STRICTLY read from JSON body
    payload = await request.json()
    session_id = payload.get("session_id")
    user_message = payload.get("content")

    if not session_id or not user_message:
        return JSONResponse({"error": "session_id and content are required"}, status_code=400)

    if session_id not in context_store:
        return JSONResponse(
            {"error": "Invalid session_id. Please upload a document first."},
            status_code=400
        )

    context_data = context_store[session_id]["context"]

    # Get answer from QA agent (NO OTHER LOGIC CHANGED)
    response = get_answer(user_message, context_data, session_id)

    return {
        "session_id": session_id,
        "content": user_message,
        "response": response
    }

@app.on_event("startup")
def warmup_llm():
    try:
        llm = ChatOllama(
            model="mistral:7b-instruct",
            temperature=0
        )
        llm.invoke("READY")
        print("LLM warm-up complete")
    except Exception as e:
        print("LLM warm-up failed:", e)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002)

