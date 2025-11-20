import uvicorn
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import json
import uuid
import base64
from io import BytesIO

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

    return process_summary(data)


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

    return process_summary(data)


# ----------------------------
# COMMON PROCESS FUNCTION
# ----------------------------
def process_summary(data: dict):
    """
    Processes uploaded data and returns summary, predictions, suggestions, and chart.
    All operations are non-blocking and complete immediately.
    """
    # Generate summary (includes chart data)
    full_response = generate_summary(data)

    # Extract context for chat
    context_text = ""
    if "raw_text" in data:
        context_text = data["raw_text"]
    elif "json_data" in data:
        context_text = json.dumps(data["json_data"], indent=2)

    # Create unique session
    session_id = str(uuid.uuid4())

    # Extract summary part (before chart data section)
    summary_part = full_response.split("### Chart Data:")[0] if "### Chart Data:" in full_response else full_response

    # Generate predictions and suggestions
    prediction = predict_kpis(data)
    suggestions = get_recommendations(data)

    # Generate chart (non-blocking now!)
    chart_obj: BytesIO | None = None
    chart_base64: str | None = None
    chart_error: str | None = None
    
    chart_result = generate_charts(full_response)
    if chart_result and chart_result.get("chart"):
        chart_obj = chart_result["chart"]
        chart_bytes = chart_obj.getvalue()
        chart_base64 = base64.b64encode(chart_bytes).decode("utf-8")
    elif chart_result and chart_result.get("error"):
        chart_error = chart_result["error"]

    # Save session context (for chat)
    context_store[session_id] = {
        "context": context_text + "\n\n" + full_response,
        "chart": chart_obj,
        "summary": summary_part
    }

    response = {
        "session_id": session_id,
        "summary": summary_part,
        "predictions": prediction,
        "suggestions": suggestions,
    }

    # Add chart data
    if chart_base64:
        response["chart"] = chart_base64
        response["chart_url"] = f"/chart/{session_id}"
    elif chart_error:
        response["chart_error"] = chart_error

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


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002)

