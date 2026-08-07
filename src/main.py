import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.Agent.build_graph import trendly_agent

app = FastAPI(title="Trendly Support Agent API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # client can omit on first call


class ChatResponse(BaseModel):
    reply: str
    session_id: str  # echoed back so the client persists it for later turns


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    session_id = req.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    try:
        result = trendly_agent.invoke(
            {"messages": [("user", req.message)]}, config=config
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"agent error: {e}")

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/health")
def health():
    return {"status": "ok"}



FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")