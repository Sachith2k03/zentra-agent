import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.services.llm_service import generate_response, list_models

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LocalMind AI",
    description="Local AI Agent Backend",
    version="0.1.0",
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "application": "LocalMind AI",
    }


@app.get("/models")
def get_models():
    try:
        return {
            "success": True,
            "models": list_models(),
        }
    except Exception:
        logger.exception("Failed to retrieve models.")
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve models from LM Studio.",
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    try:
        answer = generate_response(request.message)
        return ChatResponse(response=answer)

    except RuntimeError as error:
        logger.warning("AI service error: %s", error)
        raise HTTPException(
            status_code=503,
            detail=str(error),
        )

    except Exception:
        logger.exception("Unexpected chat endpoint error.")
        raise HTTPException(
            status_code=503,
            detail="An unexpected AI service error occurred.",
        )