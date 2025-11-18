# ai-microservice/main.py
from dotenv import load_dotenv  # Recommended for programmatic loading
import uvicorn
import os

from fastapi import FastAPI, HTTPException, status
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables early so the AI service can read them when instantiated
load_dotenv()

# Import models and service class from the local 'app' package
from app.ai_service import AIService
from app.models import (
    AIMessageResponse, 
    ChatProcessRequest, 
    TradeExtractionRequest, 
    TradeAnalysisRequest, 
    InsightsResponse, 
    TitleGenerationRequest, 
    TitleResponse,
    TradeCreate
)

# --- Configuration Loader ---

class Settings(BaseSettings):
    """Loads settings from environment variables (e.g., .env file)."""
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # This key is only needed by the AI service
    GEMINI_API_KEY: str 
    
    class Config:
        # Load environment variables from the .env file (fallback)
        env_file = ".env"

settings = Settings()

# Instantiate the AI service AFTER settings and dotenv are loaded
ai_service = AIService()

# --- FastAPI Application Setup ---

app = FastAPI(
    title="AI Microservice (GenAI 2.0)", 
    version="2.0.0",
    description="Dedicated AI service using Google GenAI SDK with Search Grounding"
)

# --- Router Endpoints (Called via Proxy from Main Backend) ---

@app.post("/ai/process-chat", response_model=AIMessageResponse)
async def process_chat(request: ChatProcessRequest):
    """
    Orchestrates the chat interaction: Generates response, handles Google Search, and extracts trade.
    """
    try:
        # 1. Generate chat response (handles search/grounding) and extract trade concurrently
        chat_result = await ai_service.generate_chat_response(
            request.user_message, 
            request.chat_history, 
            request.trade_history
        )
        extracted_trade = await ai_service.extract_trade_from_text(request.user_message)
        
        # 2. Return combined response
        return AIMessageResponse(
            message=chat_result["message"],
            is_grounded=chat_result["is_grounded"],
            trade_extracted=extracted_trade
        )

    except Exception as e:
        print(f"❌ Process Chat Error: {e}")
        # Return a 500 status on internal AI error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"AI Microservice Internal Error: {str(e)}"
        )

@app.post("/ai/extract-trade", response_model=Optional[TradeCreate])
async def extract_trade_endpoint(request: TradeExtractionRequest):
    """Direct endpoint to extract trade data from any text."""
    return await ai_service.extract_trade_from_text(request.text)

@app.post("/ai/analyze-trades", response_model=InsightsResponse)
async def analyze_trades_endpoint(request: TradeAnalysisRequest):
    """Generates insights from a list of trades."""
    result = await ai_service.analyze_trades(request.trades)
    return InsightsResponse(
        summary=result.get("summary", "Analysis unavailable"),
        insights=result.get("insights", [])
    )

@app.post("/ai/generate-title", response_model=TitleResponse)
async def generate_title_endpoint(request: TitleGenerationRequest):
    """Generates a short title for a conversation."""
    title = await ai_service.generate_title_for_chat(request.messages)
    return TitleResponse(title=title or "New Chat")

@app.get("/health")
def health():
    """Health check for the microservice."""
    return {"status": "ok", "service": "ai-microservice", "model": "gemini-2.5-flash"}

# --- Execution ---

if __name__ == "__main__":
    # Use multiprocessing.freeze_support() on Windows if you plan to use --reload
    try:
        import multiprocessing
        multiprocessing.freeze_support()
    except Exception:
        pass

    # Run the app. Avoid enabling reload here if you prefer single-process behavior.
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True,
        log_level="info", 
        factory=False
    )
