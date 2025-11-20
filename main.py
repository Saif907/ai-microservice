# ai-microservice/main.py

import uvicorn
import os
from dotenv import load_dotenv 
from fastapi import FastAPI, HTTPException, status
from pydantic_settings import BaseSettings
from typing import Optional

# --- FIX: Environment Loading ---
# Load .env file contents immediately to set variables for configuration
load_dotenv() 

# Import the final, selected AIService class from the selector file
# This class will be either the GeminiService or the GrokService
from app.provider_selector import AIService

# Import Models (these are consistent across both services)
from app.models import (
    AIMessageResponse, ChatProcessRequest, TradeExtractionRequest, 
    TradeAnalysisRequest, InsightsResponse, TitleGenerationRequest, 
    TitleResponse, TradeCreate
)

# --- Configuration Loader ---

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    
    # Keys are checked here for Pydantic validation, but actual client config happens inside AIService
    GEMINI_API_KEY: Optional[str] = None
    XAI_API_KEY: Optional[str] = None 
    GROQ_API_KEY: Optional[str] = None
    
settings = Settings()

# --- Instantiate the Selected Service ---
try:
    # This initializes the chosen AI service (Gemini or Grok) after configs are loaded
    ai_service = AIService()
    
    # This simple attribute lets the health check report the current provider name
    # We use __name__ to get the actual class name (e.g., 'AIService', 'GrokService')
    service_name = AIService.__name__
    
except Exception as e:
    print(f"❌ FATAL CONFIG ERROR: Could not initialize AI service: {e}")
    os._exit(1) # Exit cleanly if service initialization fails

app = FastAPI(title="AI Microservice (Multi-Model)", version="2.1.0")

# --- Router Endpoints (Directly call the selected ai_service instance) ---

@app.post("/ai/process-chat", response_model=AIMessageResponse)
async def process_chat(request: ChatProcessRequest):
    """
    Handles chat, extraction, and grounding by routing to the selected provider.
    """
    try:
        chat_result = await ai_service.generate_chat_response(
            request.user_message, 
            request.chat_history, 
            request.trade_history
        )
        # Note: Extraction is run concurrently with conversation inside the service
        extracted_trade = await ai_service.extract_trade_from_text(request.user_message)
        
        return AIMessageResponse(
            message=chat_result["message"],
            is_grounded=chat_result["is_grounded"],
            trade_extracted=extracted_trade
        )

    except Exception as e:
        print(f"❌ Error in process_chat: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI Microservice Error: {str(e)}")

@app.post("/ai/extract-trade", response_model=Optional[TradeCreate])
async def extract_trade_endpoint(request: TradeExtractionRequest):
    """Direct endpoint to extract trade data (e.g., for quick logging forms)."""
    try:
        return await ai_service.extract_trade_from_text(request.text)
    except Exception as e:
        print(f"❌ Error in extract_trade_endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI Microservice Error: {str(e)}")


@app.post("/ai/analyze-trades", response_model=InsightsResponse)
async def analyze_trades_endpoint(request: TradeAnalysisRequest):
    """Generates analysis insights from a list of trades."""
    try:
        result = await ai_service.analyze_trades(request.trades)
        return InsightsResponse(
            summary=result.get("summary", "Analysis unavailable"),
            insights=result.get("insights", [])
        )
    except Exception as e:
        print(f"❌ Error in analyze_trades_endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI Microservice Error: {str(e)}")


@app.post("/ai/generate-title", response_model=TitleResponse)
async def generate_title_endpoint(request: TitleGenerationRequest):
    """Generates a short title for a conversation."""
    try:
        title = await ai_service.generate_title_for_chat(request.messages)
        return TitleResponse(title=title or "New Chat")
    except Exception as e:
        print(f"❌ Error in generate_title_endpoint: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"AI Microservice Error: {str(e)}")


@app.get("/health")
def health():
    """Health check for the microservice."""
    return {"status": "ok", "service": "ai-microservice", "provider": service_name}


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=settings.HOST, 
        port=settings.PORT, 
        reload=True,
        log_level="info", 
        factory=False
    )