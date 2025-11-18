import json
import os
import asyncio 
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel
import os

# New SDK imports
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Import local models
from app.models import TradeCreate, Message

# Define the Intent Classification model (must match the enum below)
class IntentResponse(BaseModel):
    intent: Literal["LOG_TRADE", "REVIEW_ANALYSIS", "NEWS_MARKET", "PLAN_STRATEGY", "OTHER"]

class AIService:
    """
    AI Service using the modern google-genai SDK (Gemini 2.5).
    Implements a fast, sequential classification flow to differentiate intents.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Cannot initialize AI client.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"
        
        # New config for ultra-fast classification
        self.classification_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentResponse
        )


    async def classify_intent(self, text: str) -> str:
        """
        STAGE 1: Fast classification step to determine user intent.
        """
        
        prompt = f"""
        Analyze the user's input and classify their PRIMARY intent using only one of the following categories:
        - LOG_TRADE (Explicitly reporting a completed buy/sell action)
        - REVIEW_ANALYSIS (Asking to summarize or analyze past performance)
        - NEWS_MARKET (Asking about current price, news, or market conditions)
        - PLAN_STRATEGY (Asking for advice, future plans, or what to look for)
        - OTHER (Greetings, general chat, or unclear requests)
        
        Return ONLY a JSON object: {{"intent": "CATEGORY"}}
        
        Input: "{text}"
        """
        
        try:
            # Classification is very fast and should use minimal tokens
            response = await self.client.aio.models.generate_content(
                model=self.model_id, 
                contents=prompt,
                config=self.classification_config,
            )
            
            data = json.loads(response.text)
            return data.get("intent", "OTHER").upper()
        except Exception as e:
            print(f"Classification failed: {e}")
            return "OTHER"


    async def extract_trade_from_text(self, text: str) -> Optional[TradeCreate]:
        """
        STAGE 2: Only proceeds with extraction if STAGE 1 determined the intent was LOG_TRADE.
        """
        
        # --- Sequential Intent Check (The key fix for accuracy/speed) ---
        intent = await self.classify_intent(text)
        if intent != "LOG_TRADE":
            # Exit early if intent is not logging, drastically improving accuracy and avoiding cost/latency.
            return None 

        # --- Stage 2: Detailed Extraction (Only runs if intent is LOG_TRADE) ---
        today = datetime.now().strftime('%Y-%m-%d')
        MAX_RETRIES = 3
        
        prompt = f"""
        You are an EXTREMELY STRICT Data Extraction Agent. 
        Your task is to extract the details of the COMPLETED trade described below.
        
        Context: Today is {today}.
        User input: "{text}"
        
        Rules:
        1. Fill the TradeCreate schema completely.
        2. If quantity is missing, default to 1.
        """

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TradeCreate,
                    ),
                )
    
                if response.text and response.text.strip().lower() not in ["null", "none"]:
                    return TradeCreate.model_validate_json(response.text)
                return None
                
            except APIError as e:
                if "503 UNAVAILABLE" in str(e) and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"Extraction Error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Extraction Error: Final attempt failed or unrecoverable error: {e}")
                    raise
            except Exception as e:
                print(f"Extraction Error: {e}")
                raise


    async def generate_chat_response(
        self, 
        user_message: str, 
        chat_history: List[Message], 
        trade_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates a chat response. It intelligently uses the Google Search tool for live data 
        or the internal trade history for personalized analysis.
        """
        MAX_RETRIES = 3
        
        trade_context = "No previous trades available."
        if trade_history:
            trade_context = json.dumps(trade_history[-20:], indent=2, default=str)

        system_instruction = f"""
        You are an expert Trading Journal AI Assistant. Your goal is to be helpful, accurate, and context-aware.

        [YOUR DATA SOURCES]
        1. **Internal Trade History:** {trade_context}
        2. **Google Search Tool:** (Available for real-time info)

        [DECISION PROTOCOL - HOW TO HANDLE REQUESTS]
        
        This model is responsible for the CONVERSATION. The companion extraction model runs separately.
        
        1. **NEWS/MARKET & PLANNING:** If the user is asking about current data or strategy planning, use the `Google Search` tool.
        2. **REVIEW/ANALYSIS:** If the user is asking about their past performance, analyze the **Internal Trade History**.
        3. **LOGGING/OTHER:** Respond conversationally. If a trade was just logged, acknowledge it and comment on the performance (e.g., "Great win!").

        **General Rule:** Be concise. If you use Search, mention that you checked live data.
        """

        contents = []
        for msg in chat_history[-6:]:
            role = "model" if msg.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        
        fallback_message = "I apologize, but my market intelligence service is overloaded right now. Please wait a moment and try again."

        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[types.Tool(google_search=types.GoogleSearch())], 
                        temperature=0.7,
                    )
                )

                is_grounded = False
                if response.candidates and response.candidates[0].grounding_metadata:
                    if hasattr(response.candidates[0].grounding_metadata, 'search_entry_point') and \
                       response.candidates[0].grounding_metadata.search_entry_point is not None:
                        is_grounded = True

                return {
                    "message": response.text,
                    "is_grounded": is_grounded
                }
            except APIError as e:
                if "503 UNAVAILABLE" in str(e) and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    print(f"Chat Generation Error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Chat Generation Error: Final attempt failed or unrecoverable error: {e}")
                    return {"message": fallback_message, "is_grounded": False}
            except Exception as e:
                print(f"Chat Gen Error: {e}")
                return {"message": fallback_message, "is_grounded": False}


    async def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates analysis insights."""
        if not trades:
            return {"summary": "No trades to analyze.", "insights": []}
            
        prompt = f"""Analyze this trade data and return a JSON object with a 'summary' and a list of 'insights'.
        Trades: {json.dumps(trades[:50], default=str)}
        """
        
        try:
            class AnalysisResult(BaseModel):
                summary: str
                insights: List[str]

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnalysisResult
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Analysis Error: {e}")
            return {"summary": "Analysis failed.", "insights": []}

    async def generate_title_for_chat(self, messages: List[Message]) -> Optional[str]:
        """Generates a short title for the chat session."""
        prompt = "Generate a very short (3-5 words) title for this conversation. Return JSON: {'title': '...'}"
        context = "\n".join([f"{m.role}: {m.content}" for m in messages[-5:]])
        
        try:
            class TitleResult(BaseModel):
                title: str

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=f"{prompt}\n\nConversation:\n{context}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TitleResult
                )
            )
            data = json.loads(response.text)
            return data.get("title")
        except Exception:
            return "New Chat"