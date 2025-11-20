import json
import os
import asyncio 
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime
from pydantic import BaseModel

# --- Official Google GenAI SDK ---
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Local Imports ---
from app.models import TradeCreate, Message

# Define the Intent Classification model
class IntentResponse(BaseModel):
    intent: Literal["LOG_TRADE", "REVIEW_ANALYSIS", "NEWS_MARKET", "PLAN_STRATEGY", "OTHER"]

class AIService:
    """
    AI Service using Google Gemini 2.5.
    Features: Sequenti+al Classification, Robust Error Handling, and Google Search Grounding.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Cannot initialize AI client.")
        
        self.client = genai.Client(api_key=api_key)
        # Gemini 2.5 Flash is fast and cost-effective
        self.model_id = "gemini-2.5-flash"
        
        # Config for fast intent classification
        self.classification_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=IntentResponse
        )


    async def classify_intent(self, text: str) -> str:
        """
        STAGE 1: Fast classification of user intent.
        Returns 'OTHER' if the API fails, ensuring no crashes.
        """
        MAX_RETRIES = 2
        
        prompt = f"""
        Classify the user's PRIMARY intent.
        Categories: 
        - LOG_TRADE (User is explicitly reporting a completed trade)
        - REVIEW_ANALYSIS (User wants to analyze past performance)
        - NEWS_MARKET (User is asking about current prices or news)
        - PLAN_STRATEGY (User is asking for advice or planning)
        - OTHER (General chat)
        
        Input: "{text}"
        """
        
        for attempt in range(MAX_RETRIES):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_id, 
                    contents=prompt,
                    config=self.classification_config,
                )
                
                data = json.loads(response.text)
                return data.get("intent", "OTHER").upper()
            
            except Exception as e:
                # If temporary error, retry. If final error, return safe default.
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(1)
                else:
                    print(f"⚠️ Classification Failed: {e}")
                    return "OTHER"
        
        return "OTHER"


    async def extract_trade_from_text(self, text: str) -> Optional[TradeCreate]:
        """
        STAGE 2: Extracts trade data. 
        - Only runs if intent is LOG_TRADE.
        - Returns None if extraction fails (prevents 500 errors).
        """
        
        # 1. Sequential Intent Check
        intent = await self.classify_intent(text)
        if intent != "LOG_TRADE":
            return None 

        # 2. Detailed Extraction
        today = datetime.now().strftime('%Y-%m-%d')
        MAX_RETRIES = 3
        
        prompt = f"""
        You are a strict Data Extraction Agent. 
        Context Date: {today}.
        User Input: "{text}"
        
        Task: Extract details of the COMPLETED trade.
        Rules:
        - Return valid JSON matching TradeCreate schema.
        - Default quantity to 1 if missing.
        - Return the string 'null' if no valid trade is found.
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
                # Handle Service Unavailable (503)
                if "503" in str(e) and attempt < MAX_RETRIES - 1:
                    print(f"Extraction 503 Error. Retrying in {2 ** attempt}s...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"❌ Extraction Final Error: {e}")
                    return None
            except Exception as e:
                print(f"❌ Extraction Internal Error: {e}")
                return None
        
        return None


    async def generate_chat_response(
        self, 
        user_message: str, 
        chat_history: List[Message], 
        trade_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates chat response.
        CRITICAL: Forces neutral response for logging to allow backend confirmation.
        """
        MAX_RETRIES = 3
        
        trade_context = "No previous trades available."
        if trade_history:
            # Limit history to save tokens
            trade_context = json.dumps(trade_history[-15:], indent=2, default=str)

        system_instruction = f"""
        You are an expert Trading Journal AI Assistant (Gemini).

        [DATA SOURCES]
        1. **Internal Trade History:** {trade_context}
        2. **Google Search Tool:** (Use for real-time news/prices)

        [PROTOCOL]
        - **NEWS/MARKET:** You MUST use the `Google Search` tool.
        - **ANALYSIS:** Analyze the Internal Trade History.
        - **LOGGING A TRADE:** If the user says "I bought..." or "Log this...", provide a **neutral, short acknowledgment** (e.g., "Understood," "Processing entry," or "Got it").
          **IMPORTANT:** DO NOT confirm the trade details (price, ticker) yourself. The system will auto-generate the confirmation message.
        """

        # Build Chat Context
        contents = []
        for msg in chat_history[-6:]:
            role = "model" if msg.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        
        fallback_msg = "I apologize, but my market intelligence service is overloaded right now. Please try again in a moment."

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

                # Check for Grounding (Search usage)
                is_grounded = False
                if response.candidates and response.candidates[0].grounding_metadata:
                    # Updated check for the new SDK structure
                    if hasattr(response.candidates[0].grounding_metadata, 'search_entry_point') and \
                       response.candidates[0].grounding_metadata.search_entry_point is not None:
                        is_grounded = True
                
                # --- FIX: Better Text Extraction Logic ---
                message_text = ""
                
                # 1. Try direct text property (Standard success path)
                if response.text:
                    message_text = response.text
                
                # 2. Try candidates list if text property is empty (common with search results sometimes)
                elif response.candidates and response.candidates[0].content.parts:
                     # Loop through parts to find text
                     for part in response.candidates[0].content.parts:
                         if part.text:
                             message_text += part.text
                
                # 3. Check if blocked by safety filters
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                     message_text = f"I cannot answer that request. (Safety Block: {response.prompt_feedback.block_reason})"

                # 4. Last Resort - Detailed error if still empty
                if not message_text:
                    print(f"⚠️ Empty Response Debug: {response}") # Print full response to logs for debugging
                    message_text = "I processed your request but could not generate a text response. Please try rephrasing."

                return {
                    "message": message_text,
                    "is_grounded": is_grounded
                }
            
            except APIError as e:
                if "503" in str(e) and attempt < MAX_RETRIES - 1:
                    print(f"Chat 503 Error. Retrying in {2 ** attempt}s...")
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"❌ Chat Final Error: {e}")
                    return {"message": fallback_msg, "is_grounded": False}
            except Exception as e:
                print(f"❌ Chat Internal Error: {e}")
                return {"message": fallback_msg, "is_grounded": False}
        
        return {"message": fallback_msg, "is_grounded": False}


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
        try:
            # Using a simpler prompt construction
            context_text = "\n".join([f"{m.role}: {m.content}" for m in messages[-5:]])
            prompt = f"Generate a 3-5 word title for this conversation. Return JSON: {{'title': '...'}}\n\n{context_text}"
            
            class TitleResult(BaseModel):
                title: str

            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TitleResult
                )
            )
            data = json.loads(response.text)
            return data.get("title")
        except Exception:
            return "New Chat"