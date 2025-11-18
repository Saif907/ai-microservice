import json
import os
import asyncio 
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel
import os

# New SDK imports
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Import local models
from app.models import TradeCreate, Message

class AIService:
    """
    AI Service using the modern google-genai SDK (Gemini 2.5).
    Enables Google Search grounding for real-time market data and handles 
    trade extraction and analysis based on user intent.
    """

    def __init__(self):
        # NOTE: API key is loaded from the OS environment via main.py/dotenv
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing. Cannot initialize AI client.")
        
        # Initialize the new Client
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash"

    async def extract_trade_from_text(self, text: str) -> Optional[TradeCreate]:
        """
        Extracts trade details with a highly specialized prompt using FEW-SHOT EXAMPLES
        to correctly classify intent (Logging vs Planning/News).
        """
        today = datetime.now().strftime('%Y-%m-%d')
        MAX_RETRIES = 3
        
        # FEW-SHOT PROMPTING: Showing the model exactly what to do is more powerful than just telling it.
        prompt = f"""
        You are an EXTREMELY STRICT Data Extraction Agent.
        Your task is to identify COMPLETED TRADING ACTIONS for a database log.

        Current Date: {today}

        [EXAMPLES OF INPUTS THAT MUST RETURN NULL]
        - "What is the news for TSLA?" -> null (News query)
        - "Should I buy NVDA?" -> null (Asking for advice)
        - "I am planning to buy AAPL at 200" -> null (Future plan, not a completed trade)
        - "Review my last week's trades" -> null (Analysis request)
        - "AAPL is going up" -> null (Market commentary)
        - "If SPY hits 500 I will sell" -> null (Conditional/Future)

        [EXAMPLES OF INPUTS THAT MUST BE EXTRACTED]
        - "I bought 10 shares of TSLA at 200" -> {{ "ticker": "TSLA", "entry_price": 200, "quantity": 10, "entry_date": "{today}", ... }}
        - "Sold AAPL just now at 150" -> {{ "ticker": "AAPL", "exit_price": 150, ... }}
        - "Long NVDA 500 entry, 550 exit, 5 shares" -> {{ "ticker": "NVDA", "entry_price": 500, "exit_price": 550, "quantity": 5, ... }}

        [YOUR INSTRUCTIONS]
        1. **Analyze User Input:** "{text}"
        2. **Check Intent:** Does this describe a COMPLETED or EXECUTED trade?
           - IF YES -> Return the JSON object.
           - IF NO (News, Plan, Question, Review) -> Return the JSON value: null
        3. **Defaults:** If quantity is missing in a valid trade, default to 1.
        """

        for attempt in range(MAX_RETRIES):
            try:
                # Enforce JSON output matching the TradeCreate schema
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=TradeCreate, # Pass the Pydantic model directly
                    ),
                )
    
                if response.text and response.text.strip().lower() not in ["null", "none"]:
                    return TradeCreate.model_validate_json(response.text)
                return None
                
            except APIError as e:
                # Handle 503 (Service Unavailable) explicitly
                if "503 UNAVAILABLE" in str(e) and attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s...
                    print(f"Extraction Error: {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Extraction Error: Final attempt failed or unrecoverable error: {e}")
                    raise # Re-raise the error for FastAPI to catch
            except Exception as e:
                print(f"Extraction Error: {e}")
                raise # Re-raise the error for FastAPI to catch


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
        
        # 1. Prepare Context (Trade History - Internal Knowledge)
        trade_context = "No previous trades available."
        if trade_history:
            trade_context = json.dumps(trade_history[-20:], indent=2, default=str)

        system_instruction = f"""
        You are an expert Trading Journal AI Assistant. Your goal is to be helpful, accurate, and context-aware.

        [YOUR DATA SOURCES]
        1. **Internal Trade History:**
           {trade_context}
        2. **Google Search Tool:** (Available for real-time info)

        [DECISION PROTOCOL - HOW TO HANDLE REQUESTS]
        
        CASE 1: User asks about **NEWS, MARKET DATA, or CURRENT EVENTS** (e.g., "Why is TSLA down?", "AAPL price?").
        -> **ACTION:** You MUST use the `Google Search` tool. Do not rely on internal knowledge for current prices.
        
        CASE 2: User asks to **REVIEW, ANALYZE, or SUMMARIZE** their own trading (e.g., "How is my win rate?", "Review my last trade").
        -> **ACTION:** Analyze the **Internal Trade History** provided above. Do NOT search the web.
        
        CASE 3: User is **LOGGING A TRADE** (e.g., "I bought AAPL", "Sold NVDA").
        -> **ACTION:** Simply acknowledge the trade enthusiastically and perhaps comment on how it fits their history (e.g., "Nice! Adding that to your journal."). Do NOT search the web.
        
        CASE 4: User asks for **PLANNING/STRATEGY** (e.g., "What should I look for in SPY?").
        -> **ACTION:** You may use `Google Search` to find current key levels or news events to watch, then combine that with general advice.

        **General Rule:** Be concise. If you use Search, mention that you checked live data.
        """

        # 2. Convert Chat History to new SDK format
        contents = []
        for msg in chat_history[-6:]:
            role = "model" if msg.role == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=msg.content)]))
        
        contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))
        
        fallback_message = "I apologize, but my market intelligence service is overloaded right now. Please wait a moment and try again."

        for attempt in range(MAX_RETRIES):
            try:
                # 3. Generate Content with Search Tool enabled
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        tools=[types.Tool(google_search=types.GoogleSearch())], 
                        temperature=0.7,
                    )
                )

                # 4. Check for Grounding (Did it use search?)
                is_grounded = False
                if response.candidates and response.candidates[0].grounding_metadata:
                    # Robust check for search entry point
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
