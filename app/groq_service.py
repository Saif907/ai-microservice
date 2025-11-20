import json
import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

# --- Official Groq SDK ---
from groq import AsyncGroq

# --- Local Imports ---
from app.models import TradeCreate, Message
from app.news_api_tool import fetch_stock_news

class GroqService:
    """
    AI Service utilizing Groq LPU (Llama 3 Models).
    """

    def __init__(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing. Cannot initialize Groq client.")
        
        self.client = AsyncGroq(api_key=api_key)
        self.FAST_MODEL = "llama-3.1-8b-instant"
        self.SMART_MODEL = "llama-3.3-70b-versatile"

    async def classify_intent(self, text: str) -> str:
        """STAGE 1: Fast classification."""
        prompt = f"""
        You are a classifier. Determine the PRIMARY intent of the input.
        Categories: LOG_TRADE, REVIEW_ANALYSIS, NEWS_MARKET, PLAN_STRATEGY, OTHER.
        Return ONLY a JSON object: {{"intent": "CATEGORY"}}
        Input: "{text}"
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=50
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            return data.get("intent", "OTHER").upper()
        except Exception as e:
            print(f"Groq Classification Warning: {e}")
            return "OTHER"

    async def extract_trade_from_text(self, text: str) -> Optional[TradeCreate]:
        """STAGE 2: Extract trade data."""
        intent = await self.classify_intent(text)
        if intent != "LOG_TRADE":
            return None

        today = datetime.now().strftime('%Y-%m-%d')
        system_msg = f"""
        You are a strict Data Extraction Agent. Context Date: {today}.
        Extract the COMPLETED trade details into JSON.
        Schema matches TradeCreate (ticker, entry_price, quantity, etc).
        Rules:
        - Return VALID JSON only.
        - Default quantity to 1 if missing.
        - If no valid trade is found, return JSON: {{"error": "null"}}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.SMART_MODEL,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            content = response.choices[0].message.content
            if "null" in content.lower() and "error" in content.lower():
                return None
            return TradeCreate.model_validate_json(content)
        except Exception as e:
            print(f"Groq Extraction Error: {e}")
            return None

    async def generate_chat_response(
        self, 
        user_message: str, 
        chat_history: List[Message], 
        trade_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates chat response using Groq.
        """
        
        trade_context = json.dumps(trade_history[-15:], indent=2, default=str) if trade_history else "No history."
        
        # --- FIX: Added GENERAL/OTHER protocol to prevent "Copy that" loop ---
        system_instruction = f"""
        You are an expert Trading AI running on Groq.
        
        [DATA SOURCES]
        1. History: {trade_context}
        2. Tools: `fetch_stock_news` for live data.

        [PROTOCOL]
        - NEWS/MARKET: You MUST call `fetch_stock_news`.
        - ANALYSIS: Use History.
        - LOGGING: Provide a neutral, short acknowledgment (e.g., "Copy that"). DO NOT confirm details.
        - GENERAL/OTHER: Respond naturally, helpfully, and conversationally to greetings or questions.
        """

        messages = [{"role": "system", "content": system_instruction}]
        for m in chat_history[-6:]:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_message})

        tools = [{
            "type": "function",
            "function": {
                "name": "fetch_stock_news",
                "description": "Get live news for a stock ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Ticker symbol (e.g. AAPL)"}
                    },
                    "required": ["query"]
                }
            }
        }]

        try:
            response = await self.client.chat.completions.create(
                model=self.SMART_MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.6
            )
            
            msg = response.choices[0].message
            is_grounded = False

            if msg.tool_calls:
                is_grounded = True
                messages.append(msg)
                for tool_call in msg.tool_calls:
                    if tool_call.function.name == "fetch_stock_news":
                        try:
                            args = json.loads(tool_call.function.arguments)
                            news_result = await fetch_stock_news(args["query"])
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": news_result
                            })
                        except Exception as tool_err:
                             messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": str(tool_err)})
                            })

                final_response = await self.client.chat.completions.create(
                    model=self.SMART_MODEL,
                    messages=messages
                )
                return {"message": final_response.choices[0].message.content, "is_grounded": True}

            return {"message": msg.content, "is_grounded": False}

        except Exception as e:
            print(f"Groq Chat Error: {e}")
            return {
                "message": "Groq service is currently unavailable. Please try again.", 
                "is_grounded": False
            }

    async def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates analysis insights."""
        if not trades:
            return {"summary": "No trades.", "insights": []}
        
        try:
            response = await self.client.chat.completions.create(
                model=self.SMART_MODEL,
                messages=[
                    {"role": "system", "content": "Return JSON: { 'summary': str, 'insights': [str] }"},
                    {"role": "user", "content": f"Analyze: {json.dumps(trades[:50], default=str)}"}
                ],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Analysis Error: {e}")
            return {"summary": "Analysis failed.", "insights": []}

    async def generate_title_for_chat(self, messages: List[Message]) -> Optional[str]:
        """Generates a short title."""
        try:
            context = "\n".join([f"{m.role}: {m.content}" for m in messages[-5:]])
            response = await self.client.chat.completions.create(
                model=self.FAST_MODEL,
                messages=[
                    {"role": "system", "content": "Return JSON: {'title': '3-5 word title'}"},
                    {"role": "user", "content": context}
                ],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            return data.get("title", "New Chat")
        except Exception:
            return "New Chat"