import json
import os
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel

# --- Official xAI SDK Imports ---
from xai_sdk import Client
from xai_sdk.chat import system, user, assistant, tool_result

# --- Local Imports ---
from app.models import TradeCreate, Message
# Ensure you have created this file from the previous step!
from app.news_api_tool import fetch_stock_news 

class AIService:
    """
    Drop-in replacement for AIService using xAI's Grok.
    Implements Sequential Classification and Tool Use via xai-sdk.
    """

    def __init__(self):
        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY is missing. Cannot initialize Grok client.")
        
        # Initialize the official xAI Client
        self.client = Client(api_key=api_key)
        
        # 'grok-beta' is currently the standard API model
        self.MODEL_ID = "grok-beta" 

    async def classify_intent(self, text: str) -> str:
        """
        STAGE 1: Fast classification of user intent.
        """
        try:
            # Create a standalone chat session for classification
            chat = self.client.chat.create(model=self.MODEL_ID)
            
            chat.append(system(
                "You are a classifier. Determine the PRIMARY intent of the input.\n"
                "Categories: LOG_TRADE, REVIEW_ANALYSIS, NEWS_MARKET, PLAN_STRATEGY, OTHER.\n"
                "Return ONLY a JSON object: {\"intent\": \"CATEGORY\"}"
            ))
            chat.append(user(text))
            
            # Low temperature for deterministic JSON
            response = chat.sample(temperature=0.0)
            
            # Clean potential markdown wrappers (```json ... ```) if Grok adds them
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()

            data = json.loads(content)
            return data.get("intent", "OTHER").upper()
        except Exception as e:
            print(f"⚠️ Grok Classification Error: {e}")
            return "OTHER"

    async def extract_trade_from_text(self, text: str) -> Optional[TradeCreate]:
        """
        STAGE 2: Extract trade data only if intent is LOG_TRADE.
        """
        
        # 1. Check Intent
        intent = await self.classify_intent(text)
        if intent != "LOG_TRADE":
            return None

        # 2. Perform Extraction
        today = datetime.now().strftime('%Y-%m-%d')
        
        chat = self.client.chat.create(model=self.MODEL_ID)
        chat.append(system(
            "You are a strict Data Extraction Agent. Your job is to extract trade details into JSON.\n"
            "Schema: ticker (str), entry_price (float), quantity (float), action (buy/sell), etc.\n"
            "Rules:\n"
            "- Return valid JSON matching the schema.\n"
            "- Default quantity to 1 if missing.\n"
            "- Return the string 'null' if no valid trade is found."
        ))
        
        user_prompt = f"""
        Context Date: {today}
        User Input: "{text}"
        
        Extract the trade details JSON:
        """
        chat.append(user(user_prompt))

        try:
            response = chat.sample(temperature=0.0)
            content = response.content.strip()
            
            # Clean markdown if present
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()

            if "null" not in content.lower():
                return TradeCreate.model_validate_json(content)
            return None
            
        except Exception as e:
            print(f"❌ Grok Extraction Error: {e}")
            return None

    async def generate_chat_response(
        self, 
        user_message: str, 
        chat_history: List[Message], 
        trade_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates chat response using Grok.
        Handles Tool Calling for 'fetch_stock_news'.
        """
        
        trade_context = json.dumps(trade_history[-20:], indent=2, default=str) if trade_history else "No history."

        system_instruction = f"""
        You are an expert Trading AI (Grok).
        
        [DATA SOURCES]
        1. Internal History: {trade_context}
        2. News Tool: Use `fetch_stock_news` for live market info.

        [PROTOCOL]
        - NEWS/MARKET: You MUST call the tool `fetch_stock_news`.
        - ANALYSIS: Use Internal History.
        - LOGGING: Be concise and confirming.
        """

        # 1. Define Tools (xAI SDK format)
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
            # 2. Initialize Chat & Build History
            chat = self.client.chat.create(model=self.MODEL_ID)
            chat.append(system(system_instruction))
            
            for m in chat_history[-6:]:
                if m.role == "user":
                    chat.append(user(m.content))
                else:
                    chat.append(assistant(m.content))
            
            chat.append(user(user_message))

            # 3. First Sample (Model decides to use tool or not)
            # Note: In xAI SDK, tools are passed here
            response = chat.sample(tools=tools, temperature=0.7)
            
            is_grounded = False

            # 4. Handle Tool Calls
            if response.tool_calls:
                is_grounded = True
                # Add Grok's request-to-call-tool message to history
                chat.append(response) 

                for call in response.tool_calls:
                    if call.function.name == "fetch_stock_news":
                        try:
                            # Parse arguments
                            args = json.loads(call.function.arguments)
                            # Execute Python function
                            news_result = await fetch_stock_news(args["query"])
                            
                            # Add result back to chat
                            chat.append(tool_result(
                                tool_call_id=call.id,
                                content=news_result
                            ))
                        except Exception as tool_err:
                             chat.append(tool_result(
                                tool_call_id=call.id,
                                content=json.dumps({"error": str(tool_err)})
                            ))

                # 5. Final Sample (Grok answers using the tool data)
                final_response = chat.sample(tools=tools, temperature=0.7)
                return {"message": final_response.content, "is_grounded": True}

            # Case: No tool used
            return {"message": response.content, "is_grounded": False}

        except Exception as e:
            print(f"❌ Grok Chat Error: {e}")
            # Fallback if Grok API fails
            return {"message": "Grok service is currently unavailable.", "is_grounded": False}

    async def analyze_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generates analysis insights."""
        if not trades:
            return {"summary": "No trades to analyze.", "insights": []}
            
        prompt = f"""Analyze this trade data and return a JSON object with a 'summary' and a list of 'insights'.
        Trades: {json.dumps(trades[:50], default=str)}
        """
        
        try:
            chat = self.client.chat.create(model=self.MODEL_ID)
            chat.append(system("Return ONLY valid JSON: { 'summary': str, 'insights': [str] }"))
            chat.append(user(prompt))
            
            response = chat.sample(temperature=0.1)
            
            # Clean potential markdown
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()

            return json.loads(content)
        except Exception as e:
            print(f"Analysis Error: {e}")
            return {"summary": "Analysis failed.", "insights": []}

    async def generate_title_for_chat(self, messages: List[Message]) -> Optional[str]:
        """Generates a short title."""
        prompt = "Generate a very short (3-5 words) title for this conversation. Return JSON: {'title': '...'}"
        context = "\n".join([f"{m.role}: {m.content}" for m in messages[-5:]])
        
        try:
            chat = self.client.chat.create(model=self.MODEL_ID)
            chat.append(user(f"{prompt}\n\nConversation:\n{context}"))
            
            response = chat.sample(temperature=0.3)
            
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].replace("json", "").strip()
                
            data = json.loads(content)
            return data.get("title")
        except Exception:
            return "New Chat"