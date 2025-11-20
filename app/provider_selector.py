  
# ai-microservice/app/provider_selector.py

import os
from dotenv import load_dotenv
load_dotenv()
# Since main.py loads .env early, we can read the variable here
# Use a default of 'gemini' if not specified
AI_PROVIDER = os.environ.get("AI_PROVIDER", "gemini").lower() 

# --- IMPORTANT: We need to import the actual implementation classes ---

# Note: The original ai_service.py is assumed to be renamed to gemini_service.py 
# in a complete modular structure, but for simplicity, we will create aliases.

if AI_PROVIDER == "grok":
    from .grok_service import GrokService as AIService
    print("🤖 AI Provider Selected: Grok (xAI)")
    
elif AI_PROVIDER == "gemini":
    # The original file, which you currently have as 'ai_service.py', 
    # needs to be renamed or treated as the Gemini implementation.
    # For this demonstration, we assume you have renamed your current ai_service.py to gemini_service.py.
    # If not, you must rename your current ai_service.py to gemini_service.py manually.
    from .gemini_service import AIService as AIService 
    print("🤖 AI Provider Selected: Gemini")

elif AI_PROVIDER == "groq":
    from .groq_service import GroqService as AIService
    print("🤖 AI Provider Selected: Groq (Llama 3 Models)")
    
else:
    # Fallback to a class that raises an error if the config is bad
    class PlaceholderService:
        def __init__(self):
            raise ValueError(f"Invalid AI_PROVIDER '{AI_PROVIDER}' set in environment.")
        # Define necessary async methods to prevent runtime errors
        async def extract_trade_from_text(self, *args): return None
        async def generate_chat_response(self, *args): return {"message": "Service Not Configured.", "is_grounded": False}
        async def analyze_trades(self, *args): return {"summary": "Service Not Configured.", "insights": []}
        async def generate_title_for_chat(self, *args): return "Error"
        
    AIService = PlaceholderService
    print(f"❌ AI Provider Error: '{AI_PROVIDER}' is not a valid provider. Check AI_PROVIDER variable.")