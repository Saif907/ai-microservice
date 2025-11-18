from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Any
from datetime import date

# --- Core Models (Mirrored from main backend for contract consistency) ---

class TradeCreate(BaseModel):
    """Data structure for a full trade record."""
    ticker: str
    entry_date: date
    entry_price: float
    quantity: float
    exit_date: Optional[date] = None
    exit_price: Optional[float] = None
    notes: Optional[str] = None

class Message(BaseModel):
    """Simplified message model for chat context."""
    role: Literal['user', 'assistant']
    content: str
    
# --- Request Models (Inputs from Main Backend) ---

class ChatProcessRequest(BaseModel):
    """Request to process a chat message with full history context."""
    user_message: str
    chat_history: List[Message] = Field(default_factory=list)
    trade_history: List[Dict[str, Any]] = Field(default_factory=list)

class TradeExtractionRequest(BaseModel):
    """Request to extract a trade from raw text."""
    text: str

class TradeAnalysisRequest(BaseModel):
    """Request to generate insights from trades."""
    trades: List[Dict[str, Any]]

class TitleGenerationRequest(BaseModel):
    """Request to generate a chat title from messages."""
    messages: List[Message]

# --- Response Models (Outputs to Main Backend) ---

class AIMessageResponse(BaseModel):
    """AI chat response (with optional extracted trade)."""
    message: str
    trade_extracted: Optional[TradeCreate] = None
    is_grounded: bool = False # ADDED: Flag to indicate if external tool/search was used

class InsightsResponse(BaseModel):
    """AI insights response."""
    summary: str
    insights: List[str]

class TitleResponse(BaseModel):
    """AI generated title response."""
    title: str