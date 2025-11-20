import httpx
import os
import json
from typing import List, Dict, Any

# Define the base URL for a simple news fetching service (e.g., NewsAPI or a wrapper)
NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

if not NEWS_API_KEY:
    print("WARNING: NEWS_API_KEY is not set. Groq news tool will not function.")

async def fetch_stock_news(query: str, limit: int = 3) -> str:
    """
    Fetches the top news articles for a given stock ticker or market query.
    This function is designed to be called by the Groq model.
    """
    if not NEWS_API_KEY:
        return json.dumps({"error": "News API Key is missing. Cannot fetch real-time news."})

    # Restrict the query to specific financial sources for relevance
    # Example: Financial Times, Reuters, Bloomberg
    sources = "financial-times,reuters,bloomberg"
    
    params = {
        'q': query,
        'apiKey': NEWS_API_KEY,
        'pageSize': limit,
        'language': 'en',
        'sortBy': 'publishedAt',
        'sources': sources
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(NEWS_API_ENDPOINT, params=params)
            response.raise_for_status()

            data = response.json()
            
            if data['status'] != 'ok' or not data['articles']:
                return json.dumps({"error": f"No recent news found for query: {query}"})

            # Format the articles concisely for the LLM
            formatted_articles = []
            for article in data['articles']:
                formatted_articles.append({
                    "title": article.get('title'),
                    "source": article.get('source', {}).get('name'),
                    "description": article.get('description'),
                    "published_at": article.get('publishedAt')
                })
            
            return json.dumps({"articles": formatted_articles})

    except httpx.HTTPStatusError as e:
        print(f"HTTP Error fetching news: {e.response.status_code}")
        return json.dumps({"error": f"News API service error ({e.response.status_code})."})
    except Exception as e:
        print(f"General Error fetching news: {e}")
        return json.dumps({"error": "An internal error occurred while reaching the News API."})