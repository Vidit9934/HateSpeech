import requests
from typing import Dict, Any
from .constants import API_BASE_URL

def analyze_content(text: str) -> Dict[str, Any]:
    """Call FastAPI backend for content analysis"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/text/moderate",  # Now will be http://localhost:8000/text/moderate
            json={"text": text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API server")
    except requests.exceptions.Timeout:
        raise TimeoutError("API request timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API Error: {str(e)}")

def analyze_reddit_comments(post_url: str, max_comments: int = 50) -> Dict[str, Any]:
    """Analyze comments from a Reddit post"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/reddit/comments",
            json={
                "post_url": post_url,
                "max_comments": max_comments
            },
            timeout=60  # Longer timeout for Reddit API
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to API server")
    except requests.exceptions.Timeout:
        raise TimeoutError("Reddit analysis timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Reddit API Error: {str(e)}")