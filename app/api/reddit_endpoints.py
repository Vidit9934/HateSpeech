from fastapi import APIRouter, HTTPException
from typing import List
from app.models.reddit_schema import CommentAnalysisRequest, CommentAnalysis
from app.agents.reddit_agent import RedditAgent

router = APIRouter()
reddit_agent = RedditAgent()

@router.post("/comments", response_model=List[CommentAnalysis])
async def analyze_comments(request: CommentAnalysisRequest):
    """Analyze comments on a specific Reddit post"""
    try:
        return await reddit_agent.analyze_comments(
            str(request.post_url),
            request.max_comments
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))