from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime
from .moderation_schema import ModerationResponse

class SubredditRequest(BaseModel):
    """Request to analyze a subreddit's content"""
    subreddit_name: str
    max_posts: int = 10

class PostInfo(BaseModel):
    """Basic information about a Reddit post"""
    post_id: str
    title: str
    url: str
    author: str
    created_utc: datetime
    score: int
    num_comments: int

class SubredditInfo(BaseModel):
    """Information about a subreddit including its policies"""
    name: str
    description: str
    subscribers: int
    posts: List[PostInfo]
    policies: List[str]

class CommentAnalysisRequest(BaseModel):
    """Request to analyze comments on a specific post"""
    post_url: HttpUrl
    max_comments: int = 50

class CommentAnalysis(BaseModel):
    """Analysis result for a single comment"""
    comment_id: str
    author: str
    text: str
    moderation_result: ModerationResponse
    created_utc: datetime