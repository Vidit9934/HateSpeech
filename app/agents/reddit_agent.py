import asyncpraw
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
import uuid
from app.agents import HybridRetrieverAgent
from app.models.policy_schema import (
    PolicyDocument,
    PolicyReasoning
)
from app.models.base_schema import (
    ClassificationLabel,
    Severity,
    ModerationAction,
    Platform
)
from app.models.moderation_schema import (
    ClassificationResult,
    ActionRecommendation,
    ModerationResponse
)

# from app.models.__init__ import (
#     # PolicyDocument, 
#     # ModerationResponse, 
#     # Platform, 
#     # ClassificationResult, 
#     # ClassificationLabel, 
#     # Severity, 
#     # ActionRecommendation, 
#     # ModerationAction, 
#     # PolicyReasoning
# )

logger = logging.getLogger("RedditAgent")

class RedditAgent:
    def __init__(self):
        try:
            # Initialize Reddit client
            self.reddit = asyncpraw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT")
            )
            # Initialize other agents
            self.retriever = HybridRetrieverAgent()
            
            # Action matrix definition
            self.action_matrix = {
                # Hate Speech Actions
                ("toxic", "high"): (ModerationAction.IMMEDIATE_REMOVAL, "Severe toxic content detected - immediate removal required"),
                ("toxic", "medium"): (ModerationAction.WARNING, "Toxic content - warning issued"),
                ("toxic", "low"): (ModerationAction.HUMAN_REVIEW, "Mild toxic content - review needed"),
                
                # Offensive Content Actions
                ("offensive", "high"): (ModerationAction.WARNING, "Highly offensive content - warning issued"),
                ("offensive", "medium"): (ModerationAction.HUMAN_REVIEW, "Potentially offensive - review needed"),
                ("offensive", "low"): (ModerationAction.NO_ACTION, "Mildly offensive - monitoring only"),
                
                # Default Actions
                ("neutral", "*"): (ModerationAction.NO_ACTION, "Content appears safe"),
                ("ambiguous", "*"): (ModerationAction.HUMAN_REVIEW, "Unable to determine - human review needed")
            }
            logger.info("Successfully initialized Reddit agent")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit agent: {str(e)}")
            raise

    async def analyze_comments(self, post_url: str, max_comments: int = 50) -> List[Dict[str, Any]]:
        """Analyze comments from a Reddit post"""
        try:
            submission = await self.reddit.submission(url=post_url)
            await submission.comments.replace_more(limit=0)
            
            comments = await submission.comments.list()
            results = []
            
            for comment in comments[:max_comments]:
                if not comment.body or comment.body in ["[deleted]", "[removed]"]:
                    continue
                
                # Get relevant policies
                relevant_policies = await self.retriever.retrieve_relevant_policies(
                    query=comment.body,
                    top_k=5
                )
                
                # Calculate confidence and severity based on policy relevance
                max_relevance = max([p.get("similarity_score", 0) for p in relevant_policies]) if relevant_policies else 0
                avg_relevance = sum(p.get("similarity_score", 0) for p in relevant_policies) / len(relevant_policies) if relevant_policies else 0
                
                # Determine severity based on max relevance score
                severity = Severity.HIGH if max_relevance > 0.7 else (
                    Severity.MEDIUM if max_relevance > 0.4 else Severity.LOW
                )
                
                # Set classification based on policy matches
                classification_label = ClassificationLabel.TOXIC if max_relevance > 0.7 else (
                    ClassificationLabel.OFFENSIVE if max_relevance > 0.4 else (
                    ClassificationLabel.AMBIGUOUS if max_relevance > 0.2 else ClassificationLabel.NEUTRAL
                ))
                
                # Format policies
                formatted_policies = [
                    PolicyDocument(
                        title=p.get("id", "Unknown Policy"),
                        content=p.get("content", ""),
                        source=p.get("source", "unknown"),
                        relevance_score=p.get("similarity_score", 0.0),
                        policy_type=p.get("policy_type", "general"),
                        section=p.get("metadata", {}).get("section")
                    ) for p in relevant_policies
                ]
                
                # Get action from matrix
                label = str(classification_label).lower()  # Changed from .value.lower()
                severity_level = str(severity).lower()     # Changed from .value.lower()
                
                action_tuple = self.action_matrix.get(
                    (label, severity_level),  # Try exact match
                    self.action_matrix.get(
                        (label, '*'),  # Try wildcard severity
                        (ModerationAction.HUMAN_REVIEW, "Unable to determine action - human review needed")  # Default
                    )
                )

                # Use the policy violations and severity assessment in reasoning
                policy_violations = [p.content for p in formatted_policies if p.relevance_score > 0.7]
                severity_assessment = (
                    "High severity content detected"
                    if severity == Severity.HIGH
                    else "Medium severity content detected"
                    if severity == Severity.MEDIUM
                    else "Low severity content detected"
                )

                moderation_result = ModerationResponse(
                    request_id=str(uuid.uuid4()),
                    original_text=comment.body,
                    platform=Platform.REDDIT,
                    classification=ClassificationResult(
                        label=classification_label,
                        confidence=max_relevance,
                        severity=severity,
                        reasoning=f"Policy match confidence: {max_relevance:.2%}",
                        detected_categories=[p["policy_type"] for p in relevant_policies if p.get("similarity_score", 0) > 0.3],
                        keywords_flagged=[]
                    ),
                    policy_analysis=PolicyReasoning(
                        applicable_policies=formatted_policies,
                        policy_violations=policy_violations,
                        reasoning_summary=f"Found {len(formatted_policies)} relevant policies with avg match of {avg_relevance:.2%}",
                        policies_analyzed=len(formatted_policies)
                    ),
                    recommended_action=ActionRecommendation(
                        primary_action=action_tuple[0],  # Changed from 'action'
                        justification=action_tuple[1],   # Changed from 'description'
                        alternative_actions=[],
                        escalation_needed=action_tuple[0] in [ModerationAction.HUMAN_REVIEW, ModerationAction.IMMEDIATE_REMOVAL, ModerationAction.CONTENT_REMOVAL,ModerationAction.NO_ACTION,ModerationAction.MANUAL_REVIEW],
                        follow_up_actions=[],
                        ban_duration=None
                    ),
                    processing_time_ms=0.0,
                    confidence_overall=max_relevance,
                    timestamp=datetime.utcnow()
                )
                
                results.append({
                    "comment_id": comment.id,
                    "author": str(comment.author),
                    "text": comment.body,
                    "moderation_result": moderation_result.model_dump(),
                    "created_utc": datetime.fromtimestamp(comment.created_utc)
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing comments: {str(e)}")
            raise