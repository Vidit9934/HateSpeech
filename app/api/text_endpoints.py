from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import uuid
import os

from app.agents import (
    HateSpeechDetectionAgent,
    HybridRetrieverAgent,
    PolicyReasoningAgent,
    ActionRecommenderAgent,
    ErrorHandlerAgent,
)

from app.models.policy_schema import (
    PolicyDocument,
    PolicyReasoning
)
from app.models.base_schema import (
    ClassificationLabel,
    Severity,
    Platform
)
from app.models.moderation_schema import (
    ClassificationResult,
    ActionRecommendation,
    ModerationResponse,
    ModerationRequest
)
# from app.models.__init__ import (
#     ClassificationLabel,
#     Severity,
#     Platform,
#     ClassificationResult,
#     PolicyReasoning,
#     ActionRecommendation,
#     ModerationRequest,
#     ModerationResponse,
#     PolicyDocument
# )

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global agents
agents = {}

@router.on_event("startup")
async def startup_event():
    """Initialize all agents and vector store on startup"""
    try:
        # Initialize agents
        agents["hate_speech"] = HateSpeechDetectionAgent()
        agents["retriever"] = HybridRetrieverAgent()
        agents["reasoning"] = PolicyReasoningAgent()
        agents["action"] = ActionRecommenderAgent()
        agents["error"] = ErrorHandlerAgent()

        # Load policy documents
        policy_path = "data/policy_docs/"
        if os.path.exists(policy_path):
            await agents["retriever"].index_policy_documents(policy_path)
        
        logger.info("Successfully initialized text analysis agents")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        raise

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {name: agent is not None for name, agent in agents.items()}
    }

@router.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ModerationRequest):
    """Main moderation endpoint"""
    try:
        # 1. Get classification
        classification = await agents["hate_speech"].classify_text(request.text)
        
        # Convert confidence string to float
        confidence_str = classification.get("confidence", "0.0").lower()
        confidence_map = {
            "high": 0.9,
            "medium": 0.6,
            "low": 0.3
        }
        confidence = confidence_map.get(confidence_str, 0.0)
        
        # 2. Create ClassificationResult object
        class_result = ClassificationResult(
            label=ClassificationLabel(classification.get("classification", "AMBIGUOUS").lower()),
            confidence=confidence,
            severity=Severity.MEDIUM,
            reasoning=classification.get("reason", "No reason provided"),
            detected_categories=[],
            keywords_flagged=[]
        )

        # 3. Get policies
        policies = await agents["retriever"].retrieve_relevant_policies(
            query=request.text, 
            top_k=5
        )

        # Format policies to match PolicyDocument schema
        formatted_policies = []
        for p in policies:
            policy_doc = PolicyDocument(
                title=p.get("id", "Unknown Policy"),  # Use id as title if no title
                content=p.get("content", ""),
                source=p.get("source", "unknown"),
                relevance_score=p.get("similarity_score", 0.0),  # Map similarity_score to relevance_score
                policy_type=p.get("policy_type", "general"),
                section=p.get("metadata", {}).get("section")
            )
            formatted_policies.append(policy_doc)

        # 4. Generate reasoning from PolicyReasoningAgent
        reasoning = await agents["reasoning"].generate_policy_reasoning(
            original_text=request.text,
            classification_result={
                "classification": str(class_result.label),
                "confidence": str(class_result.confidence),
                "severity": str(class_result.severity),
                "reason": class_result.reasoning
            },
            relevant_policies=[p.model_dump() for p in formatted_policies]  # Convert to dict for reasoning
        )

        # 5. Create PolicyReasoning object with properly formatted policies
        policy_analysis = PolicyReasoning(
            applicable_policies=formatted_policies,  # Use formatted policies directly
            policy_violations=reasoning.get("policy_violations", []),
            reasoning_summary=reasoning.get("detailed_reasoning", "No detailed reasoning available"),
            detailed_reasoning=reasoning.get("detailed_reasoning"),
            supporting_evidence=reasoning.get("supporting_evidence"),
            severity_assessment=reasoning.get("severity_assessment"),
            additional_context=reasoning.get("additional_context"),
            original_classification={
                "classification": str(class_result.label),
                "confidence": str(class_result.confidence),
                "reason": class_result.reasoning
            },
            policies_analyzed=len(formatted_policies)
        )

        # 6. Get action recommendation
        action = await agents["action"].recommend_action(
            classification_result={
                "label": class_result.label.value.lower(),
                "severity": class_result.severity.value.lower(),
                "confidence": class_result.confidence
            },
            reasoning_result={
                # Fix: Access dictionary items correctly
                "policy_violations": reasoning.get("policy_violations", []),
                "severity_assessment": reasoning.get("severity_assessment", "")
            }
        )

        # Create response
        response = ModerationResponse(
            request_id=str(uuid.uuid4()),
            original_text=request.text,
            platform=Platform.GENERAL,
            classification=class_result,
            policy_analysis=policy_analysis,
            recommended_action=ActionRecommendation(**action),
            processing_time_ms=0.0,
            confidence_overall=class_result.confidence,
            timestamp=datetime.utcnow()
        )
        
        return response

    except Exception as e:
        logger.error(f"Moderation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
