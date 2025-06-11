from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import os
from datetime import datetime
import uuid

from app.agents import (
    HateSpeechDetectionAgent,
    HybridRetrieverAgent,
    PolicyReasoningAgent,
    ActionRecommenderAgent,
    ErrorHandlerAgent,
    ErrorType
)
from app.models.schemas import (
    ClassificationLabel,
    Severity,
    Platform,
    ClassificationResult,
    PolicyReasoning,
    ActionRecommendation,
    ModerationRequest,
    ModerationResponse,
    PolicyDocument  # Add this import
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hate Speech Detection API",
    description="Toxicity detection system using multi-agent pipeline",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agents
agents = {}

@app.on_event("startup")
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
        
        logger.info("Successfully initialized all agents")
    except Exception as e:
        logger.error(f"Failed to initialize agents: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": {name: agent is not None for name, agent in agents.items()}
    }

@app.post("/moderate", response_model=ModerationResponse)
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
            classification_result=class_result.model_dump(),
            reasoning_result=reasoning
        )

        # 7. Create response
        response = ModerationResponse(
            request_id=str(uuid.uuid4()),
            original_text=request.text,
            platform=Platform.GENERAL,
            classification=class_result,
            policy_analysis=policy_analysis,  # Use the properly constructed object
            recommended_action=ActionRecommendation(**action),
            processing_time_ms=0.0,
            confidence_overall=class_result.confidence,
            timestamp=datetime.utcnow()
        )
        
        return response

    except Exception as e:
        logger.error(f"Moderation failed: {str(e)}", exc_info=True)
        error_response = agents["error"].handle_error(
            error=e,
            error_type=ErrorType.SYSTEM_ERROR,
            context={"text": request.text}
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": error_response["message"],
                "error_type": "moderation_failed",
                "error_detail": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)