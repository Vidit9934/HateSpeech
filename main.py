# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

from .app.agents.hate_speech_detection import HateSpeechDetectionAgent
from .app.agents.hybrid_retriever import HybridRetrieverAgent
from .app.agents.policy_reasoning import PolicyReasoningAgent
from .app.agents.action_recommender import ActionRecommenderAgent
from app.models.schemas import ClassificationRequest, ModerationResponse

load_dotenv()

app = FastAPI(title="Hate Speech Detection API", version="1.0.0")

# CORS middleware for Streamlit integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
hate_speech_agent = HateSpeechDetectionAgent(os.getenv("OPENAI_API_KEY"))
retriever_agent = HybridRetrieverAgent(
    model_name=os.getenv("EMBEDDING_MODEL"),
    policy_docs_path=os.getenv("POLICY_DOCS_PATH"),
    vector_db_path=os.getenv("VECTOR_DB_PATH")
)
reasoning_agent = PolicyReasoningAgent(os.getenv("OPENAI_API_KEY"))
action_agent = ActionRecommenderAgent()

@app.on_event("startup")
async def startup_event():
    # Initialize vector database
    retriever_agent.load_and_index_documents()

@app.get("/")
async def root():
    return {"message": "Hate Speech Detection API is running"}

@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(request: ClassificationRequest):
    try:
        # Step 1: Classify content
        classification = await hate_speech_agent.classify_text(request.text)
        
        # Step 2: Retrieve relevant policies
        policies = await retriever_agent.retrieve_relevant_policies(request.text)
        
        # Step 3: Generate reasoning
        reasoning = await reasoning_agent.generate_reasoning(
            classification, policies, request.text
        )
        
        # Step 4: Recommend action
        action = action_agent.recommend_action(classification.label)
        
        return ModerationResponse(
            classification=classification,
            retrieved_policies=policies,
            reasoning=reasoning,
            recommended_action=action,
            success=True
        )
        
    except Exception as e:
        return ModerationResponse(
            classification=None,
            retrieved_policies=[],
            reasoning="",
            recommended_action="review",
            success=False,
            error_message=str(e)
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}