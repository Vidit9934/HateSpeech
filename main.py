from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.text_endpoints import router as text_router
from app.api.reddit_endpoints import router as reddit_router

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

# Mount endpoints
app.include_router(text_router, prefix="/text", tags=["Text Analysis"])
app.include_router(reddit_router, prefix="/reddit", tags=["Reddit Analysis"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)