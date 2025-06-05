# api/app.py
from fastapi import FastAPI
import logging
import uvicorn

# Import routes
from .routes import router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Inferloop NLP Synthetic Data API", version="0.1.0")

# Include routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    try:
        # Initialization is now handled in routes.py
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Inferloop NLP Synthetic Data API",
        "version": "0.1.0",
        "endpoints": ["/generate", "/validate", "/format", "/health"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
