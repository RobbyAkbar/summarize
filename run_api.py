#!/usr/bin/env python3
"""Script to run the FastAPI server."""
import uvicorn
import os

if __name__ == "__main__":
    # Disable reload in production for better model caching
    # Set FASTAPI_DEV=1 environment variable to enable reload during development
    reload_mode = os.getenv("FASTAPI_DEV", "0") == "1"

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=reload_mode,
        workers=1 if not reload_mode else None  # Use single worker for model caching
    )
