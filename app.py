"""FastAPI application for video summarization."""
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
import tempfile
import asyncio
from datetime import datetime

from summarizer.core import main

app = FastAPI(
    title="Video Summarizer API",
    description="API for transcribing and summarizing videos from various sources",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class SummarizeRequest(BaseModel):
    """Request model for video summarization."""
    source: str = Field(..., description="Video source URL or path")
    base_url: str = Field(..., description="API endpoint URL")
    model: str = Field(..., description="Model to use")
    api_key: Optional[str] = Field(None, description="API key (or use .env)")
    type: str = Field("YouTube Video", description="Source type")
    force_download: bool = Field(False, description="Skip captions, use audio")
    output_dir: Optional[str] = Field("summaries", description="Save directory")
    no_save: bool = Field(False, description="Don't save to files")
    prompt_type: str = Field("Questions and answers", description="Summary style")
    language: str = Field("auto", description="Language code")
    chunk_size: int = Field(10000, description="Input text chunk size")
    parallel_calls: int = Field(30, description="Parallel API calls")
    max_tokens: int = Field(4096, description="Max output tokens for each chunk")
    transcription: str = Field("Cloud Whisper", description="Transcription method")
    whisper_model: str = Field("base", description="Whisper model size (tiny/base/small/medium/large)")
    whisper_device: Optional[str] = Field(None, description="Device for Whisper (cuda/cpu/None=auto)")
    verbose: bool = Field(False, description="Enable detailed progress output")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "https://www.youtube.com/watch?v=VIDEO_ID",
                "base_url": "http://10.40.52.25:30000/v1",
                "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
                "type": "YouTube Video",
                "transcription": "Local Whisper",
                "prompt_type": "Questions and answers",
                "chunk_size": 10000,
                "parallel_calls": 30,
                "max_tokens": 4096
            }
        }


class SummarizeFileRequest(BaseModel):
    """Request model for file upload summarization."""
    base_url: str = Field(..., description="API endpoint URL")
    model: str = Field(..., description="Model to use")
    api_key: Optional[str] = Field(None, description="API key (or use .env)")
    output_dir: Optional[str] = Field("summaries", description="Save directory")
    no_save: bool = Field(False, description="Don't save to files")
    prompt_type: str = Field("Questions and answers", description="Summary style")
    language: str = Field("auto", description="Language code")
    chunk_size: int = Field(10000, description="Input text chunk size")
    parallel_calls: int = Field(30, description="Parallel API calls")
    max_tokens: int = Field(4096, description="Max output tokens for each chunk")
    transcription: str = Field("Local Whisper", description="Transcription method")
    whisper_model: str = Field("base", description="Whisper model size (tiny/base/small/medium/large)")
    whisper_device: Optional[str] = Field(None, description="Device for Whisper (cuda/cpu/None=auto)")
    verbose: bool = Field(False, description="Enable detailed progress output")


# Response Models
class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    success: bool
    summary: Optional[str] = None
    file_path: Optional[str] = None
    error: Optional[str] = None
    source: Optional[str] = None
    timestamp: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


def build_config_from_request(request: SummarizeRequest, source_path: str) -> Dict[str, Any]:
    """Build configuration dictionary from request."""
    # Smart caption logic: If user explicitly specifies transcription method,
    # they want to use that method, so disable captions for YouTube videos
    smart_force_download = request.force_download or (
            request.transcription in ["Cloud Whisper", "Local Whisper"]
    )

    config = {
        "type_of_source": request.type,
        "use_youtube_captions": not smart_force_download,
        "transcription_method": request.transcription,
        "whisper_model": request.whisper_model,
        "whisper_device": request.whisper_device,
        "language": request.language,
        "prompt_type": request.prompt_type,
        "chunk_size": request.chunk_size,
        "parallel_api_calls": request.parallel_calls,
        "max_output_tokens": request.max_tokens,
        "base_url": request.base_url,
        "model": request.model,
        "verbose": request.verbose,
        "source_url_or_path": source_path
    }

    # If API key provided, use it
    if request.api_key:
        config["api_key"] = request.api_key

    return config


def save_summary_if_needed(
        summary: str,
        source: str,
        output_dir: str,
        no_save: bool
) -> Optional[str]:
    """Save summary to file if needed."""
    if no_save:
        return None

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_source = source.split("?")[0].split("/")[-1]
    filename = f"{clean_source}_{timestamp}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Summary for: {source}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(summary)

    return filepath


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_video(request: SummarizeRequest):
    """
    Summarize a video from a URL or file path.
    
    Supports:
    - YouTube videos (with automatic caption support)
    - Google Drive links
    - Dropbox links
    - Local file paths
    """
    try:
        config = build_config_from_request(request, request.source)

        # Run the main processing function
        # Note: main() uses asyncio internally, but we need to run it in executor
        # to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, main, config)

        # Save if needed
        file_path = None
        if not request.no_save:
            file_path = save_summary_if_needed(
                summary,
                request.source,
                request.output_dir or "summaries",
                request.no_save
            )

        return SummarizeResponse(
            success=True,
            summary=summary,
            file_path=file_path,
            source=request.source,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/file", response_model=SummarizeResponse)
async def summarize_uploaded_file(
        file: UploadFile = File(...),
        base_url: str = Form(...),
        model: str = Form(...),
        api_key: Optional[str] = Form(None),
        output_dir: Optional[str] = Form("summaries"),
        no_save: bool = Form(False),
        prompt_type: str = Form("Questions and answers"),
        language: str = Form("auto"),
        chunk_size: int = Form(10000),
        parallel_calls: int = Form(30),
        max_tokens: int = Form(4096),
        transcription: str = Form("Local Whisper"),
        whisper_model: str = Form("base"),
        whisper_device: Optional[str] = Form(None),
        verbose: bool = Form(False)
):
    """
    Summarize an uploaded video file.
    
    Upload a video file and get its summary. The file is temporarily saved
    and processed, then deleted after processing.
    """
    # Save uploaded file temporarily
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f"upload_{file.filename}")

    try:
        # Save uploaded file
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Build config
        config = {
            "type_of_source": "Local File",
            "use_youtube_captions": False,
            "transcription_method": transcription,
            "whisper_model": whisper_model,
            "whisper_device": whisper_device,
            "language": language,
            "prompt_type": prompt_type,
            "chunk_size": chunk_size,
            "parallel_api_calls": parallel_calls,
            "max_output_tokens": max_tokens,
            "base_url": base_url,
            "model": model,
            "verbose": verbose,
            "source_url_or_path": temp_file_path
        }

        if api_key:
            config["api_key"] = api_key

        # Process
        loop = asyncio.get_event_loop()
        summary = await loop.run_in_executor(None, main, config)

        # Save if needed
        file_path = None
        if not no_save:
            file_path = save_summary_if_needed(
                summary,
                file.filename,
                output_dir or "summaries",
                no_save
            )

        return SummarizeResponse(
            success=True,
            summary=summary,
            file_path=file_path,
            source=file.filename,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


@app.post("/summarize/batch", response_model=List[SummarizeResponse])
async def summarize_batch(requests: List[SummarizeRequest]):
    """
    Summarize multiple videos in batch.
    
    Process multiple video sources with the same or different configurations.
    """
    results = []

    for request in requests:
        try:
            config = build_config_from_request(request, request.source)

            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, main, config)

            file_path = None
            if not request.no_save:
                file_path = save_summary_if_needed(
                    summary,
                    request.source,
                    request.output_dir or "summaries",
                    request.no_save
                )

            results.append(SummarizeResponse(
                success=True,
                summary=summary,
                file_path=file_path,
                source=request.source,
                timestamp=datetime.now().isoformat()
            ))
        except Exception as e:
            results.append(SummarizeResponse(
                success=False,
                error=str(e),
                source=request.source,
                timestamp=datetime.now().isoformat()
            ))

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
