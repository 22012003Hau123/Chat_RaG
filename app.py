"""
FastAPI Web Application for RAG Chatbot

This provides a web interface and REST API for the RAG system:
- REST API endpoint for asking questions
- Simple HTML/JavaScript chat interface
- CORS support for frontend development
- Health check endpoint

Usage:
    python app.py
    Then visit: http://localhost:8000
"""

import os
import tempfile
import subprocess
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request, UploadFile, File  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel

# Our RAG system
from src.rag_chain import RAGChain
from src.configuration import resolve_config_path


# Global variable to hold RAG chain instance
rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes RAG chain on startup and cleans up on shutdown.
    """
    global rag_chain
    
    # Startup: Initialize RAG chain
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    try:
        config_path = resolve_config_path()
        print(f"Using configuration: {config_path}")
        rag_chain = RAGChain(config_path=config_path)
        print("\nRAG system ready!")
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        print("Make sure:")
        print("1. Vector index exists (run src/ingest.py first)")
        print("2. OPENAI_API_KEY is set in environment")
        raise
    
    print("="*60 + "\n")
    
    yield
    
    # Shutdown: Cleanup if needed
    print("\nShutting down RAG system...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation chatbot for document Q&A",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str
    method: str = "similarity"
    history: Optional[list[Dict[str, str]]] = None


class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str
    sources: list[str]
    method_used: str


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - returns API information and available endpoints."""
    return {
        "name": "RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "ask": "/ask"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns 200 if system is healthy, 503 if not initialized.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    return {
        "status": "healthy",
        "rag_chain": "initialized",
        "vector_store": "loaded",
        "embedding_model": "ready",
        "llm": "configured"
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main question-answering endpoint using RAG.
    
    Retrieves relevant documents and generates answers using LLM.
    Returns answer with source references.
    """
    if rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check logs."
        )
    
    # Process history
    history_ctx = request.history or []
    print(f"\n📝 Received request: '{request.question}'")
    print(f"📚 History context: {len(history_ctx)} messages")
    
    try:
        result = rag_chain.query(
            question=request.question,
            method=request.method,
            history=history_ctx
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            method_used=request.method
        )
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.
    
    Accepts PDF, DOCX, PPTX files, processes them using ingest_single_file.py,
    and adds chunks to the vector database.
    """
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    # Validate file type
    allowed_extensions = ['.pdf', '.docx', '.pptx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Create temp file
    temp_file = None
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp:
            content = await file.read()
            temp.write(content)
            temp_file = temp.name
        
        print(f"\n{'='*60}")
        print(f"PROCESSING UPLOAD: {file.filename}")
        print(f"Temp file: {temp_file}")
        print(f"{'='*60}\n")
        
        # Run ingestion script
        result = subprocess.run(
            ['python', 'ingest_single_file.py', '--file', temp_file],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = result.stderr or "Processing failed"
            print(f"❌ Ingestion failed: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Document processing failed: {error_msg}")
        
        # Parse output to get chunk count
        output = result.stdout
        chunks_created = 0
        for line in output.split('\n'):
            if 'Created' in line and 'chunks' in line:
                try:
                    chunks_created = int(line.split('Created')[1].split('chunks')[0].strip())
                except:
                    pass
        
        print(f"✅ Document processed successfully: {chunks_created} chunks created\n")
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": chunks_created,
            "message": "Document uploaded and processed successfully"
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Processing timeout - file too large")
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_file}: {e}")


@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    """Serve the HTML chat interface."""
    return templates.TemplateResponse("chat.html", {"request": request})


# We'll add server startup in the next step


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
