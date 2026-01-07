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
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from pydantic import BaseModel

# Our RAG system
from src.rag_chain import RAGChain
from src.configuration import resolve_config_path
from src.session_manager import SessionManager
from src.mistral_image_search import get_mistral_image_search


# Global variables to hold RAG chain and session manager
rag_chain: Optional[RAGChain] = None
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Initializes RAG chain on startup and cleans up on shutdown.
    """
    global rag_chain, session_manager
    
    # Startup: Initialize RAG chain
    print("\n" + "="*60)
    print("INITIALIZING RAG SYSTEM")
    print("="*60)
    
    try:
        config_path = resolve_config_path()
        print(f"Using configuration: {config_path}")
        rag_chain = RAGChain(config_path=config_path)
        
        # Initialize SessionManager
        print("\nInitializing Session Manager...")
        session_manager = SessionManager(
            cleanup_interval_minutes=10
        )
        
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
    session_id: Optional[str] = None  # Frontend-generated UUID


class AnswerResponse(BaseModel):
    """Response model for answers."""
    answer: str
    sources: list[str]
    method_used: str
    session_id: str  # Return session_id to frontend


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


# Helper function for file type detection
def _detect_file_type(filename: str) -> str:
    """
    Determine file type from filename extension.
    
    Returns: 'image', 'pdf', or 'unknown'
    """
    import os
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
        return 'image'
    elif ext == '.pdf':
        return 'pdf'
    else:
        return 'unknown'


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    question: str = Form(...),
    image: Optional[UploadFile] = File(None),
    method: str = Form("mmr"),
    session_id: Optional[str] = Form(None)
):
    """
    Main question-answering endpoint using RAG with optional image analysis.
    
    Now supports:
    - Text-only questions (original functionality)
    - Image + text questions (vision analysis + RAG)
    - Session-based conversation memory
    
    Returns answer with source references and session_id.
    """
    if rag_chain is None or session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please check logs."
        )
    
    # Generate session ID if not provided
    import uuid
    session_id_str = session_id or str(uuid.uuid4())
    
    print(f"\n📝 Received request: '{question}'")
    print(f"🆔 Session ID: {session_id_str[:8]}...")
    
    # Handle file if provided (image or document)
    file_context = ""
    temp_file_path = None
    
    if image:  # 'image' parameter name kept for backward compatibility
        file_type = _detect_file_type(image.filename)
        print(f"📎 File attached: {image.filename} (type: {file_type})")
        
        # MISTRAL IMAGE SEARCH: Detect if this is pure "find this image" query
        pure_search_keywords = [
            # Vietnamese
            'tìm', 'tim', 'chỗ nào',
            # French
            'où', 'trouve', 'trouver', 'fichier', 'quel', 'dans',
            # English
            'find', 'search', 'locate', 'which', 'where'
        ]
        analysis_keywords = ['phân tích', 'mô tả', 'mo ta', 'analyze', 'describe', 'what', 'how', 'explain']
        
        is_pure_image_search = (
            file_type == 'image' and
            any(kw in question.lower() for kw in pure_search_keywords) and
            not any(kw in question.lower() for kw in analysis_keywords)
        )
        
        if is_pure_image_search:
            print("🔍 MISTRAL IMAGE SEARCH: Semantic annotation matching")
            
            try:
                from src.vision_analyzer import save_uploaded_image, cleanup_temp_image
                import os
                
                temp_file_path = save_uploaded_image(image)
                
                # Get Mistral API key
                mistral_key = os.getenv('MISTRAL_API_KEY')
                if not mistral_key:
                    print("⚠️ MISTRAL_API_KEY not found, falling back to normal search")
                else:
                    # Mistral annotation search
                    searcher = get_mistral_image_search(mistral_key)
                    match = searcher.search_by_annotation(temp_file_path, rag_chain)
                    
                    cleanup_temp_image(temp_file_path)
                    
                    if match:
                        return AnswerResponse(
                            answer=f"""✅ Image trouvée!

**Document:** {match['doc_name']}
**Image:** [Voir l'image]({match['image_url']})

**Annotation:** {match['annotation']}""",
                            sources=match['full_result'].get('sources', []),
                            method_used="mistral_annotation_search",
                            session_id=session_id_str
                        )
                    else:
                        return AnswerResponse(
                            answer="❌ Aucune image correspondante trouvée.",
                            sources=[],
                            method_used="mistral_annotation_search",
                            session_id=session_id_str
                        )
            except Exception as e:
                print(f"⚠️ Mistral search error: {e}")
                import traceback
                traceback.print_exc()
                if temp_file_path:
                    cleanup_temp_image(temp_file_path)
            
            try:
                from src.vision_analyzer import save_uploaded_image, cleanup_temp_image
                temp_file_path = save_uploaded_image(image)
                
                # Get all documents for comparison
                all_docs = rag_chain.retriever.retrieve(query="", k=100, method="similarity")
                
                # Direct image search
                searcher = get_direct_image_search(similarity_threshold=0.80)
                match = searcher.search_all_images(temp_file_path, all_docs)
                
                cleanup_temp_image(temp_file_path)
                
                if match:
                    return AnswerResponse(
                        answer=f"""✅ Image trouvée! (Similarité: {match['similarity']:.1%})

**Fichier:** [{match['file_name']}]({match['file_url']})
**Image:** [Voir l'image]({match['image_url']})""",
                        sources=[f"[{match['file_name']}]({match['file_url']})"],
                        method_used="direct_image_search",
                        session_id=session_id_str
                    )
                else:
                    return AnswerResponse(
                        answer="❌ Aucune image correspondante trouvée dans la base de données.",
                        sources=[],
                        method_used="direct_image_search",
                        session_id=session_id_str
                    )
            except Exception as e:
                print(f"⚠️ Direct search error: {e}")
                if temp_file_path:
                    cleanup_temp_image(temp_file_path)
                # Fall through to normal flow
        
        # NORMAL FLOW: Vision analysis for content questions
        try:
            if file_type == 'image':
                # Image analysis with vision model
                from src.vision_analyzer import VisionAnalyzer, save_uploaded_image, cleanup_temp_image
                
                temp_file_path = save_uploaded_image(image)
                
                vision_config = rag_chain.config.get('llm', {})
                vision_model = vision_config.get('vision_model', 'gpt-4o-mini')
                
                analyzer = VisionAnalyzer(model=vision_model)
                image_analysis = analyzer.analyze_image(temp_file_path, question)
                
                file_context = f"\n\n[Image Context]\n{image_analysis}\n"
                print(f"✓ Vision analysis complete")
                
            elif file_type == 'pdf':
                # PDF → Convert to images → Vision analysis
                from src.vision_analyzer import VisionAnalyzer, save_uploaded_image, cleanup_temp_image
                from pdf2image import convert_from_path
                import os
                
                # Save PDF temporarily
                temp_file_path = save_uploaded_image(image)
                
                print(f"📄 Converting PDF to images for vision analysis...")
                
                # Convert PDF pages to images
                try:
                    # Convert only first 5 pages to avoid excessive cost
                    images = convert_from_path(
                        temp_file_path, 
                        dpi=150,  # Good quality for vision
                        first_page=1,
                        last_page=5  # Limit to first 5 pages
                    )
                    
                    print(f"✓ Converted {len(images)} pages to images")
                    
                    # Initialize vision analyzer
                    vision_config = rag_chain.config.get('llm', {})
                    vision_model = vision_config.get('vision_model', 'gpt-4o-mini')
                    analyzer = VisionAnalyzer(model=vision_model)
                    
                    # Analyze each page
                    page_analyses = []
                    for i, img in enumerate(images, 1):
                        # Save image temporarily
                        temp_img_path = f"/tmp/pdf_page_{i}_{os.urandom(8).hex()}.png"
                        img.save(temp_img_path, 'PNG')
                        
                        # Vision analysis
                        analysis = analyzer.analyze_image(temp_img_path, question)
                        page_analyses.append(f"--- Page {i} ---\n{analysis}")
                        
                        # Clean up temp image
                        try:
                            os.remove(temp_img_path)
                        except:
                            pass
                    
                    # Combine all page analyses
                    combined_analysis = "\n\n".join(page_analyses)
                    file_context = f"\n\n[PDF Analysis - {len(images)} pages]\n{combined_analysis}\n"
                    print(f"✓ PDF vision analysis complete")
                    
                except Exception as e:
                    print(f"⚠️  PDF conversion error: {e}")
                    file_context = f"\n\n[Note: Could not analyze PDF: {str(e)}]"
                
            else:
                file_context = f"\n\n[Note: Unsupported file type: {file_type}]"
                print(f"⚠️  Unsupported file type: {file_type}")
            
        except Exception as e:
            print(f"❌ Error processing file: {e}")
            import traceback
            traceback.print_exc()
            # Continue without file analysis rather than failing completely
            file_context = f"\n\n[Note: File was provided but could not be processed: {str(e)}]"
    
    # Get or create session
    session = session_manager.get_or_create_session(session_id_str)
    
    try:
        # Format question with file context
        # Important: Keep original question clear so conversation context works
        if file_context:
            # Add file context as additional information, not replacing question
            enhanced_question = f"""{question}

[INFORMATION ADDITIONNELLE DU FICHIER JOINT]
{file_context.strip()}
[FIN DU FICHIER]

Note: Utilisez l'historique de conversation pour comprendre le contexte de cette question."""
        else:
            enhanced_question = question
        
        result = rag_chain.query(
            question=enhanced_question,
            method=method,
            session=session
        )
        
        return AnswerResponse(
            answer=result["answer"],
            sources=result["sources"],
            method_used=method,
            session_id=session_id_str
        )
        
    except Exception as e:
        print(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_file_path:
            from src.vision_analyzer import cleanup_temp_image
            cleanup_temp_image(temp_file_path)


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
        # Pass the original filename via --storage-path to ensure correct metadata and source name
        cmd = ['python', 'ingest_single_file.py', '--file', temp_file, '--storage-path', file.filename]
        
        result = subprocess.run(
            cmd,
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
            # Match output from batch_process_all.py: "✓ Inserted 5 documents to Supabase"
            if 'Inserted' in line and 'documents' in line:
                try:
                    # Extract number between 'Inserted' and 'documents'
                    # Line format: "✓ Total inserted: 3 documents"
                    parts = line.split('Inserted')[1].split('documents')[0]
                    # Remove non-digit characters (like ':')
                    num_str = ''.join(filter(str.isdigit, parts))
                    if num_str:
                        chunks_created = int(num_str)
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
