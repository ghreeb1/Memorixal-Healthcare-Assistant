"""
FastAPI Router for Memory Storytelling Feature
Converted from Flask routes to FastAPI async routes
"""
import os
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from typing import List, Optional

from .core import db as models
from .core import db as crud
from .core.db import engine, get_db
from .services import ai_services

# Create database tables
models.Base.metadata.create_all(bind=engine)

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Create router
storytelling_router = APIRouter()

# Static directory for serving images and audio
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(static_dir, "audio"), exist_ok=True)

# Root endpoint -> serve the main home page with direct access to features
@storytelling_router.get("/", response_class=HTMLResponse, name="storytelling-home")
async def storytelling_home(request: Request):
    """Serve the storytelling home page with direct access to main features"""
    return templates.TemplateResponse("storytelling/home.html", {"request": request})

# API info endpoint
@storytelling_router.get("/api/")
async def api_info():
    """API info and available endpoints"""
    return {
        "message": "Memory Storytelling API for Dementia Patients",
        "version": "1.0.0",
        "endpoints": {
            "memories": "/memories/",
            "add_memory": "POST /memories/",
            "get_memory": "GET /memories/{id}/",
            "summarize": "POST /memories/{id}/summarize/",
            "tts": "GET /memories/{id}/tts/"
        }
    }

# Get all memories
@storytelling_router.get("/memories/", response_model=List[models.MemoryResponse])
async def get_memories(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Retrieve all stored memories with pagination"""
    memories = crud.get_memories(db, skip=skip, limit=limit)
    
    # Convert to response format with URLs
    response_memories = []
    for memory in memories:
        memory_data = {
            "id": memory.id,
            "title": memory.title,
            "description": memory.description,
            "created_at": memory.created_at,
            "image_url": crud.build_file_url(memory.image_path, static_dir) if memory.image_path else None,
            "summary": memory.summary,
            "audio_url": crud.build_file_url(memory.audio_path, static_dir) if memory.audio_path else None
        }
        response_memories.append(models.MemoryResponse(**memory_data))
    
    return response_memories

# Add new memory
@storytelling_router.post("/memories/", response_model=models.MemoryResponse)
async def create_memory(
    title: str = Form(...),
    description: str = Form(...),
    image: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """Create a new memory with optional image upload"""
    
    image_path = None
    
    # Handle image upload if provided
    if image:
        try:
            # Validate image file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Read image content
            image_content = await image.read()
            
            # Save image with unique name
            image_path = crud.save_uploaded_file(image_content, image.filename, static_dir)
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")
    
    # Create memory in database
    try:
        db_memory = crud.create_memory(db, title=title, description=description, image_path=image_path)
        
        # Prepare response
        memory_data = {
            "id": db_memory.id,
            "title": db_memory.title,
            "description": db_memory.description,
            "created_at": db_memory.created_at,
            "image_url": crud.build_file_url(db_memory.image_path, static_dir) if db_memory.image_path else None,
            "summary": db_memory.summary,
            "audio_url": crud.build_file_url(db_memory.audio_path, static_dir) if db_memory.audio_path else None
        }
        
        return models.MemoryResponse(**memory_data)
        
    except Exception as e:
        # Clean up uploaded image if database operation fails
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        raise HTTPException(status_code=500, detail=f"Memory creation failed: {str(e)}")

# Get specific memory
@storytelling_router.get("/memories/{memory_id}/", response_model=models.MemoryResponse)
async def get_memory(memory_id: int, db: Session = Depends(get_db)):
    """Get details of a specific memory by ID"""
    db_memory = crud.get_memory(db, memory_id=memory_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Prepare response with URLs
    memory_data = {
        "id": db_memory.id,
        "title": db_memory.title,
        "description": db_memory.description,
        "created_at": db_memory.created_at,
        "image_url": crud.build_file_url(db_memory.image_path, static_dir) if db_memory.image_path else None,
        "summary": db_memory.summary,
        "audio_url": crud.build_file_url(db_memory.audio_path, static_dir) if db_memory.audio_path else None
    }
    
    return models.MemoryResponse(**memory_data)

# Summarize memory text
@storytelling_router.post("/memories/{memory_id}/summarize/", response_model=models.SummarizeResponse)
async def summarize_memory(
    memory_id: int, 
    request: models.SummarizeRequest, 
    db: Session = Depends(get_db)
):
    """Generate simplified summary of memory text for dementia patients"""
    db_memory = crud.get_memory(db, memory_id=memory_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Use provided text or memory description
    text_to_summarize = request.text if request.text else db_memory.description
    
    try:
        # Generate summary using AI service
        summary = await ai_services.summarize_text(text_to_summarize)
        
        # Update memory with summary
        updated_memory = crud.update_memory_summary(db, memory_id, summary)
        
        return models.SummarizeResponse(summary=summary, memory_id=memory_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# Generate text-to-speech
@storytelling_router.get("/memories/{memory_id}/tts/", response_model=models.TTSResponse)
async def generate_tts(memory_id: int, db: Session = Depends(get_db)):
    """Generate audio file for memory text using text-to-speech"""
    db_memory = crud.get_memory(db, memory_id=memory_id)
    if db_memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    # Return existing audio if available and is a real MP3 file
    if (
        db_memory.audio_path 
        and os.path.exists(db_memory.audio_path)
        and db_memory.audio_path.lower().endswith('.mp3')
    ):
        audio_url = crud.build_file_url(db_memory.audio_path, static_dir)
        return models.TTSResponse(audio_url=audio_url, memory_id=memory_id)

    # If a placeholder text file exists from a previous failed attempt, remove it to regenerate audio
    if (
        db_memory.audio_path 
        and os.path.exists(db_memory.audio_path)
        and db_memory.audio_path.lower().endswith('.txt')
    ):
        try:
            os.remove(db_memory.audio_path)
        except Exception:
            pass
    
    try:
        # Use summary if available, otherwise use description
        text_for_tts = db_memory.summary if db_memory.summary else db_memory.description
        
        # Generate audio file
        audio_path = await ai_services.text_to_speech(text_for_tts, static_dir)
        
        # Update memory with audio path
        updated_memory = crud.update_memory_audio(db, memory_id, audio_path)
        
        # Build URL for response
        audio_url = crud.build_file_url(audio_path, static_dir)
        
        return models.TTSResponse(audio_url=audio_url, memory_id=memory_id)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-speech generation failed: {str(e)}")

# Health check endpoint
@storytelling_router.get("/health/")
async def health_check():
    """Health check endpoint with AI services status"""
    services_status = ai_services.get_available_services()
    return {
        "status": "healthy",
        "ai_services": services_status
    }

# Get AI services information
@storytelling_router.get("/ai-status/")
async def ai_status():
    """Get information about available AI services"""
    return ai_services.get_available_services()

# Frontend routes
@storytelling_router.get("/add", response_class=HTMLResponse, name="storytelling-add")
async def add_memory_page(request: Request):
    """Serve the add memory page"""
    return templates.TemplateResponse("storytelling/add.html", {"request": request})

@storytelling_router.get("/memories", response_class=HTMLResponse, name="storytelling-memories")
async def memories_page(request: Request):
    """Serve the memories page"""
    return templates.TemplateResponse("storytelling/memories.html", {"request": request})
