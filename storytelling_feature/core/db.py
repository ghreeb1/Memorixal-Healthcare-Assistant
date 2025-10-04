"""
Core database module: configuration, models, schemas, and CRUD operations
Merged from database.py, models.py, and crud.py
"""
from __future__ import annotations

# Database configuration and setup for SQLite with SQLAlchemy
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func

# Pydantic schemas and typing
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

# Utilities
import os
import uuid

# ============================
# Database setup (from database.py)
# ============================

# Store SQLite DB in project root
DB_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "memories.db")

# SQLite database URL
SQLITE_DATABASE_URL = f"sqlite:///{DB_FILE_PATH}"

# Create SQLAlchemy engine
engine = create_engine(
    SQLITE_DATABASE_URL,
    connect_args={"check_same_thread": False}  # Needed for SQLite
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============================
# SQLAlchemy model and Pydantic schemas (from models.py)
# ============================

class MemoryDB(Base):
    """SQLAlchemy model for storing memories in database"""
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    image_path = Column(String(500), nullable=True)
    summary = Column(Text, nullable=True)  # Simplified text for dementia patients
    audio_path = Column(String(500), nullable=True)  # Path to TTS audio file
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Pydantic Models for API
class MemoryBase(BaseModel):
    """Base memory schema"""
    title: str
    description: str


class MemoryCreate(MemoryBase):
    """Schema for creating a new memory"""
    pass


class MemoryResponse(MemoryBase):
    """Schema for memory API responses"""
    id: int
    image_url: Optional[str] = None
    summary: Optional[str] = None
    audio_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class SummarizeRequest(BaseModel):
    """Schema for text summarization request"""
    text: Optional[str] = None  # If not provided, use memory description


class SummarizeResponse(BaseModel):
    """Schema for summarization response"""
    summary: str
    memory_id: int


class TTSResponse(BaseModel):
    """Schema for TTS response"""
    audio_url: str
    memory_id: int


# ============================
# CRUD operations and helpers (from crud.py)
# ============================

def get_memory(db: Session, memory_id: int) -> Optional[MemoryDB]:
    """Get a single memory by ID"""
    return db.query(MemoryDB).filter(MemoryDB.id == memory_id).first()


def get_memories(db: Session, skip: int = 0, limit: int = 100) -> List[MemoryDB]:
    """Get list of all memories with pagination"""
    return db.query(MemoryDB).offset(skip).limit(limit).all()


def create_memory(
    db: Session,
    title: str,
    description: str,
    image_path: Optional[str] = None
) -> MemoryDB:
    """Create a new memory in database"""
    db_memory = MemoryDB(
        title=title,
        description=description,
        image_path=image_path
    )
    db.add(db_memory)
    db.commit()
    db.refresh(db_memory)
    return db_memory


def update_memory_summary(db: Session, memory_id: int, summary: str) -> Optional[MemoryDB]:
    """Update memory with generated summary"""
    db_memory = get_memory(db, memory_id)
    if db_memory:
        db_memory.summary = summary
        db.commit()
        db.refresh(db_memory)
    return db_memory


def update_memory_audio(db: Session, memory_id: int, audio_path: str) -> Optional[MemoryDB]:
    """Update memory with generated audio file path"""
    db_memory = get_memory(db, memory_id)
    if db_memory:
        db_memory.audio_path = audio_path
        db.commit()
        db.refresh(db_memory)
    return db_memory


def save_uploaded_file(file_content: bytes, original_filename: str, static_base_dir: str) -> str:
    """Save uploaded file with unique name and return the file path"""
    # Create upload directory in static/images
    upload_dir = os.path.join(static_base_dir, "images")
    os.makedirs(upload_dir, exist_ok=True)

    # Generate unique filename
    file_extension = os.path.splitext(original_filename)[1] if original_filename else ""
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


def build_file_url(file_path: str, static_base_dir: str) -> Optional[str]:
    """Convert file path to accessible URL"""
    if file_path and os.path.exists(file_path):
        try:
            # Get relative path from static directory
            rel_path = os.path.relpath(file_path, static_base_dir)
            # Ensure forward slashes for URLs
            url_path = rel_path.replace(os.path.sep, '/')
            return f"/static/{url_path}"
        except ValueError:
            # Path is not relative to static dir
            return f"/static/{os.path.basename(file_path)}"
    return None
