import os
from dotenv import load_dotenv
import random
import json
import uvicorn
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
import logging

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from fastapi import APIRouter


# Load environment variables from .env if present
load_dotenv()

# --- TEMPLATES & STATIC FILES ---
# Create directories if they don't exist
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Initialize Jinja2Templates
templates = Jinja2Templates(directory="templates")


# --- UNIFIED FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Integrated Patient & Caregiver API",
    description="A unified backend for a RAG chatbot, patient activities, and caregiver monitoring.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- USER AUTHENTICATION SETUP ---
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext
import jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi import status, Form
from fastapi.responses import RedirectResponse

SECRET_KEY = "your_secret_key_here"  # Change this in production
ALGORITHM = "HS256"

Base = declarative_base()
engine = create_engine("sqlite:///./users.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


# --- Register ENDPOINT ---
@app.post("/register")
async def register(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user:
        # Redirect back with an error message
        return RedirectResponse(url="/?msg=Username already taken&type=error", status_code=303)

    hashed_password = get_password_hash(password)
    new_user = User(username=username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return RedirectResponse(url="/?msg=Registered successfully&type=success", status_code=303)

# --- LOGIN ENDPOINT ---
@app.post("/login")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        return RedirectResponse(url="/?msg=Incorrect username or password&type=error", status_code=303)

    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/role-selection", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response


@app.get("/logout")
def logout(request: Request):
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie(key="access_token")
    return response

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# WEBSOCKET CONNECTION MANAGER FOR REAL-TIME UPDATES
# =============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.dashboard_connections: List[WebSocket] = []

    async def connect_dashboard(self, websocket: WebSocket):
        await websocket.accept()
        self.dashboard_connections.append(websocket)
        logger.info(f"Dashboard WebSocket connected. Total dashboard connections: {len(self.dashboard_connections)}")

    def disconnect_dashboard(self, websocket: WebSocket):
        if websocket in self.dashboard_connections:
            self.dashboard_connections.remove(websocket)
        logger.info(f"Dashboard WebSocket disconnected. Total dashboard connections: {len(self.dashboard_connections)}")

    async def broadcast_to_dashboards(self, message: dict):
        """Broadcast real-time updates to all connected dashboard clients"""
        if not self.dashboard_connections:
            return
        
        disconnected = []
        for connection in self.dashboard_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.disconnect_dashboard(conn)

# Global connection manager instance
connection_manager = ConnectionManager()

# Helper function to broadcast patient activity updates
async def broadcast_patient_activity_update(patient_id: str, activity_type: str, action: str, metadata: dict = None):
    """Helper function to broadcast patient activity updates to dashboard"""
    if connection_manager:
        message = {
            "type": "dashboard_update",
            "update_type": f"activity_{action}",  # 'activity_started', 'activity_ended'
            "patient_id": str(patient_id),
            "data": {
                "activity_type": activity_type,
                "action": action,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            await connection_manager.broadcast_to_dashboards(message)
            logger.info(f"Broadcasted {action} update for patient {patient_id}, activity: {activity_type}")
        except Exception as e:
            logger.error(f"Error broadcasting activity update: {e}")


# --- MIDDLEWARE & STATIC FILES ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")





@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main login page."""
    return templates.TemplateResponse("index.html", {"request": request})



@app.get("/role-selection", response_class=HTMLResponse)
async def read_home(request: Request):
    username = None
    access_token = request.cookies.get("access_token")
    if access_token and access_token.startswith("Bearer "):
        try:
            payload = jwt.decode(access_token.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except Exception:
            pass
    return templates.TemplateResponse("role_selection.html", {"request": request, "username": username})

@app.get("/patient", response_class=HTMLResponse)
async def read_patient_interface(request: Request):
    username = None
    access_token = request.cookies.get("access_token")
    if access_token and access_token.startswith("Bearer "):
        try:
            payload = jwt.decode(access_token.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except Exception:
            pass
    return templates.TemplateResponse("patient_interface.html", {"request": request, "username": username})

@app.get("/caregiver", response_class=HTMLResponse)
async def read_caregiver_dashboard(request: Request):
    username = None
    access_token = request.cookies.get("access_token")
    if access_token and access_token.startswith("Bearer "):
        try:
            payload = jwt.decode(access_token.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except Exception:
            pass
    return templates.TemplateResponse("unified_dashboard.html", {"request": request, "username": username})

@app.get("/dashboard", response_class=HTMLResponse)
async def read_unified_dashboard(request: Request):
    """Unified healthcare monitoring dashboard"""
    username = None
    access_token = request.cookies.get("access_token")
    if access_token and access_token.startswith("Bearer "):
        try:
            payload = jwt.decode(access_token.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
        except Exception:
            pass
    return templates.TemplateResponse("unified_dashboard.html", {"request": request, "username": username})

# =============================================================================
# WEBSOCKET ENDPOINTS FOR REAL-TIME UPDATES
# =============================================================================

@app.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    await connection_manager.connect_dashboard(websocket)
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "message": "Dashboard WebSocket connected successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for any client messages (like ping/pong for keepalive)
                data = await websocket.receive_text()
                # Echo back for keepalive purposes
                await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    except WebSocketDisconnect:
        pass
    finally:
        connection_manager.disconnect_dashboard(websocket)

# --- GAMES ROUTES (embedded in same window) ---
@app.get("/games", response_class=HTMLResponse)
async def games_index(request: Request):
    return templates.TemplateResponse("games/index.html", {"request": request})

@app.get("/games/Domino", response_class=HTMLResponse)
async def game_domino(request: Request):
    return templates.TemplateResponse("games/Domino.html", {"request": request})

@app.get("/games/memory_card", response_class=HTMLResponse)
async def game_memory_card(request: Request):
    return templates.TemplateResponse("games/memory_card.html", {"request": request})

@app.get("/games/Simon", response_class=HTMLResponse)
async def game_simon(request: Request):
    return templates.TemplateResponse("games/Simon.html", {"request": request})

@app.get("/games/jig", response_class=HTMLResponse)
async def game_jig(request: Request):
    return templates.TemplateResponse("games/jig.html", {"request": request})

@app.get("/games/block_blast", response_class=HTMLResponse)
async def game_block_blast(request: Request):
    return templates.TemplateResponse("games/block_blast.html", {"request": request})

# --- RELAXATION ROUTE ---
@app.get("/relaxation", response_class=HTMLResponse)
async def relaxation_page(request: Request):
    # Serve the full relaxation page so it can be embedded via iframe in the patient interface
    return templates.TemplateResponse("relaxation/relaxation.html", {"request": request})



# --- AI ART ROUTER (mount under /ai_art) ---
ai_router = APIRouter(prefix="/ai_art")

from fastapi.responses import RedirectResponse

@ai_router.get("/", response_class=HTMLResponse)
async def ai_art_index(request: Request):
    # The AI Art UI was embedded into the main patient interface.
    # Redirect direct /ai_art/ visits to the patient interface so users land in the integrated view.
    return RedirectResponse(url="/patient")

from fastapi import UploadFile, File, Form
import base64
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import requests
import datetime as _dt
from urllib.parse import quote

HF_API_URL = "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny"
HF_API_KEY = os.getenv("HF_API_KEY", "")
POLLINATIONS_FALLBACK_URL = "https://image.pollinations.ai/prompt/"

def _prepare_canny_conditioning(sketch_image):
    if len(sketch_image.shape) == 3:
        gray = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = sketch_image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_rgb

@ai_router.post("/process_drawing")
async def process_drawing(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', edges_bgr)
    processed_base64 = base64.b64encode(buffer).decode('utf-8')
    return JSONResponse({"status": "success", "processed_image": f"data:image/png;base64,{processed_base64}"})

@ai_router.post("/generate_image")
async def generate_image(drawing: UploadFile = File(...), description: str = Form(...), style: str = Form(default="realistic")):
    sketch_contents = await drawing.read()
    nparr = np.frombuffer(sketch_contents, np.uint8)
    sketch_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    sketch_resized = cv2.resize(sketch_img, (512, 512))
    canny_image = _prepare_canny_conditioning(sketch_resized)
    style_prompts = {
        "realistic": "photorealistic, high quality, detailed, realistic lighting and shadows",
        "cartoon": "cartoon style, colorful, friendly, simple shapes, cheerful",
        "anime": "anime style, vibrant colors, expressive, beautiful character design",
        "artistic": "artistic painting, impressionist style, beautiful brushstrokes, fine art"
    }
    style_modifier = style_prompts.get(style, "realistic, high quality")
    prompt = f"{description}, {style_modifier}, high quality, detailed, vibrant colors, peaceful, therapeutic, warm lighting"
    negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, extra limbs, text, watermark"
    # Try HF
    if HF_API_KEY:
        try:
            canny_pil = Image.fromarray(canny_image)
            img_buffer = io.BytesIO()
            canny_pil.save(img_buffer, format='PNG')
            canny_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            headers = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json", "Accept": "image/png"}
            payload = {
                "inputs": prompt,
                "image": canny_base64,
                "negative_prompt": negative_prompt,
                "controlnet_type": "canny",
                "parameters": {"num_inference_steps": 20, "guidance_scale": 7.5, "height": 512, "width": 512, "controlnet_conditioning_scale": 0.8, "seed": 42}
            }
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200 and 'image' in resp.headers.get('content-type', ''):
                image_base64 = base64.b64encode(resp.content).decode('utf-8')
                return JSONResponse({"status": "success", "generated_image": f"data:image/png;base64,{image_base64}", "prompt_used": prompt, "method": "HF ControlNet"})
        except Exception:
            pass
    # Fallback pollinations
    enhanced_prompt = f"IMPORTANT: Follow the exact lines and shapes in this sketch. {prompt}. The image must match the sketch composition and structure precisely."
    api_url = f"{POLLINATIONS_FALLBACK_URL}{quote(enhanced_prompt)}"
    r = requests.get(api_url, params={"width": "512", "height": "512", "model": "flux", "enhance": "true"}, timeout=60)
    if r.status_code == 200:
        image_base64 = base64.b64encode(r.content).decode('utf-8')
        return JSONResponse({"status": "success", "generated_image": f"data:image/png;base64,{image_base64}", "prompt_used": prompt, "method": "pollinations"})
    raise HTTPException(status_code=500, detail="Image generation failed")

@ai_router.post("/enhance_image")
async def enhance_image(file: UploadFile = File(...), enhancement_type: str = Form(default="brightness")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)
    if enhancement_type == "brightness":
        enhanced_image = ImageEnhance.Brightness(pil_image).enhance(1.3)
    elif enhancement_type == "contrast":
        enhanced_image = ImageEnhance.Contrast(pil_image).enhance(1.2)
    elif enhancement_type == "saturation":
        enhanced_image = ImageEnhance.Color(pil_image).enhance(1.4)
    elif enhancement_type == "sharpness":
        enhanced_image = ImageEnhance.Sharpness(pil_image).enhance(1.5)
    elif enhancement_type == "smooth":
        enhanced_image = pil_image.filter(ImageFilter.SMOOTH_MORE)
    elif enhancement_type == "warm":
        temp_image = ImageEnhance.Brightness(pil_image).enhance(1.1)
        enhanced_image = ImageEnhance.Color(temp_image).enhance(1.2)
    else:
        enhanced_image = ImageEnhance.Brightness(pil_image).enhance(1.2)
    buf = io.BytesIO()
    enhanced_image.save(buf, format='PNG')
    enhanced_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return JSONResponse({"status": "success", "enhanced_image": f"data:image/png;base64,{enhanced_base64}", "enhancement_type": enhancement_type})




app.include_router(ai_router)

# --- STORYTELLING FEATURE INTEGRATION ---
from storytelling_feature.routes import storytelling_router
app.include_router(storytelling_router, prefix="/storytelling", tags=["Storytelling"])

# --- DASHBOARD ROUTES INTEGRATION ---
from routes_dashboard import dashboard_router, set_connection_manager
set_connection_manager(connection_manager)  # Pass the connection manager to dashboard routes
app.include_router(dashboard_router, tags=["Dashboard"])


# --- CAREGIVER CHATBOT INTEGRATION (ported from chatbot_main.py) ---
# We'll initialize models on startup and expose streaming/text/audio endpoints so
# the UI (embedded in patient_interface.html) can call /chat and /audio-chat.
chatbot_globals = {}

from pathlib import Path
import tempfile
import uuid
import shutil
import traceback
import re

try:
    from config import SYSTEM_PROMPT, DB_FAISS_PATH, EMBEDDING_MODEL, OLLAMA_MODEL, MAX_UPLOAD_SIZE, ALLOWED_AUDIO_EXTENSIONS
except Exception:
    # Provide safe defaults if config isn't available (won't break imports)
    SYSTEM_PROMPT = ""
    DB_FAISS_PATH = "faiss_index"
    EMBEDDING_MODEL = None
    OLLAMA_MODEL = None
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024
    ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}


def _safe_write_upload_to_temp(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename).suffix or ".wav"
    temp_dir = Path(tempfile.gettempdir()) / "rag_audio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}{suffix}"
    with temp_path.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    return temp_path


def _cleanup_temp(path: Path):
    try:
        if path and path.exists():
            path.unlink()
    except Exception:
        pass


@app.on_event("startup")
async def chatbot_startup_event():
    """Initialize embeddings, vector store and LLM (best-effort)."""
    print("--- Chatbot: startup initialization (best-effort) ---")
    try:
        # Import heavy libraries lazily so importing main.py doesn't fail if deps missing
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.chat_models import ChatOllama
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
    except Exception as e:
        print("Chatbot dependencies are not available:", e)
        chatbot_globals['retrieval_chain'] = None
        chatbot_globals['retriever'] = None
        chatbot_globals['llm'] = None
        return

    # Load embedding model
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    except Exception as e:
        print("Error loading embedding model:", e)
        chatbot_globals['retrieval_chain'] = None
        chatbot_globals['retriever'] = None
        chatbot_globals['llm'] = None
        return

    # Ensure DB exists or create a minimal demo vector store
    try:
        if not Path(DB_FAISS_PATH).exists():
            print(f"Vector DB not found at {DB_FAISS_PATH}. Creating demo index...")
            from langchain.schema import Document
            sample_doc = Document(page_content="Welcome to the Caregiver AI Assistant. I'm here to help with caregiving questions.", metadata={"source": "system"})
            vector_store = FAISS.from_documents([sample_doc], embeddings)
            vector_store.save_local(DB_FAISS_PATH)
        else:
            vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print("Failed to load/create FAISS index:", e)
        chatbot_globals['retrieval_chain'] = None
        chatbot_globals['retriever'] = None
        chatbot_globals['llm'] = None
        return

    retriever = vector_store.as_retriever(search_kwargs={'k': 4})

    # Initialize Ollama LLM (best-effort)
    try:
        llm = ChatOllama(model=OLLAMA_MODEL, temperature=0.6) if OLLAMA_MODEL else None
        if llm:
            try:
                llm.invoke("Hello")
            except Exception:
                print("Warning: Unable to invoke Ollama model during startup check.")
    except Exception as e:
        print("Ollama initialization failed:", e)
        llm = None

    if llm:
        try:
            answer_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
            document_chain = create_stuff_documents_chain(llm, answer_prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
        except Exception as e:
            print("Failed to build retrieval chain:", e)
            retrieval_chain = None
    else:
        retrieval_chain = None

    chatbot_globals['retrieval_chain'] = retrieval_chain
    chatbot_globals['retriever'] = retriever
    chatbot_globals['llm'] = llm
    print("--- Chatbot initialization complete ---")


def _clamp_temperature(value: float, low: float = 0.2, high: float = 0.6) -> float:
    try:
        return max(low, min(high, float(value)))
    except Exception:
        return low


def _safe_reply(user_text: str) -> str:
    # Provide a concise, supportive, non-medical fallback
    return (
        "I’m here to help and listen. I can’t give medical advice, but I can offer general support and point you to helpful steps. "
        "If this is urgent, please contact a healthcare professional."
    )


@app.post("/api/chat")
async def chat_api(request: Request):
    try:
        data = await request.json()
        user_input = data.get("message")
        req_temp = data.get("temperature")
        if not user_input:
            return JSONResponse({"error": "No message provided."}, status_code=400)

        # Prefer existing retrieval chain unless a specific temperature is requested
        retrieval_chain = chatbot_globals.get('retrieval_chain')
        custom_chain = None

        if req_temp is not None:
            try:
                from langchain_community.chat_models import ChatOllama
                from langchain.prompts import ChatPromptTemplate
                from langchain.chains.combine_documents import create_stuff_documents_chain
                # reuse existing retriever
                retriever = chatbot_globals.get('retriever')
                if retriever is None:
                    retrieval_chain = None
                else:
                    temp = _clamp_temperature(req_temp)
                    if OLLAMA_MODEL:
                        llm = ChatOllama(model=OLLAMA_MODEL, temperature=temp)
                        answer_prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{input}")])
                        document_chain = create_stuff_documents_chain(llm, answer_prompt)
                        from langchain.chains import create_retrieval_chain
                        custom_chain = create_retrieval_chain(retriever, document_chain)
                    else:
                        custom_chain = None
            except Exception as e:
                print("Custom temp chain build failed:", e)
                custom_chain = None

        chain_to_use = custom_chain or retrieval_chain

        async def stream_generator():
            try:
                if chain_to_use:
                    async for chunk in chain_to_use.astream({"input": user_input}):
                        if answer_part := chunk.get('answer'):
                            yield answer_part
                else:
                    # Fallback safe single-chunk reply
                    yield _safe_reply(user_input)
            except Exception as e:
                yield _safe_reply(user_input)

        return StreamingResponse(stream_generator(), media_type="text/plain")
    except Exception as e:
        print("/api/chat error:", e)
        # Return a short safe reply even on handler error
        return StreamingResponse(iter([_safe_reply("")]), media_type="text/plain")


@app.post("/chat")
async def chat_alias(request: Request):
    return await chat_api(request)


@app.post("/api/audio-chat")
async def audio_chat(file: UploadFile = File(...)):
    temp_audio_path = None
    temp_tts_path = None
    try:
        ext = Path(file.filename or "").suffix.lower()
        if ext and ext not in ALLOWED_AUDIO_EXTENSIONS:
            return JSONResponse({"error": "Unsupported audio format."}, status_code=400)

        temp_audio_path = _safe_write_upload_to_temp(file)
        try:
            if temp_audio_path.stat().st_size > MAX_UPLOAD_SIZE:
                _cleanup_temp(temp_audio_path)
                return JSONResponse({"error": "File too large."}, status_code=400)
        except Exception:
            pass

        # Try faster-whisper then whisper as fallback
        transcription = ""
        language = ""
        try:
            from faster_whisper import WhisperModel
            fw_model = WhisperModel("small", device="cpu")
            segments, info = fw_model.transcribe(str(temp_audio_path), beam_size=5)
            transcription = " ".join([s.text for s in segments]).strip()
            language = getattr(info, 'language', '') or ''
            if not re.search(r"[\u0600-\u06FF]", transcription):
                try:
                    segments2, info2 = fw_model.transcribe(str(temp_audio_path), beam_size=5, language='ar')
                    forced = " ".join([s.text for s in segments2]).strip()
                    if re.search(r"[\u0600-\u06FF]", forced):
                        transcription = forced
                        language = 'ar'
                except Exception:
                    pass
        except Exception:
            try:
                import whisper
                model = whisper.load_model('small')
                result = model.transcribe(str(temp_audio_path))
                transcription = (result.get('text') or '').strip()
                language = result.get('language') or ''
            except Exception as ie:
                print('STT failed:', ie)
                return JSONResponse({
                    "text": "Could not transcribe audio on the server. Please try typing.",
                    "audio_url": None,
                    "transcription": "",
                    "warning": "No STT available on server."
                }, status_code=200)

        if not language:
            language = 'ar' if re.search(r"[\u0600-\u06FF]", transcription) else 'en'

        if not transcription:
            return JSONResponse({"text": "No clear speech detected.", "audio_url": None, "transcription": ""}, status_code=200)

        retrieval_chain = chatbot_globals.get('retrieval_chain')
        if not retrieval_chain:
            # Friendly fallback reply without RAG
            return JSONResponse({
                "text": _safe_reply(transcription),
                "audio_url": None,
                "transcription": transcription
            }, status_code=200)

        user_query = transcription
        if language and language.startswith('ar'):
            user_query = "تعليمات مهمة: أجب باللغة العربية الفصحى فقط دون استخدام أي لغة أخرى. ثم أجب على السؤال التالي:\n" + transcription

        full_answer = ""
        try:
            async for chunk in retrieval_chain.astream({"input": user_query}):
                if part := chunk.get('answer'):
                    full_answer += part
        except Exception as e:
            full_answer = _safe_reply(user_query)

        # TTS using gTTS if available
        try:
            from gtts import gTTS
        except Exception as te:
            print('gTTS not available:', te)
            return {"text": full_answer or "Response generated.", "audio_url": None, "transcription": transcription, "warning": "TTS not available"}

        tts_lang = 'ar' if language and language.startswith('ar') else 'en'
        temp_tts_path = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=full_answer, lang=tts_lang)
        tts.save(str(temp_tts_path))

        serve_dir = Path(tempfile.gettempdir()) / "rag_tts"
        serve_dir.mkdir(parents=True, exist_ok=True)
        target_path = serve_dir / temp_tts_path.name
        shutil.copy(str(temp_tts_path), str(target_path))

        return {"text": full_answer, "audio_url": f"/api/audio/{target_path.name}", "transcription": transcription}

    except Exception as e:
        print('Error in /api/audio-chat:', e)
        return JSONResponse({"error": "Internal server error."}, status_code=500)
    finally:
        if temp_audio_path:
            _cleanup_temp(temp_audio_path)


@app.post('/audio-chat')
async def audio_chat_alias(file: UploadFile = File(...)):
    return await audio_chat(file)


@app.get('/api/audio/{filename}')
async def serve_audio(filename: str):
    serve_dir = Path(tempfile.gettempdir()) / "rag_tts"
    file_path = serve_dir / filename
    if not file_path.exists():
        return JSONResponse({"error": "Audio file not found."}, status_code=404)
    return FileResponse(path=str(file_path), media_type='audio/mpeg', filename=filename)


@app.get('/audio/{filename}')
async def serve_audio_alias(filename: str):
    return await serve_audio(filename)


@app.get('/api/health')
async def health_check():
    llm_status = 'available' if chatbot_globals.get('llm') else 'unavailable'
    retrieval_status = 'ready' if chatbot_globals.get('retrieval_chain') else 'not ready'
    return {"status": "ok", "llm_status": llm_status, "retrieval_status": retrieval_status, "vector_db_path": DB_FAISS_PATH, "embedding_model": EMBEDDING_MODEL, "ollama_model": OLLAMA_MODEL}


@app.get('/health')
async def health_alias():
    return await health_check()


# =============================================================================
# PATIENT ACTIVITY TRACKING ENDPOINTS FOR REAL-TIME UPDATES
# =============================================================================

@app.post("/api/patient-activity/start")
async def start_patient_activity(request: Request):
    """Start tracking a patient activity and broadcast to dashboards"""
    try:
        data = await request.json()
        patient_id = data.get("patient_id", "1")
        activity_type = data.get("activity_type", "unknown")
        
        # Broadcast the activity start
        await broadcast_patient_activity_update(
            patient_id=patient_id,
            activity_type=activity_type,
            action="started"
        )
        
        return JSONResponse({
            "status": "success",
            "message": f"Activity {activity_type} started for patient {patient_id}"
        })
    except Exception as e:
        logger.error(f"Error starting patient activity: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/patient-activity/end")
async def end_patient_activity(request: Request):
    """End tracking a patient activity and broadcast to dashboards"""
    try:
        data = await request.json()
        patient_id = data.get("patient_id", "1")
        activity_type = data.get("activity_type", "unknown")
        duration = data.get("duration", 0)
        score = data.get("score")
        
        metadata = {"duration": duration}
        if score is not None:
            metadata["score"] = score
        
        # Broadcast the activity end
        await broadcast_patient_activity_update(
            patient_id=patient_id,
            activity_type=activity_type,
            action="ended",
            metadata=metadata
        )
        
        return JSONResponse({
            "status": "success",
            "message": f"Activity {activity_type} ended for patient {patient_id}"
        })
    except Exception as e:
        logger.error(f"Error ending patient activity: {e}")
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)


@app.get('/chatbot', response_class=HTMLResponse)
async def serve_chatbot_ui(request: Request):
    # Serve the standalone chatbot UI so we can embed it via iframe in the patient interface
    return templates.TemplateResponse('chatbot/all.html', {"request": request})


# --- UTIL
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=404, content={"message": "Endpoint not found", "path": str(request.url.path)})

@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(status_code=500, content={"message": "Internal server error"})

if __name__ == "__main__":
    logger.info("Starting FastAPI application...")
    # NOTE: Changed back to localhost from the original user-provided code for consistency
    uvicorn.run("main:app", host="localhost", port=8000, reload=True, log_level="info")
