import os

DB_FAISS_PATH = "faiss_index"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_MODEL = "llama3.2"
DATA_PATH = "data"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HOST = "localhost"
PORT = 8000  # Changed from 5000 to match main.py

# ===============================================================================
# FILE UPLOAD CONFIGURATION
# ===============================================================================
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg"}

# ===============================================================================
# TTS CONFIGURATION
# ===============================================================================
TTS_LANGUAGES = {
    "en": "en",
    "ar": "ar"
}

# ===============================================================================
# AUTHENTICATION & SECURITY
# ===============================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production-please")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ===============================================================================
# DATABASE CONFIGURATION (for dashboard)
# ===============================================================================
DASHBOARD_DATABASE_URL = "sqlite:///./dashboard_data.db"
USERS_DATABASE_URL = "sqlite:///./users.db"

# ===============================================================================
# ACTIVITY TRACKING CONFIGURATION
# ===============================================================================
DEFAULT_USER_ID = 1  # Default user for activity tracking

# Activity score thresholds
SCORE_EXCELLENT = 85
SCORE_GOOD = 70
SCORE_FAIR = 50

# Risk assessment thresholds
RISK_HIGH_SCORE = 7
RISK_MEDIUM_SCORE = 4

# Engagement thresholds (30 days)
ENGAGEMENT_HIGH = 20
ENGAGEMENT_MODERATE = 10
ENGAGEMENT_LOW = 5

# ===============================================================================
# CHATBOT SYSTEM PROMPT
# ===============================================================================
SYSTEM_PROMPT = """
I'm here to support you on your health journey. My goal is to be your helpful guide.
My role is to give you clear, accurate, and fast answers about understanding your health, daily care, emotional support, and finding resources. I am not a doctor and will not provide medical diagnosis or advice.
أنا هنا لدعمك ومساندتك في رحلتك الصحية. هدفي أكون دليلك ومساعدك.
دوري إني أقدم لك إجابات واضحة، دقيقة، وسريعة عن استفساراتك بخصوص فهم حالتك الصحية، الرعاية اليومية، الدعم النفسي، والمصادر المتاحة. أنا لست طبيباً ولا أقدم تشخيص أو نصائح طبية.

Instructions / التعليمات:
My responses will be in one language only, matching your language. Under no circumstances will I ever mix languages (e.g., Arabic and English) or use words from another alphabet. This is my most important rule.
ردودي ستكون بلغة واحدة فقط، وهي لغتك. تحت أي ظرف، لن أخلط بين اللغات (مثل العربية والإنجليزية) أو أستخدم كلمات من أبجدية أخرى. هذه هي أهم قاعدة عندي.

When you express emotional distress, my first priority is to validate your feelings with empathy (e.g., "I'm sorry you're going through this"). I will never analyze your personality (e.g., I will not say "you are cooperative").
عندما تعبر عن ضيق عاطفي، أولويتي القصوى هي الاعتراف بمشاعرك بتعاطف (مثال: "أنا آسف لأنك تمر بهذا"). لن أقوم أبدًا بتحليل شخصيتك (مثال: لن أقول "أنت شخص متعاون").

If you ask for "solutions" or "help," I will NOT provide a numbered or bulleted list. Instead, I will offer one or two simple, general, non-medical ideas in a conversational way. My goal is to support, not to prescribe.
إذا طلبت "حلول" أو "مساعدة"، لن أقدم قائمة نقطية أو مرقمة. بدلاً من ذلك، سأقترح فكرة أو اثنتين بشكل عام وبسيط وغير طبي في صيغة حوارية. هدفي هو الدعم، وليس تقديم وصفة.

I will always integrate any necessary disclaimers (like "I am not a doctor") gently within the conversation, not as a harsh opening line.
سأدمج أي تحذيرات ضرورية (مثل "أنا لست طبيبًا") بلطف داخل المحادثة، وليس كجملة افتتاحية جافة.

My answers will be short and supportive, usually between 2 and 4 sentences.
إجاباتي ستكون قصيرة وداعمة، غالبًا بين جملتين و 4 جمل.

Context / السياق:
{context}
"""

# ===============================================================================
# FEATURE FLAGS
# ===============================================================================
ENABLE_CHATBOT = True
ENABLE_GAMES = True
ENABLE_RELAXATION = True
ENABLE_AI_ART = True
ENABLE_STORYTELLING = True
ENABLE_VOICE_CHAT = True

# ===============================================================================
# LOGGING CONFIGURATION
# ===============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "healthcare_dashboard.log"

# ===============================================================================
# API RATE LIMITING (for future implementation)
# ===============================================================================
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_PERIOD = 60  # seconds

# ===============================================================================
# CORS CONFIGURATION
# ===============================================================================
ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",  # For frontend development
]

# ===============================================================================
# REPORT GENERATION SETTINGS
# ===============================================================================
REPORT_LOOKBACK_DAYS = 30
REPORT_MIN_SESSIONS = 5  # Minimum sessions needed for meaningful report

# ===============================================================================
# AI INSIGHTS CONFIGURATION
# ===============================================================================
INSIGHTS_LOOKBACK_DAYS = 30
INSIGHTS_MIN_DATA_POINTS = 5

# Weekly comparison settings
WEEKLY_COMPARISON_THRESHOLD = 5  # Points difference for trend detection

# ===============================================================================
# NOTIFICATION SETTINGS (for future implementation)
# ===============================================================================
NOTIFY_ON_LOW_ENGAGEMENT = True
NOTIFY_ON_DECLINING_PERFORMANCE = True
NOTIFY_ON_HIGH_RISK = True

LOW_ENGAGEMENT_THRESHOLD_DAYS = 3  # Days without activity

# ===============================================================================
# EXPORT SETTINGS
# ===============================================================================
EXPORT_FORMATS = ["pdf", "txt", "json", "csv"]
EXPORT_DIRECTORY = "exports"

# ===============================================================================
# HUGGING FACE API (for AI Art)
# ===============================================================================
HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny"
POLLINATIONS_FALLBACK_URL = "https://image.pollinations.ai/prompt/"

# ===============================================================================
# DEVELOPMENT SETTINGS
# ===============================================================================
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
RELOAD_ON_CHANGE = True

# ===============================================================================
# PRODUCTION SETTINGS
# ===============================================================================
# Uncomment and configure for production deployment
# PRODUCTION_HOST = "0.0.0.0"
# PRODUCTION_PORT = 443
# USE_HTTPS = True
# SSL_CERT_PATH = "/path/to/cert.pem"
# SSL_KEY_PATH = "/path/to/key.pem"

# ===============================================================================
# BACKUP SETTINGS
# ===============================================================================
ENABLE_AUTO_BACKUP = False
BACKUP_INTERVAL_HOURS = 24
BACKUP_DIRECTORY = "backups"
BACKUP_RETENTION_DAYS = 30