"""
AI Services for text summarization and text-to-speech
Uses Gemini (Google Generative AI) for summarization
and gTTS for TTS fallback
"""
import os
import uuid
from dotenv import load_dotenv
import google.generativeai as genai

# تحميل الـ env variables
load_dotenv()

# قراءة مفتاح Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# تهيئة Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # Try different model names in order of preference
    try:
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
    except:
        try:
            gemini_model = genai.GenerativeModel("gemini-1.0-pro")
        except:
            try:
                gemini_model = genai.GenerativeModel("gemini-pro")
            except:
                gemini_model = None
                print("لم يتم العثور على نموذج Gemini متاح")
else:
    gemini_model = None
    print("لم يتم العثور على GEMINI_API_KEY في ملف .env")

# Optional local TTS fallback (gTTS)
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

async def summarize_text(text: str) -> str:
    """
    Summarize and simplify text for dementia patients
    Uses Gemini first, then local fallback
    """
    # Try Gemini first
    if gemini_model:
        try:
            response = gemini_model.generate_content(
                f"قم بتبسيط هذا النص ليكون مناسباً لمريض الخرف:\n\n{text}"
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini summarization failed: {e}")
    
    # Local fallback - simple text truncation
    print("Using local fallback for text summarization")
    sentences = text.split('.')
    simple_text = '. '.join(sentences[:3]).strip()
    if len(simple_text) > 200:
        simple_text = simple_text[:200] + "..."
    
    return simple_text if simple_text else "ملخص بسيط للذكرى."

async def text_to_speech(text: str, static_base_dir: str) -> str:
    """
    Convert text to speech audio file
    Uses gTTS (local dependency) as fallback
    """
    output_dir = os.path.join(static_base_dir, "audio")
    os.makedirs(output_dir, exist_ok=True)

    audio_filename = f"tts_{uuid.uuid4()}.mp3"
    audio_path = os.path.join(output_dir, audio_filename)

    # gTTS fallback
    if GTTS_AVAILABLE:
        try:
            tts = gTTS(text=text, lang='ar')
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            print(f"gTTS failed: {e}")

    # Final fallback: create a placeholder text file
    print("No TTS service available. Audio generation skipped.")
    placeholder_path = audio_path.replace('.mp3', '.txt')
    with open(placeholder_path, 'w', encoding='utf-8') as f:
        f.write(f"نص الذكرى: {text}")
    return placeholder_path

def get_available_services() -> dict:
    """Return information about which AI services are available"""
    return {
        "gemini_available": bool(gemini_model),
        "gtts_available": bool(GTTS_AVAILABLE),
        "services_info": {
            "summarization": "Gemini" if gemini_model else "Local fallback",
            "tts": "gTTS" if GTTS_AVAILABLE else "Local fallback"
        }
    }

# Example usage and testing functions
async def test_ai_services():
    """Test function for AI services"""
    test_text = "هذه ذكرى جميلة من طفولتي عندما كنت ألعب في الحديقة مع أصدقائي."

    print("Testing summarization...")
    summary = await summarize_text(test_text)
    print(f"Summary: {summary}")

    print("Testing TTS...")
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)
    audio_path = await text_to_speech(test_text, static_dir)
    print(f"Audio saved to: {audio_path}")

    print("Available services:")
    services = get_available_services()
    print(services)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ai_services())
