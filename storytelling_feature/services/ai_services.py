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
    # Try different model names in order of preference (current available models)
    model_names = [
        "gemini-2.0-flash",
        "gemini-2.5-flash", 
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-pro",
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-flash-latest"
    ]
    
    gemini_model = None
    for model_name in model_names:
        try:
            gemini_model = genai.GenerativeModel(model_name)
            print(f"Successfully initialized Gemini model: {model_name}")
            break
        except Exception as e:
            continue
    
    if gemini_model is None:
        print("لم يتم العثور على نموذج Gemini متاح - سيتم استخدام النسخة المحلية")
else:
    gemini_model = None
    print("لم يتم العثور على GEMINI_API_KEY في ملف .env - سيتم استخدام النسخة المحلية")

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
    if not text:
        return ""
    
    # Try Gemini first if available
    if gemini_model:
        try:
            # Use simpler prompt to avoid quota issues
            prompt = f"اجعل هذا النص أبسط وأقصر:\n\n{text}" if any('\u0600' <= c <= '\u06FF' for c in text) else f"Make this text simpler and shorter:\n\n{text}"
            
            response = gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                print("Gemini API quota exceeded, using local fallback")
            elif "403" in error_msg:
                print("Gemini API access denied, check API key")
            else:
                print(f"Gemini summarization failed: {e}")
    
    # Local fallback - improved text processing
    print("Using local fallback for text summarization")
    try:
        # Clean and prepare text
        clean_text = text.strip()
        
        # If text is already short enough, return as is
        if len(clean_text) <= 200:
            return clean_text
        
        # Try to split by sentences (Arabic and English)
        sentence_endings = ['.', '。', '؟', '!', '؟', '！']
        sentences = []
        current_sentence = ""
        
        for char in clean_text:
            current_sentence += char
            if char in sentence_endings:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text as last sentence if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Select sentences that fit within 200 characters
        selected_sentences = []
        char_count = 0
        
        for sentence in sentences:
            if sentence and char_count + len(sentence) <= 190:  # Leave some margin
                selected_sentences.append(sentence)
                char_count += len(sentence) + 1
            else:
                break
        
        if selected_sentences:
            result = ' '.join(selected_sentences)
            # Ensure proper ending
            if not any(result.endswith(ending) for ending in sentence_endings):
                result += '.' if not any('\u0600' <= c <= '\u06FF' for c in result) else '.'
            return result
        else:
            # If no complete sentences fit, take first 200 chars with smart truncation
            truncated = clean_text[:190]
            # Try to end at a word boundary
            last_space = truncated.rfind(' ')
            if last_space > 150:  # Only if we can save a reasonable amount
                truncated = truncated[:last_space]
            return truncated + "..."
                
    except Exception as e:
        print(f"Local summarization failed: {e}")
        # Final fallback
        return "ملخص بسيط للذكرى." if any('\u0600' <= c <= '\u06FF' for c in text) else "Simple memory summary."

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
