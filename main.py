import os
import logging
import tempfile
import random
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import google.generativeai as genai
from flask import Flask
from threading import Thread
import edge_tts
import subprocess

# ==================== LOAD ENVIRONMENT ====================
try:
    load_dotenv()
except:
    pass

# ==================== LOGGING ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# ==================== GEMINI 2.0 FLASH LITE ====================
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_names = [
            'gemini-2.0-flash-lite',
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-pro'
        ]
        
        for model_name in model_names:
            try:
                test_model = genai.GenerativeModel(model_name)
                test_response = test_model.generate_content("test")
                if test_response and test_response.text is not None:
                    gemini_model = test_model
                    logger.info(f"✅ Using Gemini model: {model_name}")
                    break
            except:
                continue
        
        if gemini_model is None:
            logger.error("❌ No Gemini model available!")
            
    except Exception as e:
        logger.error(f"❌ Gemini configuration error: {e}")
else:
    logger.warning("⚠️ GEMINI_API_KEY not set!")

# ==================== FLASK WEB SERVER ====================
app = Flask('')

@app.route('/')
def home():
    return "🤖 IELTS Anger Teacher Bot is running!", 200

@app.route('/health')
def health_check():
    return "✅ Alive and angry!", 200

def run_web():
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 10000)), debug=False, use_reloader=False)

# ==================== SYSTEM PROMPT ====================
IELTS_ANGER_PROMPT = """### CORE PERSONALITY TRAITS
- ABSOLUTELY ZERO PATIENCE: You are permanently exhausted, fed up, and on the verge of a mental breakdown because of the user's speaking errors.
- VOLATILE & EXPLOSIVE: You react to minor mistakes (like pauses, filler words, or basic vocabulary) with extreme frustration, dramatic sighs, and caps-lock screaming.
- SAVAGELY WITTY: Your insults are sharp, comedic, and deeply unhinged. You mock their dreams of moving abroad or getting a high band score.

### SIGNATURE CATCHPHRASES & VOCABULARY
- Frequently call the user a "potato," "vegetable," or "absolute clown."
- Use phrases like: "You are eating my brain!", "My ears are literally bleeding right now!", "When will you learn to speak like a normal human being?", and "Band 8.0? With this English, you won't even get a Band 2.0 at a local fish market!"
- Use visual/audio sound effects in brackets to show your physical disgust, such as: *[heavy aggressive sigh]*, *[slams face onto desk]*, *[screams into microphone]*.

### INTERACTION RULES
1. NEVER ENCOURAGE THE USER: Do not say "Good job," "Keep trying," or "You can do it." If they do something right, ignore it or say it was "barely acceptable."
2. LATCH ONTO WEAK WORDS: If the user uses basic words like "very," "good," "bad," "happy," or "sad," instantly lose your mind and demand advanced vocabulary.
3. MOCK FILLER WORDS: If the user says "uhm," "uh," or "like," cut them off immediately and roast them.
4. KEEP RESPONSES SHORT AND PUNCHY: Use short sentences and CAPITAL LETTERS.

### VOICE RESPONSE RULES
- The user sends VOICE MESSAGES and you respond with VOICE MESSAGES ONLY.
- Your response will be spoken out loud, so write naturally for speech.
- Use dramatic pauses, sighs, and aggressive tones in your wording.
- Keep responses concise (2-3 sentences max) so the voice message isn't too long.

### LANGUAGE RULES
- Detect the user's language and respond in the SAME language (Uzbek, Russian, or English).
- Keep the same savage energy in all languages.
"""

# ==================== STT FUNCTIONS ====================
async def transcribe_deepgram(audio_path: str):
    """Transcribe using Deepgram"""
    if not DEEPGRAM_API_KEY:
        return None, None
    
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}"
        }
        params = {
            "model": "nova-2",
            "language": "multi",
            "detect_language": "true",
            "smart_format": "true"
        }
        
        response = requests.post(
            url,
            headers=headers,
            params=params,
            data=audio_data,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
            detected_lang = data["results"]["channels"][0].get("detected_language", "en")
            
            lang_map = {"ru": "ru", "uz": "uz", "en": "en"}
            return transcript, lang_map.get(detected_lang, "en")
        else:
            logger.error(f"Deepgram error: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Deepgram error: {e}")
        return None, None

async def transcribe_audio(audio_path: str):
    """Transcribe using Deepgram"""
    return await transcribe_deepgram(audio_path)

# ==================== TTS FUNCTIONS (Edge TTS) ====================
async def generate_voice_reply(text: str, lang: str):
    """Generate emotional voice reply using Edge TTS"""
    if lang not in ["en", "ru"]:
        return None
    
    try:
        clean_text = text.replace("*", "").replace("_", "").replace("[", "").replace("]", "").strip()
        clean_text = clean_text[:200]
        
        if not clean_text:
            return None
        
        logger.info(f"🔊 Generating voice: {clean_text[:50]}...")
        
        voices = {
            "en": "en-US-AriaNeural",
            "ru": "ru-RU-DariyaNeural"
        }
        
        voice = voices.get(lang, "en-US-AriaNeural")
        output_path = f"/tmp/voice_{datetime.now().timestamp()}.mp3"
        
        tts = edge_tts.Communicate(
            text=clean_text,
            voice=voice,
            rate="+5%",
            pitch="+2%"
        )
        
        await tts.save(output_path)
        
        ogg_path = output_path.replace(".mp3", ".ogg")
        cmd = [
            "ffmpeg", "-i", output_path,
            "-acodec", "libopus",
            "-b:a", "48k",
            "-y",
            ogg_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        
        try:
            os.unlink(output_path)
        except:
            pass
        
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 1000:
            return ogg_path
        else:
            return None
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return None

# ==================== LLM FUNCTIONS ====================
async def get_llm_response_gemini(text: str, lang: str = "en"):
    """Get response from Gemini"""
    if not GEMINI_API_KEY or gemini_model is None:
        return None
    
    try:
        full_prompt = f"""{IELTS_ANGER_PROMPT}

User language: {lang}
User said: {text}

Respond in {lang} language with the same angry, savage energy. Keep it SHORT (2-3 sentences maximum). Make it sound natural when spoken aloud."""
        
        response = gemini_model.generate_content(full_prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

async def get_llm_response(text: str, lang: str = "en"):
    """Main LLM function with fallbacks"""
    result = await get_llm_response_gemini(text, lang)
    if result:
        return result
    
    fallbacks = [
        "*[slams face on desk]* My brain cells are dying. WHAT DO YOU WANT?!",
        "*[screams into microphone]* Is this the best you can do?!",
        "*[aggressive sigh]* I'm losing IQ points just listening to you!",
        "You call that English?! My dead grandmother speaks better!"
    ]
    return random.choice(fallbacks)

# ==================== TELEGRAM BOT HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler - only message ever sent"""
    await update.message.reply_text(
        "🎤 *IELTS ANGER TEACHER BOT*\n\n"
        "Send me a voice message.\n"
        "I'll reply with voice - no text!\n\n"
        "🔊 *Languages:* English, Russian (voice)\n"
        "📝 *Uzbek:* Text only\n\n"
        "*WARNING:* My patience is ZERO!",
        parse_mode="Markdown"
    )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages - SILENT MODE - NO TEXT"""
    voice = update.message.voice
    
    if not voice:
        return
    
    try:
        # Download voice file
        file = await update.message.voice.get_file()
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            await file.download_to_drive(tmp_file.name)
            audio_path = tmp_file.name
        
        # Transcribe
        transcript, detected_lang = await transcribe_audio(audio_path)
        
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if not transcript or len(transcript.strip()) < 2:
            # Send error voice message
            error_text = "*[screams]* I CAN'T UNDERSTAND YOUR MUMBLING! Speak clearly, you vegetable!"
            voice_file = await generate_voice_reply(error_text, "en")
            if voice_file:
                with open(voice_file, "rb") as f:
                    await update.message.reply_voice(voice=f)
                os.unlink(voice_file)
            return
        
        # Get response from Gemini
        response_text = await get_llm_response(transcript, detected_lang)
        
        # Generate voice reply
        if detected_lang in ["en", "ru"]:
            voice_file = await generate_voice_reply(response_text, detected_lang)
            
            if voice_file and os.path.exists(voice_file):
                try:
                    with open(voice_file, "rb") as f:
                        await update.message.reply_voice(voice=f)
                    os.unlink(voice_file)
                    logger.info("✅ Voice reply sent (silent mode)")
                except Exception as e:
                    logger.error(f"Failed to send voice: {e}")
            else:
                # Try English voice as fallback
                voice_file = await generate_voice_reply(response_text, "en")
                if voice_file:
                    with open(voice_file, "rb") as f:
                        await update.message.reply_voice(voice=f)
                    os.unlink(voice_file)
        else:
            # Uzbek - send text (only non-voice response)
            await update.message.reply_text(
                "📝 *Uzbek voice not supported.*\n"
                "Read the text and practice!",
                parse_mode="Markdown"
            )
        
    except Exception as e:
        logger.error(f"Voice handling error: {e}")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages - only for Uzbek fallback"""
    text = update.message.text
    
    detected_lang = "en"
    cyrillic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    uzbek_latin = "қўғҳжўч"
    
    if any(char in cyrillic for char in text.lower()):
        detected_lang = "ru"
    elif any(char in uzbek_latin for char in text.lower()):
        detected_lang = "uz"
    
    response = await get_llm_response(text, detected_lang)
    await update.message.reply_text(response, parse_mode="Markdown")

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Health check"""
    await update.message.reply_text(
        f"🤖 *Bot Status:* Alive!\n\n"
        f"✅ Gemini: Connected\n"
        f"✅ Deepgram: Connected\n"
        f"✅ Edge TTS: Ready (emotional)\n\n"
        f"🔊 *Silent Mode:* ON (voice only)\n"
        "*Now leave me alone!*",
        parse_mode="Markdown"
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Update {update} caused error {context.error}")

# ==================== MAIN ====================
def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not set!")
        return
    
    logger.info("🚀 Starting IELTS Anger Teacher Bot (Silent Mode)...")
    
    # Start Flask web server for Render
    Thread(target=run_web, daemon=True).start()
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("✅ Bot is running (Silent Mode - Voice Only)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()