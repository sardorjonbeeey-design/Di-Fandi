import os
import logging
import tempfile
import base64
import json
import random
import asyncio
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import google.generativeai as genai
from flask import Flask
from threading import Thread
from gtts import gTTS
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# ==================== GEMINI 2.0 FLASH LITE ====================
gemini_model = None

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Try Gemini 2.0 Flash Lite first (fastest, best free tier)
        model_names = [
            'gemini-2.0-flash-lite',    # Fastest, best limits
            'gemini-2.0-flash-lite-preview-02-05',  # Preview version
            'gemini-2.0-flash',         # Regular flash
            'gemini-1.5-flash',         # Stable fallback
            'gemini-pro'                # Legacy fallback
        ]
        
        for model_name in model_names:
            try:
                logger.info(f"🔄 Trying Gemini model: {model_name}")
                test_model = genai.GenerativeModel(model_name)
                # Quick test
                test_response = test_model.generate_content("test")
                if test_response and test_response.text is not None:
                    gemini_model = test_model
                    logger.info(f"✅ Using Gemini model: {model_name}")
                    break
            except Exception as e:
                logger.warning(f"❌ Model {model_name} failed: {str(e)[:50]}")
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
- The user sends VOICE MESSAGES and you respond with VOICE MESSAGES.
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
        logger.error("❌ DEEPGRAM_API_KEY not set!")
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
        
        logger.info("🎤 Sending audio to Deepgram...")
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
            logger.info(f"✅ Deepgram transcribed: {transcript[:50]}...")
            return transcript, lang_map.get(detected_lang, "en")
        else:
            logger.error(f"❌ Deepgram error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        logger.error(f"❌ Deepgram error: {e}")
        return None, None

async def transcribe_audio(audio_path: str):
    """Transcribe using Deepgram"""
    result = await transcribe_deepgram(audio_path)
    if result and result[0]:
        return result
    
    logger.error("❌ All transcription methods failed!")
    return None, None

# ==================== TTS FUNCTIONS ====================
async def generate_voice_reply(text: str, lang: str):
    """Generate voice reply using gTTS"""
    if lang not in ["en", "ru"]:
        return None
    
    try:
        clean_text = text.replace("*", "").replace("_", "").replace("[", "").replace("]", "").strip()
        clean_text = clean_text[:200]
        
        if not clean_text:
            return None
        
        logger.info(f"🔊 Generating TTS for: {clean_text[:50]}...")
        
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        mp3_path = f"/tmp/voice_{datetime.now().timestamp()}.mp3"
        tts.save(mp3_path)
        
        ogg_path = mp3_path.replace(".mp3", ".ogg")
        cmd = [
            "ffmpeg", "-i", mp3_path,
            "-acodec", "libopus",
            "-b:a", "48k",
            "-y",
            ogg_path
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        
        try:
            os.unlink(mp3_path)
        except:
            pass
        
        if os.path.exists(ogg_path) and os.path.getsize(ogg_path) > 1000:
            logger.info(f"✅ Voice file created: {ogg_path}")
            return ogg_path
        else:
            logger.error("❌ FFmpeg conversion failed")
            return None
            
    except Exception as e:
        logger.error(f"❌ TTS error: {e}")
        return None

# ==================== LLM FUNCTIONS ====================
async def get_llm_response_gemini(text: str, lang: str = "en"):
    """Get response from Gemini"""
    if not GEMINI_API_KEY:
        logger.error("❌ GEMINI_API_KEY not set!")
        return None
    
    if gemini_model is None:
        logger.error("❌ No Gemini model available!")
        return None
    
    try:
        full_prompt = f"""{IELTS_ANGER_PROMPT}

User language: {lang}
User said: {text}

Respond in {lang} language with the same angry, savage energy. Keep it SHORT (2-3 sentences maximum). Make it sound natural when spoken aloud."""
        
        response = gemini_model.generate_content(full_prompt)
        
        if response and response.text:
            logger.info(f"✅ Gemini response: {response.text[:50]}...")
            return response.text.strip()
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Gemini error: {e}")
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
    """Start command handler"""
    model_name = "Unknown"
    if gemini_model:
        try:
            model_name = str(gemini_model._model_name).replace("models/", "")
        except:
            model_name = "Gemini 2.0 Flash Lite"
    
    await update.message.reply_text(
        f"🎤 *IELTS ANGER TEACHER BOT*\n\n"
        "Send me a *voice message* in English or Russian.\n"
        "I'll criticize you mercilessly with VOICE replies!\n\n"
        f"🧠 *AI Model:* {model_name}\n"
        "🎤 *STT:* Deepgram\n"
        "🔊 *TTS:* gTTS (free)\n\n"
        "🔊 *Voice replies:* English & Russian\n"
        "📝 *Uzbek:* Text only\n\n"
        "*WARNING:* My patience is ZERO!",
        parse_mode="Markdown"
    )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    voice = update.message.voice
    
    if not voice:
        await update.message.reply_text("*[aggressive sigh]* Send a VOICE message, you potato!")
        return
    
    await update.message.reply_text("*[cracks knuckles]* Let's see what trash you sent me...")
    
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
            await update.message.reply_text("*[screams]* I CAN'T UNDERSTAND YOUR MUMBLING! Speak clearly, you vegetable!")
            return
        
        # Send transcription
        await update.message.reply_text(
            f"🗣️ *You said:*\n_{transcript[:300]}_\n\n"
            f"🌍 *Language:* {detected_lang.upper()}",
            parse_mode="Markdown"
        )
        
        # Get response from Gemini
        response_text = await get_llm_response(transcript, detected_lang)
        
        # Send text response
        await update.message.reply_text(response_text, parse_mode="Markdown")
        
        # Try voice reply
        if detected_lang in ["en", "ru"]:
            await update.message.reply_text("🔊 *Generating voice reply...*", parse_mode="Markdown")
            
            voice_file = await generate_voice_reply(response_text, detected_lang)
            
            if voice_file and os.path.exists(voice_file):
                try:
                    with open(voice_file, "rb") as f:
                        await update.message.reply_voice(voice=f)
                    os.unlink(voice_file)
                    logger.info("✅ Voice reply sent successfully!")
                except Exception as e:
                    logger.error(f"❌ Failed to send voice: {e}")
                    await update.message.reply_text("*[voice failed]* Just read my text, potato!")
            else:
                await update.message.reply_text("*[TTS failed]* Read my text and cry about it!")
        else:
            await update.message.reply_text(
                "📝 *Uzbek voice not supported yet.*\n"
                "Read the text and practice!",
                parse_mode="Markdown"
            )
        
    except Exception as e:
        logger.error(f"❌ Voice handling error: {e}")
        await update.message.reply_text("*[system error]* YOU BROKE ME! Congratulations!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
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
    model_name = "Unknown"
    if gemini_model:
        try:
            model_name = str(gemini_model._model_name).replace("models/", "")
        except:
            model_name = "Gemini 2.0 Flash Lite"
    
    await update.message.reply_text(
        f"🤖 *Bot Status:* Alive!\n\n"
        f"🧠 *Gemini:* {model_name} ✅\n"
        f"🎤 *Deepgram:* {'✅ Connected' if DEEPGRAM_API_KEY else '❌ Missing'}\n"
        f"🔊 *gTTS:* Ready (free voice)\n\n"
        f"🔊 Voice replies: English, Russian\n"
        f"📝 Text replies: English, Russian, Uzbek\n\n"
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
    
    logger.info("🚀 Starting IELTS Anger Teacher Bot...")
    
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
    logger.info("✅ Bot is running on Telegram!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()