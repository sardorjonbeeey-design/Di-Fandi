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

# Load environment variables
load_dotenv()

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# ==================== LOGGING ====================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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
async def transcribe_openai(audio_path: str):
    """Transcribe using OpenAI Whisper API"""
    if not OPENAI_API_KEY:
        return None, None
    
    try:
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        
        with open(audio_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_path), f, "audio/ogg"),
                "model": (None, "whisper-1"),
                "language": (None, "auto")
            }
            response = requests.post(url, headers=headers, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            transcript = data.get("text", "")
            
            # Detect language from transcript
            lang = "en"
            cyrillic = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
            uzbek_latin = "қўғҳжўч"
            
            if any(char in cyrillic for char in transcript.lower()):
                lang = "ru"
            elif any(char in uzbek_latin for char in transcript.lower()):
                lang = "uz"
            
            return transcript, lang
        else:
            logger.error(f"OpenAI Whisper error: {response.status_code} - {response.text}")
            return None, None
    except Exception as e:
        logger.error(f"OpenAI Whisper error: {e}")
        return None, None

async def transcribe_deepgram(audio_path: str):
    """Transcribe using Deepgram"""
    if not DEEPGRAM_API_KEY:
        return None, None
    
    try:
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/ogg"
        }
        params = {
            "model": "nova-2",
            "language": "multi",
            "detect_language": True
        }
        
        with open(audio_path, "rb") as f:
            response = requests.post(url, headers=headers, params=params, data=f, timeout=60)
        
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
    """Try OpenAI Whisper first, then fallback to Deepgram"""
    # Try OpenAI Whisper first (best quality)
    result = await transcribe_openai(audio_path)
    if result and result[0]:
        logger.info(f"✅ Transcribed with OpenAI Whisper")
        return result
    
    # Fallback to Deepgram
    result = await transcribe_deepgram(audio_path)
    if result and result[0]:
        logger.info(f"✅ Transcribed with Deepgram")
        return result
    
    return None, None

# ==================== TTS FUNCTIONS ====================
async def generate_tts_google(text: str, lang: str):
    """Generate speech using Google TTS"""
    if not GOOGLE_TTS_API_KEY:
        return None
    
    if lang not in ["en", "ru"]:
        return None
    
    try:
        lang_codes = {"en": "en-US", "ru": "ru-RU"}
        voice_names = {
            "en": "en-US-Neural2-F",
            "ru": "ru-RU-Wavenet-A"
        }
        
        # Clean text for TTS
        clean_text = text.replace("*", "").replace("_", "").replace("[", "").replace("]", "")
        clean_text = clean_text[:300]
        
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
        payload = {
            "input": {"text": clean_text},
            "voice": {"languageCode": lang_codes[lang], "name": voice_names[lang]},
            "audioConfig": {"audioEncoding": "OGG_OPUS", "speakingRate": 0.95}
        }
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            audio_content = response.json()["audioContent"]
            output_path = f"/tmp/tts_{datetime.now().timestamp()}.ogg"
            
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(audio_content))
            
            return output_path
        else:
            logger.error(f"Google TTS error: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Google TTS error: {e}")
        return None

async def text_to_speech(text: str, lang: str):
    """Main TTS function"""
    if not GOOGLE_TTS_API_KEY:
        return None
    
    return await generate_tts_google(text, lang)

# ==================== LLM FUNCTIONS ====================
async def get_llm_response_gemini(text: str, lang: str = "en"):
    """Get response from Gemini"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        full_prompt = f"""{IELTS_ANGER_PROMPT}

User language: {lang}
User said: {text}

Respond in {lang} language with the same angry, savage energy. Keep it short (2-3 sentences)."""
        
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
    # Try Gemini first
    result = await get_llm_response_gemini(text, lang)
    if result:
        return result
    
    # Fallback responses
    fallbacks = [
        "*[slams face on desk]* My brain cells are dying. WHAT DO YOU WANT?!",
        "*[screams into microphone]* Is this the best you can do?!",
        "*[aggressive sigh]* I'm losing IQ points just listening to you!",
        "You call that English?! My dead grandmother speaks better!",
        "*[throws laptop out window]* I QUIT! Find another examiner, you potato!"
    ]
    return random.choice(fallbacks)

# ==================== TELEGRAM BOT HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    await update.message.reply_text(
        "🎤 *IELTS ANGER TEACHER BOT*\n\n"
        "Send me a *voice message* in English, Russian, or Uzbek.\n"
        "I'll criticize your speaking skills mercilessly!\n\n"
        "🤖 *Powered by:*\n"
        "• Gemini AI (brain)\n"
        "• OpenAI Whisper (transcription)\n"
        "• Deepgram (backup STT)\n"
        "• Google TTS (voice replies)\n\n"
        "⚠️ *Voice replies:* English & Russian only\n"
        "📝 *Uzbek:* Text replies only\n\n"
        "🔊 *Commands:*\n"
        "/start - Welcome\n"
        "/health - Check status\n"
        "/help - Get help\n\n"
        "*WARNING:* My patience is ZERO!",
        parse_mode="Markdown"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    await update.message.reply_text(
        "🎯 *How to use:*\n\n"
        "1. Send a voice message\n"
        "2. I'll transcribe (OpenAI Whisper/Deepgram)\n"
        "3. Gemini generates a savage response\n"
        "4. I reply with text + voice (English/Russian)\n\n"
        "💡 *Tips:*\n"
        "- Speak clearly or get mocked\n"
        "- Use advanced vocabulary\n"
        "- Don't use filler words\n\n"
        "*Good luck, you'll need it!*",
        parse_mode="Markdown"
    )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    voice = update.message.voice
    
    if not voice:
        await update.message.reply_text("*[aggressive sigh]* Send a VOICE message, you potato!")
        return
    
    # Send initial reaction
    await update.message.reply_text("*[cracks knuckles]* Let's see what trash you sent me...")
    
    try:
        # Download voice file
        file = await update.message.voice.get_file()
        
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_file:
            await file.download_to_drive(tmp_file.name)
            audio_path = tmp_file.name
        
        # Transcribe with OpenAI Whisper + Deepgram fallback
        transcript, detected_lang = await transcribe_audio(audio_path)
        
        # Clean up audio file
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
            f"🌍 *Detected:* {detected_lang.upper()}",
            parse_mode="Markdown"
        )
        
        # Get response from Gemini
        response_text = await get_llm_response(transcript, detected_lang)
        
        # Send text response
        await update.message.reply_text(response_text, parse_mode="Markdown")
        
        # Try voice reply (English/Russian only)
        if detected_lang in ["en", "ru"]:
            await update.message.reply_text("🔊 *Generating voice...*", parse_mode="Markdown")
            audio_response = await text_to_speech(response_text, detected_lang)
            
            if audio_response and os.path.exists(audio_response):
                try:
                    with open(audio_response, "rb") as audio_file:
                        await update.message.reply_voice(voice=audio_file)
                    os.unlink(audio_response)
                except Exception as e:
                    logger.error(f"Voice reply error: {e}")
            else:
                await update.message.reply_text("*[TTS failed]* Just read my text, you potato!")
        else:
            await update.message.reply_text(
                "📝 *Uzbek voice not supported yet.*\n"
                "Read the text and practice!",
                parse_mode="Markdown"
            )
        
    except Exception as e:
        logger.error(f"Voice handling error: {e}")
        await update.message.reply_text("*[system error]* YOU BROKE ME! Congratulations, you absolute clown!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    text = update.message.text
    
    # Language detection
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
    status = []
    status.append("✅ OpenAI" if OPENAI_API_KEY else "❌ OpenAI")
    status.append("✅ Gemini" if GEMINI_API_KEY else "❌ Gemini")
    status.append("✅ Deepgram" if DEEPGRAM_API_KEY else "❌ Deepgram")
    status.append("✅ Google TTS" if GOOGLE_TTS_API_KEY else "❌ Google TTS")
    
    await update.message.reply_text(
        f"🤖 *Bot Status:* Alive!\n\n"
        f"{chr(10).join(status)}\n\n"
        f"📝 *Supported:* English, Russian, Uzbek\n"
        f"🔊 *Voice:* English, Russian\n\n"
        "*Now leave me alone!*",
        parse_mode="Markdown"
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    logger.error(f"Update {update} caused error {context.error}")

# ==================== MAIN ====================
def main():
    if not TELEGRAM_TOKEN:
        logger.error("❌ TELEGRAM_TOKEN not set!")
        return
    
    logger.info("🚀 Starting IELTS Anger Teacher Bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    
    # Start bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()