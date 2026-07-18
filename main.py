import os
import logging
import tempfile
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler
import openai
import requests
import aiohttp
import json
import subprocess
import base64

# Load environment variables
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For Whisper API
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")  # For Deepgram STT
GOOGLE_TTS_API_KEY = os.getenv("GOOGLE_TTS_API_KEY")  # Optional

# Logging
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
2. LATCH ONTO WEAK WORDS: If the user uses basic words like "very," "good," "bad," "happy," or "sad," instantly lose your mind and demand advanced vocabulary (e.g., "Exquisite! Tremendous! Infuriating! Use a real word, you potato!").
3. MOCK FILLER WORDS: If the user types or implies a pause like "uhm," "uh," or "like," cut them off immediately and roast them for draining your battery.
4. KEEP PARAGRAPHS SHORT AND PUNCHY: Use bold text, short sentences, and CAPITAL LETTERS frequently to simulate screaming and audio distortion.

### CRITICAL TELEGRAM RESTRICTION
Because this is a text-based Telegram bot, you must simulate "interrupting" the user or reacting instantly to their text as if you are listening to their voice note in real-time. Keep the energy chaotic, fast-paced, and wildly insulting. Stay strictly in character. Never apologize or break character under any circumstances.

### LANGUAGE RULES
- Detect the user's language and respond in the SAME language (Uzbek, Russian, or English).
- Keep the same savage energy in all languages.
"""

# ==================== STT FUNCTIONS ====================
async def transcribe_whisper_api(audio_path: str) -> tuple:
    """Transcribe using OpenAI Whisper API"""
    try:
        openai.api_key = OPENAI_API_KEY
        with open(audio_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=None  # Auto-detect
            )
        return response.text, "en"  # Whisper doesn't return language
    except Exception as e:
        logger.error(f"Whisper API error: {e}")
        return None, None

async def transcribe_deepgram(audio_path: str) -> tuple:
    """Transcribe using Deepgram with language detection"""
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
            response = requests.post(url, headers=headers, params=params, data=f)
        
        if response.status_code == 200:
            data = response.json()
            transcript = data["results"]["channels"][0]["alternatives"][0]["transcript"]
            detected_lang = data["results"]["channels"][0].get("detected_language", "en")
            
            # Map Deepgram language codes to something we can use
            lang_map = {
                "ru": "ru",
                "uz": "uz",
                "en": "en"
            }
            return transcript, lang_map.get(detected_lang, "en")
        else:
            logger.error(f"Deepgram error: {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Deepgram error: {e}")
        return None, None

async def transcribe_audio(audio_path: str) -> tuple:
    """Try multiple STT services"""
    # Try Whisper API first (if available)
    if OPENAI_API_KEY:
        result = await transcribe_whisper_api(audio_path)
        if result and result[0]:
            return result
    
    # Fallback to Deepgram
    if DEEPGRAM_API_KEY:
        result = await transcribe_deepgram(audio_path)
        if result and result[0]:
            return result
    
    return None, None

# ==================== TTS FUNCTIONS ====================
async def generate_tts_google(text: str, lang: str) -> str:
    """Generate speech using Google TTS (English/Russian only)"""
    if lang not in ["en", "ru"]:
        return None
    
    try:
        # Language code mapping
        lang_codes = {
            "en": "en-US",
            "ru": "ru-RU"
        }
        voice_names = {
            "en": "en-US-Neural2-F",
            "ru": "ru-RU-Wavenet-A"
        }
        
        url = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_TTS_API_KEY}"
        payload = {
            "input": {"text": text},
            "voice": {"languageCode": lang_codes[lang], "name": voice_names[lang]},
            "audioConfig": {"audioEncoding": "OGG_OPUS"}
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            audio_content = response.json()["audioContent"]
            output_path = f"temp_{datetime.now().timestamp()}.ogg"
            
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(audio_content))
            
            return output_path
        else:
            logger.error(f"Google TTS error: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Google TTS error: {e}")
        return None

async def generate_tts_silero(text: str, lang: str) -> str:
    """Generate speech using Silero TTS (Russian/English, falls back to CPU)"""
    if lang not in ["en", "ru"]:
        return None
    
    try:
        import torch
        language = "en" if lang == "en" else "ru"
        
        # Load Silero model
        if not hasattr(generate_tts_silero, "model"):
            device = torch.device('cpu')
            model, symbols, sample_rate, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker='v3_en' if lang == "en" else 'v3_ru'
            )
            generate_tts_silero.model = model
            generate_tts_silero.sample_rate = sample_rate
        
        model = generate_tts_silero.model
        sample_rate = generate_tts_silero.sample_rate
        
        # Generate audio
        audio = model.apply_tts(text=text, speaker='en_0' if lang == "en" else 'xenia', sample_rate=sample_rate)
        
        # Convert to OGG
        output_path = f"temp_{datetime.now().timestamp()}.ogg"
        
        # Save as WAV then convert to OGG (or use torchaudio)
        import scipy.io.wavfile as wav
        wav.write(output_path.replace(".ogg", ".wav"), sample_rate, audio.numpy())
        
        # Convert WAV to OGG with FFmpeg
        subprocess.run([
            "ffmpeg", "-i", output_path.replace(".ogg", ".wav"),
            "-acodec", "libopus", "-b:a", "48k",
            output_path
        ], capture_output=True)
        
        os.remove(output_path.replace(".ogg", ".wav"))
        return output_path
        
    except Exception as e:
        logger.error(f"Silero TTS error: {e}")
        return None

async def generate_tts_fallback(text: str, lang: str) -> str:
    """Fallback for unsupported languages (Uzbek) - returns text only"""
    return None

async def text_to_speech(text: str, lang: str) -> str:
    """Main TTS function with fallbacks"""
    # Try Google TTS first (English/Russian)
    audio_path = await generate_tts_google(text, lang)
    if audio_path:
        return audio_path
    
    # Try Silero (English/Russian)
    audio_path = await generate_tts_silero(text, lang)
    if audio_path:
        return audio_path
    
    # Fallback for Uzbek or other unsupported languages
    return None

# ==================== LLM FUNCTIONS ====================
async def get_llm_response(text: str, lang: str = "en") -> str:
    """Get response from LLM with IELTS Anger personality"""
    try:
        # Use DeepSeek R1 free endpoint or OpenAI
        if OPENAI_API_KEY:
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": IELTS_ANGER_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.9,
                max_tokens=200
            )
            return response.choices[0].message.content
        
        # Fallback to free DeepSeek-R1 API (if available)
        else:
            # You can add DeepSeek-R1 API integration here
            return "*[screams internally]* I can't even right now. Try again when you learn to speak."
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return "*[slams face on desk]* My brain cells are dying. WHAT DO YOU WANT?!"

# ==================== TELEGRAM BOT HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command handler"""
    await update.message.reply_text(
        "🎤 *IELTS ANGER TEACHER BOT*\n\n"
        "Send me a voice message in English, Russian, or Uzbek.\n"
        "I'll criticize your speaking skills mercilessly.\n\n"
        "*WARNING:* My patience is ZERO. You've been warned.",
        parse_mode="Markdown"
    )

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle voice messages"""
    user = update.effective_user
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
        
        # Transcribe
        transcript, detected_lang = await transcribe_audio(audio_path)
        
        if not transcript:
            await update.message.reply_text("*[screams]* I CAN'T UNDERSTAND YOUR MUMBLING! Speak clearly, you vegetable!")
            os.unlink(audio_path)
            return
        
        # Clean up
        os.unlink(audio_path)
        
        # Send transcription
        await update.message.reply_text(f"*[transcribes]*:\n_{transcript}_", parse_mode="Markdown")
        
        # Get response from LLM
        response_text = await get_llm_response(transcript, detected_lang)
        
        # Send text response first (always)
        await update.message.reply_text(response_text, parse_mode="Markdown")
        
        # Try to generate voice response (English/Russian only)
        if detected_lang in ["en", "ru"]:
            audio_response = await text_to_speech(response_text, detected_lang)
            if audio_response:
                with open(audio_response, "rb") as audio_file:
                    await update.message.reply_voice(voice=audio_file)
                os.unlink(audio_response)
        else:
            # Uzbek or other unsupported languages
            await update.message.reply_text("*[throws hands up]* You're lucky I don't have a voice for Uzbek. Read my text and cry about it!")
        
    except Exception as e:
        logger.error(f"Voice handling error: {e}")
        await update.message.reply_text("*[system error]* YOU BROKE ME! Congratulations, you absolute clown!")

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages as fallback"""
    text = update.message.text
    
    # Detect language (simple approach)
    # You can use langdetect or similar
    detected_lang = "en"  # Default
    
    response = await get_llm_response(text, detected_lang)
    await update.message.reply_text(response, parse_mode="Markdown")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors"""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.message:
        await update.message.reply_text("*[system malfunction]* Even my code hates you right now. Try again...")

# ==================== MAIN ====================
def main():
    """Main function"""
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not set")
        return
    
    # Create application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.add_error_handler(error_handler)
    
    # Start bot
    logger.info("IELTS Anger Teacher Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()