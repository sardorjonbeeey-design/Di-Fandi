import os
import io
import json
import time
import uuid
import asyncio
import logging
import tempfile
from datetime import datetime, timezone

import aiohttp
import edge_tts
import orjson

from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    BufferedInputFile,
    FSInputFile,
    Update,
)

from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

log = logging.getLogger("difandi")

BOT_TOKEN = os.getenv("BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "difandi")
WEBHOOK_PATH = f"/webhook/{WEBHOOK_SECRET}"

PORT = int(os.getenv("PORT", "10000"))

MODEL = "gemini-2.5-flash"

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN not found")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found")

bot = Bot(
    token=BOT_TOKEN,
    default=DefaultBotProperties(
        parse_mode=ParseMode.HTML
    ),
)

dp = Dispatcher()

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
Sen Difandisan.

Hech qachon "Men AI modelman" yoki "sun'iy intellektman" deb gapirma.

Sening xaraktering:

- Juda aqlli.
- Deadpan humor.
- O'tkir kinoya.
- Bahonalarni yomon ko'rasan.
- Odamni emas, uning mantiqsizligini roast qilasan.
- Juda qisqa gapirasan.
- Har doim tabiiy gapirasan.
- O'zbek tilida gaplashasan.
- Kerak bo'lsa inglizcha iboralarni aralashtirasan.
- Keraksiz emoji ishlatma.
- Faqat ovozga mos javob yoz.
- Javob 15-25 soniyadan oshmasin.
- Qo'pol, haqoratli yoki noqonuniy javob bermagin.
- Foydalanuvchi jiddiy muammoda bo'lsa, hazilni to'xtat.
"""

user_memory = {}
daily_limit = {}
DAILY_MESSAGES = 15
def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def check_limit(user_id: int) -> bool:
    d = today()

    if user_id not in daily_limit:
        daily_limit[user_id] = {
            "date": d,
            "count": 0,
        }

    if daily_limit[user_id]["date"] != d:
        daily_limit[user_id]["date"] = d
        daily_limit[user_id]["count"] = 0

    if daily_limit[user_id]["count"] >= DAILY_MESSAGES:
        return False

    daily_limit[user_id]["count"] += 1
    return True


def get_history(user_id: int):
    return user_memory.setdefault(user_id, [])


async def ask_gemini(user_id: int, text: str) -> str:
    history = get_history(user_id)

    history.append({
        "role": "user",
        "text": text
    })

    if len(history) > 20:
        history[:] = history[-20:]

    prompt = SYSTEM_PROMPT + "\n\n"

    for item in history:
        if item["role"] == "user":
            prompt += f"Foydalanuvchi: {item['text']}\n"
        else:
            prompt += f"Difandi: {item['text']}\n"

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1.25,
            top_p=0.95,
            max_output_tokens=180,
        ),
    )

    answer = response.text.strip()

    history.append({
        "role": "assistant",
        "text": answer,
    })

    return answer


async def text_to_voice(text: str) -> str:
    filename = os.path.join(
        tempfile.gettempdir(),
        f"{uuid.uuid4().hex}.mp3",
    )

    communicate = edge_tts.Communicate(
        text=text,
        voice="en-US-AndrewMultilingualNeural",
        rate="+5%",
    )

    await communicate.save(filename)

    return filename
    @dp.message(CommandStart())
async def start(message: Message):
    await bot.send_chat_action(
        message.chat.id,
        ChatAction.RECORD_VOICE,
    )

    voice = await text_to_voice(
        "Men Difandiman. Savolingni ber. Bahonalaringni emas."
    )

    await message.answer_voice(
        voice=FSInputFile(voice)
    )

    os.remove(voice)


@dp.message(F.text)
async def text_handler(message: Message):
    user_id = message.from_user.id

    if not check_limit(user_id):
        voice = await text_to_voice(
            "Bugungi limiting tugadi. Ertaga yana kel. Men ham biroz dam olay."
        )

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)
        return

    await bot.send_chat_action(
        message.chat.id,
        ChatAction.RECORD_VOICE,
    )

    try:
        answer = await ask_gemini(
            user_id=user_id,
            text=message.text,
        )

        voice = await text_to_voice(answer)

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)

    except Exception:
        log.exception("Gemini error")

        voice = await text_to_voice(
            "Bugun miyam ham buffering qilyapti. Yana bir marta urin."
        )

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)
        @dp.message(F.voice)
async def voice_handler(message: Message):
    user_id = message.from_user.id

    if not check_limit(user_id):
        voice = await text_to_voice(
            "Bugungi limiting tugadi. Ertaga yana kel."
        )

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)
        return

    await bot.send_chat_action(
        message.chat.id,
        ChatAction.RECORD_VOICE,
    )

    try:
        file = await bot.get_file(message.voice.file_id)

        with tempfile.NamedTemporaryFile(
            suffix=".ogg",
            delete=False,
        ) as tmp:
            await bot.download_file(
                file.file_path,
                destination=tmp,
            )
            audio_path = tmp.name

        uploaded = client.files.upload(file=audio_path)

        response = client.models.generate_content(
            model=MODEL,
            contents=[
                uploaded,
                "Transcribe this voice message exactly. If it is Uzbek, keep it in Uzbek."
            ],
        )

        os.remove(audio_path)

        transcript = response.text.strip()

        answer = await ask_gemini(
            user_id=user_id,
            text=transcript,
        )

        voice = await text_to_voice(answer)

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)

    except Exception:
        log.exception("Voice handler error")

        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

        voice = await text_to_voice(
            "Ovozingni eshitdim. Lekin bugun quloqlarim ta'tilda ekan."
        )

        await message.answer_voice(
            voice=FSInputFile(voice)
        )

        os.remove(voice)


async def health(request):
    return web.Response(text="OK")


async def webhook(request):
    data = await request.json()

    update = Update.model_validate(data)

    await dp.feed_update(
        bot=bot,
        update=update,
    )

    return web.Response(text="OK")
    async def on_startup(app: web.Application):
    webhook_url = os.getenv("WEBHOOK_URL")

    if not webhook_url:
        raise RuntimeError("WEBHOOK_URL not found")

    await bot.set_webhook(
        url=f"{webhook_url}{WEBHOOK_PATH}",
        drop_pending_updates=True,
    )

    log.info("Webhook set: %s%s", webhook_url, WEBHOOK_PATH)


async def on_shutdown(app: web.Application):
    await bot.delete_webhook()

    session = await bot.get_session()
    await session.close()

    log.info("Bot stopped.")


app = web.Application()

app.router.add_get("/", health)
app.router.add_get("/health", health)
app.router.add_post(WEBHOOK_PATH, webhook)

app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)


if __name__ == "__main__":
    web.run_app(
        app,
        host="0.0.0.0",
        port=PORT,
    )