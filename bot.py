#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube (& more) Transcriber Bot
---------------------------------
Transcribes YouTube/TikTok/Twitter/etc. via yt-dlp + faster-whisper.
Uses python-telegram-bot v20+ with asyncio polling.

Usage:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in BOT_TOKEN
    python bot.py
"""

# ──────────────────────────────────────────────
#  SECTION 1: IMPORTS
# ──────────────────────────────────────────────
import asyncio
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, MessageHandler, filters, InlineQueryHandler, ContextTypes
import os
from fastapi import FastAPI, Request
import uvicorn

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Update,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    InlineQueryHandler,
    MessageHandler,
    filters,
)

load_dotenv()

# ──────────────────────────────────────────────
#  SECTION 2: LOGGING
# ──────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("transcriber_bot")

# ──────────────────────────────────────────────
#  SECTION 3: CONFIGURATION
# ──────────────────────────────────────────────
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "small")
MAX_FILE_BYTES: int = int(os.getenv("MAX_FILE_MB", "45")) * 1024 * 1024  # 45 MB safety margin
DATA_FILE: Path = Path(os.getenv("DATA_FILE", "user_data.json"))
OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "transcriptions"))
ALLOWED_USERS: list[int] = [
    int(x) for x in os.getenv("ALLOWED_USERS", "").split(",") if x.strip().isdigit()
]
ADMIN_IDS: list[int] = [
    int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()
]
MODEL_CHANGE_ADMIN_ONLY: bool = os.getenv("MODEL_CHANGE_ADMIN_ONLY", "false").lower() == "true"

MODELS: list[str] = ["tiny", "base", "small", "medium", "large-v2"]

# Full faster-whisper language list (ISO 639-1 codes)
LANGUAGES: dict[str, Optional[str]] = {
    "Auto-detect": None,
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar",
    "Armenian": "hy", "Assamese": "as", "Azerbaijani": "az", "Bashkir": "ba",
    "Basque": "eu", "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs",
    "Breton": "br", "Bulgarian": "bg", "Burmese": "my", "Castilian": "es",
    "Catalan": "ca", "Chinese": "zh", "Croatian": "hr", "Czech": "cs",
    "Danish": "da", "Dutch": "nl", "English": "en", "Estonian": "et",
    "Faroese": "fo", "Finnish": "fi", "French": "fr", "Galician": "gl",
    "Georgian": "ka", "German": "de", "Greek": "el", "Gujarati": "gu",
    "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he",
    "Hindi": "hi", "Hungarian": "hu", "Icelandic": "is", "Indonesian": "id",
    "Italian": "it", "Japanese": "ja", "Javanese": "jw", "Kannada": "kn",
    "Kazakh": "kk", "Khmer": "km", "Korean": "ko", "Lao": "lo",
    "Latin": "la", "Latvian": "lv", "Lingala": "ln", "Lithuanian": "lt",
    "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg",
    "Malay": "ms", "Malayalam": "ml", "Maltese": "mt", "Maori": "mi",
    "Marathi": "mr", "Moldavian": "ro", "Mongolian": "mn", "Nepali": "ne",
    "Norwegian": "no", "Nynorsk": "nn", "Occitan": "oc", "Pashto": "ps",
    "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa",
    "Romanian": "ro", "Russian": "ru", "Sanskrit": "sa", "Serbian": "sr",
    "Shona": "sn", "Sindhi": "sd", "Sinhala": "si", "Slovak": "sk",
    "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su",
    "Swahili": "sw", "Swedish": "sv", "Tagalog": "tl", "Tajik": "tg",
    "Tamil": "ta", "Tatar": "tt", "Telugu": "te", "Thai": "th",
    "Tibetan": "bo", "Turkish": "tr", "Turkmen": "tk", "Ukrainian": "uk",
    "Urdu": "ur", "Uzbek": "uz", "Valencian": "es", "Vietnamese": "vi",
    "Welsh": "cy", "Yiddish": "yi", "Yoruba": "yo",
}

# Deduplicated ordered list for display (remove code duplicates, keep first name)
_seen_codes: set = set()
LANG_LIST: list[tuple[str, Optional[str]]] = []
for _name, _code in LANGUAGES.items():
    key = _code or "auto"
    if key not in _seen_codes:
        _seen_codes.add(key)
        LANG_LIST.append((_name, _code))

# ──────────────────────────────────────────────
#  SECTION 4: PERSISTENT USER DATA
# ──────────────────────────────────────────────
_user_data: dict[int, dict] = {}


def _load_user_data() -> None:
    global _user_data
    if DATA_FILE.exists():
        try:
            _user_data = {int(k): v for k, v in json.loads(DATA_FILE.read_text()).items()}
            logger.info("Loaded user data for %d users.", len(_user_data))
        except Exception as exc:
            logger.warning("Could not load user data: %s", exc)


def _save_user_data() -> None:
    try:
        DATA_FILE.write_text(json.dumps({str(k): v for k, v in _user_data.items()}, indent=2))
    except Exception as exc:
        logger.warning("Could not save user data: %s", exc)


def get_user_pref(user_id: int, key: str, default=None):
    return _user_data.get(user_id, {}).get(key, default)


def set_user_pref(user_id: int, key: str, value) -> None:
    _user_data.setdefault(user_id, {})[key] = value
    _save_user_data()

# ──────────────────────────────────────────────
#  SECTION 5: WHISPER MODEL (loaded once)
# ──────────────────────────────────────────────
_whisper_model = None
_current_model_name: str = DEFAULT_MODEL
_model_lock = asyncio.Lock()
_executor = ThreadPoolExecutor(max_workers=2)


def _load_model_sync(model_size: str):
    """Load (or reload) the Whisper model synchronously."""
    from faster_whisper import WhisperModel
    try:
        m = WhisperModel(model_size, device="cuda", compute_type="float16")
        logger.info("Loaded model '%s' on CUDA.", model_size)
    except Exception:
        m = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info("Loaded model '%s' on CPU.", model_size)
    return m


async def ensure_model(model_size: Optional[str] = None) -> None:
    """Load model if not loaded, or reload if size changed."""
    global _whisper_model, _current_model_name
    target = model_size or _current_model_name
    async with _model_lock:
        if _whisper_model is None or target != _current_model_name:
            loop = asyncio.get_event_loop()
            _whisper_model = await loop.run_in_executor(_executor, _load_model_sync, target)
            _current_model_name = target

# ──────────────────────────────────────────────
#  SECTION 6: QUEUE SYSTEM
# ──────────────────────────────────────────────
_job_queue: deque = deque()
_active_jobs: dict[int, asyncio.Task] = {}   # user_id -> Task
_queue_lock = asyncio.Lock()


def queue_position(user_id: int) -> int:
    """1-based position in queue, 0 if not queued."""
    for i, uid in enumerate(_job_queue):
        if uid == user_id:
            return i + 1
    return 0

# ──────────────────────────────────────────────
#  SECTION 7: TRANSCRIPTION LOGIC (ported from GUI app)
# ──────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.strip().replace(" ", "_")
    return name[:120] or "transcript"


def _seconds_to_srt_time(s: float) -> str:
    ms = int((s % 1) * 1000)
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _segments_to_txt(segments) -> str:
    return " ".join(seg.text.strip() for seg in segments)


def _do_download_and_transcribe(
    url: str,
    language: Optional[str],
    model,
    progress_cb,        # callable(step: str) – runs in thread
) -> tuple[str, str]:
    """
    Exactly the same logic as TranscribeWorker._run() in the GUI app.
    Returns (title, txt_content).
    Raises RuntimeError on any failure.
    """
    import yt_dlp

    # ── Step 1: resolve video metadata ──────────────────────────────
    progress_cb("🔍 Fetching video info…")
    ydl_opts_info = {"quiet": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info = ydl.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "age" in msg.lower():
            raise RuntimeError("❌ Video is age-restricted and cannot be downloaded.")
        if "private" in msg.lower():
            raise RuntimeError("❌ Video is private or unavailable.")
        raise RuntimeError(f"❌ Could not fetch video info:\n<code>{msg[:300]}</code>")

    title: str = info.get("title", "video")

    # ── Step 2: download audio ───────────────────────────────────────
    progress_cb("⬇ Downloading audio…")
    tmpdir = tempfile.mkdtemp(prefix="tgbot_transcriber_")
    audio_path = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts_dl = {
        "format": "bestaudio/best",
        "outtmpl": audio_path,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"❌ Download failed:\n<code>{str(e)[:300]}</code>")

    audio_file: Optional[str] = None
    for f in Path(tmpdir).iterdir():
        if f.suffix in {".mp3", ".m4a", ".wav", ".ogg", ".opus", ".webm"}:
            audio_file = str(f)
            break

    if not audio_file:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("❌ Audio file not found after download. Is ffmpeg installed?")

    # ── Step 3: transcribe ───────────────────────────────────────────
    lang_label = language or "auto"
    progress_cb(f"✍ Transcribing ({lang_label})…")

    transcribe_kwargs: dict = {
        "beam_size": 5,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language:
        transcribe_kwargs["language"] = language

    segments_gen, _ = model.transcribe(audio_file, **transcribe_kwargs)
    segments = list(segments_gen)

    shutil.rmtree(tmpdir, ignore_errors=True)

    txt_content = _segments_to_txt(segments)
    return title, txt_content

# ──────────────────────────────────────────────
#  SECTION 8: ACCESS CONTROL HELPERS
# ──────────────────────────────────────────────

def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USERS:
        return True  # open to everyone
    return user_id in ALLOWED_USERS or user_id in ADMIN_IDS


def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_IDS

# ──────────────────────────────────────────────
#  SECTION 9: INLINE KEYBOARD BUILDERS
# ──────────────────────────────────────────────

def _lang_keyboard(page: int = 0) -> InlineKeyboardMarkup:
    """Paginated language selection keyboard (6 per page)."""
    per_page = 6
    total_pages = math.ceil(len(LANG_LIST) / per_page)
    start = page * per_page
    chunk = LANG_LIST[start: start + per_page]

    rows: list[list[InlineKeyboardButton]] = []
    for name, code in chunk:
        cb_data = f"lang:{code or 'auto'}"
        rows.append([InlineKeyboardButton(name, callback_data=cb_data)])

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀ Prev", callback_data=f"langpage:{page - 1}"))
    if page < total_pages - 1:
        nav.append(InlineKeyboardButton("Next ▶", callback_data=f"langpage:{page + 1}"))
    if nav:
        rows.append(nav)

    return InlineKeyboardMarkup(rows)


def _model_keyboard() -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(m, callback_data=f"model:{m}")] for m in MODELS]
    return InlineKeyboardMarkup(rows)

# ──────────────────────────────────────────────
#  SECTION 10: COMMAND HANDLERS
# ──────────────────────────────────────────────

async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not is_allowed(user.id):
        await update.message.reply_text("⛔ You are not authorised to use this bot.")
        return

    lang_name = get_user_pref(user.id, "lang_name", "Auto-detect")
    model_name = get_user_pref(user.id, "model", _current_model_name)

    await update.message.reply_text(
        f"👋 <b>Welcome, {user.first_name}!</b>\n\n"
        "I can transcribe YouTube videos (and TikTok, Twitter, and more) "
        "using OpenAI Whisper.\n\n"
        "<b>How to use:</b>\n"
        "1. Send me a video URL\n"
        "2. Pick a language (or keep Auto-detect)\n"
        "3. Wait while I download & transcribe\n"
        "4. Receive your .txt transcript\n\n"
        f"<b>Current settings:</b>\n"
        f"• Language: {lang_name}\n"
        f"• Model: {model_name}\n\n"
        "<b>Commands:</b>\n"
        "/language – Change transcription language\n"
        "/model – Change Whisper model size\n"
        "/cancel – Cancel ongoing transcription\n"
        "/help – Show this message",
        parse_mode=ParseMode.HTML,
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await cmd_start(update, ctx)


async def cmd_language(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_allowed(update.effective_user.id):
        return
    await update.message.reply_text(
        "🌐 Choose your transcription language:\n"
        "(Use the arrows to browse all languages)",
        reply_markup=_lang_keyboard(0),
    )


async def cmd_model(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        return
    if MODEL_CHANGE_ADMIN_ONLY and not is_admin(user_id):
        await update.message.reply_text("⛔ Only admins can change the model.")
        return

    await update.message.reply_text(
        "🤖 Choose Whisper model size:\n\n"
        "• <b>tiny</b> – Fastest, least accurate (~1 GB RAM)\n"
        "• <b>base</b> – Fast, decent accuracy\n"
        "• <b>small</b> – Good balance ✅ (default)\n"
        "• <b>medium</b> – Better accuracy, slower\n"
        "• <b>large-v2</b> – Best accuracy, slowest (~10 GB RAM)\n",
        reply_markup=_model_keyboard(),
        parse_mode=ParseMode.HTML,
    )


async def cmd_cancel(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    task = _active_jobs.get(user_id)
    if task and not task.done():
        task.cancel()
        await update.message.reply_text("🛑 Transcription cancelled.")
    else:
        # Check if queued
        if user_id in _job_queue:
            _job_queue.remove(user_id)
            await update.message.reply_text("🛑 Removed from queue.")
        else:
            await update.message.reply_text("ℹ️ No active transcription to cancel.")

# ──────────────────────────────────────────────
#  SECTION 11: CALLBACK QUERY HANDLERS
# ──────────────────────────────────────────────

async def handle_callback(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data: str = query.data

    # ── Language page navigation ─────────────────────────────────────
    if data.startswith("langpage:"):
        page = int(data.split(":")[1])
        try:
            await query.edit_message_reply_markup(_lang_keyboard(page))
        except BadRequest:
            pass
        return

    # ── Language selection ───────────────────────────────────────────
    if data.startswith("lang:"):
        code_raw = data.split(":", 1)[1]
        lang_code: Optional[str] = None if code_raw == "auto" else code_raw
        lang_name = next((n for n, c in LANG_LIST if (c or "auto") == code_raw), "Auto-detect")
        set_user_pref(user_id, "lang_code", lang_code)
        set_user_pref(user_id, "lang_name", lang_name)
        try:
            await query.edit_message_text(f"✅ Language set to <b>{lang_name}</b>.", parse_mode=ParseMode.HTML)
        except BadRequest:
            pass

        # If there's a pending URL for this user, start transcription now
        pending_url = ctx.user_data.pop("pending_url", None)
        if pending_url:
            await _enqueue_transcription(pending_url, user_id, query.message.chat_id, ctx)
        return

    # ── Model selection ──────────────────────────────────────────────
    if data.startswith("model:"):
        if MODEL_CHANGE_ADMIN_ONLY and not is_admin(user_id):
            await query.answer("⛔ Only admins can change the model.", show_alert=True)
            return
        model_size = data.split(":", 1)[1]
        set_user_pref(user_id, "model", model_size)
        try:
            await query.edit_message_text(
                f"✅ Model set to <b>{model_size}</b>.\n"
                "The model will be loaded on next transcription.",
                parse_mode=ParseMode.HTML,
            )
        except BadRequest:
            pass
        return

# ──────────────────────────────────────────────
#  SECTION 12: URL MESSAGE HANDLER
# ──────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+")


async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not is_allowed(user_id):
        await update.message.reply_text("⛔ You are not authorised to use this bot.")
        return

    text = (update.message.text or "").strip()
    match = _URL_RE.search(text)
    if not match:
        await update.message.reply_text(
            "ℹ️ Please send a video URL (YouTube, TikTok, Twitter, etc.) "
            "or use /help for instructions."
        )
        return

    url = match.group(0)

    # Save URL for after language selection if needed
    lang_code = get_user_pref(user_id, "lang_code", "UNSET")
    if lang_code == "UNSET":
        # First time: ask for language preference
        ctx.user_data["pending_url"] = url
        await update.message.reply_text(
            "🌐 First, choose your preferred language for transcription:\n"
            "(This will be remembered for future requests)",
            reply_markup=_lang_keyboard(0),
        )
        return

    await _enqueue_transcription(url, user_id, update.effective_chat.id, ctx)


async def _enqueue_transcription(
    url: str, user_id: int, chat_id: int, ctx: ContextTypes.DEFAULT_TYPE
) -> None:
    """Add job to queue and start processing if idle."""
    async with _queue_lock:
        if user_id in _active_jobs and not _active_jobs[user_id].done():
            await ctx.bot.send_message(chat_id, "⚠️ You already have an active transcription. Use /cancel first.")
            return
        if user_id in _job_queue:
            pos = queue_position(user_id)
            await ctx.bot.send_message(chat_id, f"⏳ You're already in queue at position {pos}.")
            return

        _job_queue.append(user_id)
        pos = queue_position(user_id)

    if pos > 1:
        await ctx.bot.send_message(
            chat_id,
            f"⏳ You're <b>#{pos}</b> in queue. I'll start your transcription soon!",
            parse_mode=ParseMode.HTML,
        )
    else:
        await ctx.bot.send_message(chat_id, "🔄 Starting transcription…")

    task = asyncio.create_task(
        _run_transcription(url, user_id, chat_id, ctx)
    )
    _active_jobs[user_id] = task


async def _run_transcription(
    url: str, user_id: int, chat_id: int, ctx: ContextTypes.DEFAULT_TYPE
) -> None:
    """Core transcription coroutine — waits for queue slot, then runs."""
    # Wait until we're first in queue
    while True:
        async with _queue_lock:
            if _job_queue and _job_queue[0] == user_id:
                break
        await asyncio.sleep(1.0)

    status_msg = None
    try:
        lang_code: Optional[str] = get_user_pref(user_id, "lang_code", None)
        model_size: str = get_user_pref(user_id, "model", _current_model_name)

        # ── Load model (once, or reload if size changed) ─────────────
        status_msg = await ctx.bot.send_message(chat_id, f"🤖 Loading model <b>{model_size}</b>…", parse_mode=ParseMode.HTML)
        await ensure_model(model_size)

        async def update_status(text: str) -> None:
            nonlocal status_msg
            try:
                status_msg = await status_msg.edit_text(text, parse_mode=ParseMode.HTML)
            except BadRequest:
                pass

        loop = asyncio.get_event_loop()

        # Thread-safe progress callback (schedules coroutine on event loop)
        def sync_progress(step: str) -> None:
            asyncio.run_coroutine_threadsafe(update_status(step), loop)

        # ── Run download + transcription in thread pool ───────────────
        await update_status("🔍 Fetching video info…")
        title, txt_content = await loop.run_in_executor(
            _executor,
            _do_download_and_transcribe,
            url, lang_code, _whisper_model, sync_progress,
        )

        await update_status("💾 Saving & sending file…")

        # ── Write .txt file ──────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_title = _sanitize_filename(title)
        txt_path = OUTPUT_DIR / f"{safe_title}.txt"
        txt_path.write_text(txt_content, encoding="utf-8")

        # ── Send to Telegram ─────────────────────────────────────────
        file_size = txt_path.stat().st_size
        preview = txt_content[:300].strip()
        if len(txt_content) > 300:
            preview += "…"

        await update_status(f"✅ Done! <b>{title}</b>")

        caption = (
            f"📄 <b>{title}</b>\n\n"
            f"<i>Preview:</i>\n{preview}"
        )

        if file_size <= MAX_FILE_BYTES:
            with open(txt_path, "rb") as f:
                await ctx.bot.send_document(
                    chat_id=chat_id,
                    document=f,
                    filename=f"{safe_title}.txt",
                    caption=caption[:1024],
                    parse_mode=ParseMode.HTML,
                )
        else:
            # Split into multiple messages (Telegram 4096 char limit per message)
            await ctx.bot.send_message(
                chat_id,
                f"📄 <b>{title}</b>\n\n"
                "File too large to send directly. Splitting into parts…",
                parse_mode=ParseMode.HTML,
            )
            chunk_size = 4000
            parts = [txt_content[i:i + chunk_size] for i in range(0, len(txt_content), chunk_size)]
            for idx, part in enumerate(parts, 1):
                await ctx.bot.send_message(
                    chat_id,
                    f"Part {idx}/{len(parts)}:\n\n{part}",
                )
                await asyncio.sleep(0.3)  # avoid flood limits

        # Clean up file after sending
        try:
            txt_path.unlink()
        except Exception:
            pass

    except asyncio.CancelledError:
        if status_msg:
            try:
                await status_msg.edit_text("🛑 Transcription cancelled.")
            except BadRequest:
                pass
    except RuntimeError as exc:
        err_msg = str(exc)
        logger.warning("Transcription error for user %d: %s", user_id, err_msg)
        if status_msg:
            try:
                await status_msg.edit_text(f"⚠️ {err_msg}", parse_mode=ParseMode.HTML)
            except BadRequest:
                await ctx.bot.send_message(chat_id, f"⚠️ {err_msg}", parse_mode=ParseMode.HTML)
        else:
            await ctx.bot.send_message(chat_id, f"⚠️ {err_msg}", parse_mode=ParseMode.HTML)
    except Exception as exc:
        logger.exception("Unexpected error for user %d", user_id)
        msg = f"❌ Unexpected error: <code>{str(exc)[:200]}</code>"
        if status_msg:
            try:
                await status_msg.edit_text(msg, parse_mode=ParseMode.HTML)
            except BadRequest:
                pass
        else:
            await ctx.bot.send_message(chat_id, msg, parse_mode=ParseMode.HTML)
    finally:
        # Remove from queue and active jobs
        async with _queue_lock:
            if user_id in _job_queue:
                _job_queue.remove(user_id)
            _active_jobs.pop(user_id, None)

        # Notify next in queue
        if _job_queue:
            next_user = _job_queue[0]
            next_task = _active_jobs.get(next_user)
            if next_task and not next_task.done():
                try:
                    # The next user's task is already running, it just needed to wait in queue
                    pass
                except Exception:
                    pass

# ──────────────────────────────────────────────
#  SECTION 13: INLINE MODE
# ──────────────────────────────────────────────

async def handle_inline(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Inline query: @botname <url> — triggers transcription via inline."""
    query = update.inline_query
    text = query.query.strip()
    if not _URL_RE.match(text):
        results = [
            InlineQueryResultArticle(
                id="help",
                title="Send a video URL to transcribe",
                input_message_content=InputTextMessageContent(
                    "Send me a YouTube/TikTok/Twitter URL in a direct message to transcribe it!"
                ),
                description="Type a URL after @botname",
            )
        ]
        await query.answer(results, cache_time=10)
        return

    results = [
        InlineQueryResultArticle(
            id="transcribe",
            title="📝 Transcribe this URL",
            input_message_content=InputTextMessageContent(text),
            description="Tap to send URL to bot for transcription",
        )
    ]
    await query.answer(results, cache_time=5)

# ──────────────────────────────────────────────
#  SECTION 14: ENTRY POINT
# ──────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        logger.critical("BOT_TOKEN is not set! Add it to your .env file.")
        raise SystemExit(1)

    if not shutil.which("ffmpeg"):
        logger.warning(
            "ffmpeg not found on PATH! Audio extraction will fail.\n"
            "Install: sudo apt install ffmpeg  |  brew install ffmpeg  |  winget install ffmpeg"
        )

    _load_user_data()

    logger.info("Building application…")
    app = Application.builder().token(BOT_TOKEN).build()

    # Commands
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("language", cmd_language))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(CommandHandler("cancel", cmd_cancel))

    # Callbacks (inline keyboards)
    app.add_handler(CallbackQueryHandler(handle_callback))

    # URL messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Inline mode
    app.add_handler(InlineQueryHandler(handle_inline))

    logger.info("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()