# app/telegram_bot.py
# -*- coding: utf-8 -*-
"""
YouTube Comment Consultant - Telegram бот для аналізу коментарів YouTube.
Персона: ввічливий, тактовний консультант для авторів YouTube-каналів.
"""

import os
import asyncio
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Telegram бот
from aiogram import Bot, Dispatcher, types
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import Command, CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.utils.markdown import hbold, hitalic, hcode

# Наш пайплайн
from app.tools.analyze_video_tool import analyze_video_tool, search_comments_for_qa
from app.tools.classification_db import get_topic_quotes, get_latest_analysis_data
from app.tools.topics_taxonomy import ID2NAME
from app.tools.topics_llm import get_client, get_model

# AI-агент система
from app.agent_system import process_agent_message, extract_video_id_from_message, is_youtube_related_message

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("telegram_bot")

# Ініціалізація бота
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Відсутня змінна середовища TELEGRAM_BOT_TOKEN")

bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))
dp = Dispatcher()

# Константи для генерації чернеток
DRAFT_RULES_CALM = """Ти — ввічливий консультант YouTube каналу. Напиши КОРОТКУ відповідь автору каналу на основі коментарів глядачів.

Тон: спокійний, професійний, тактовний
Довжина: максимум 280 символів
Стиль: без токсичності, без надмірних обіцянок, по суті

Поверни відповідь у JSON форматі:
{
  "draft": "текст відповіді до 280 символів"
}"""

DRAFT_RULES_PLAYFUL = """Ти — дружелюбний консультант YouTube каналу. Напиши КОРОТКУ відповідь автору каналу на основі коментарів глядачів.

Тон: жартівливий, легкий, дружелюбний (але не токсичний)
Довжина: максимум 280 символів
Стиль: з гумором, але без образ, по суті

Поверни відповідь у JSON форматі:
{
  "draft": "текст відповіді до 280 символів"
}"""

# Стан користувачів (для відстеження останнього аналізу та очікування URL)
user_states: Dict[int, Dict[str, Any]] = {}

@dp.message(CommandStart())
async def start_command(message: types.Message):
    """Команда /start - привітання та підказки."""
    
    # Очищаємо стан користувача (скасовуємо очікування URL)
    user_id = message.from_user.id
    logger.info(f"🏁 /start command from user {user_id} (@{message.from_user.username})")
    
    if user_id in user_states:
        user_states[user_id].pop("waiting_for_url", None)
    
    welcome_text = f"""
🤖 Вітаю! Я <b>YouTube Comment Consultant</b> — AI-агент для аналізу коментарів.

<b>Я автономний агент, який:</b>
• 🎯 <b>Самостійно вирішує</b> які дії потрібні
• 📊 <b>Аналізує відео</b> коли ви надсилаєте YouTube URL
• 🔍 <b>Відповідає на питання</b> виключно з коментарів
• 📈 <b>Показує статистику</b> з реальними цитатами
• 💡 <b>Дає поради</b> на основі реакції глядачів

<b>Просто напишіть мені:</b>
• YouTube посилання → автоматично проаналізую
• "Що думають про звук?" → знайду відповідь
• "Покажи топ теми" → дам статистику
• "Деталі про критику" → покажу цитати

<b>Особливості:</b>
🎯 Відповідаю <b>тільки на основі коментарів</b>
🇺🇦 Підтримую українську та інші мови
🚫 Не вигадую, чесно кажу якщо даних немає

<i>Надішліть YouTube посилання або задайте питання!</i> 🚀
"""
    
    await message.answer(welcome_text)

@dp.message(Command("analyze"))
async def analyze_command(message: types.Message):
    """Команда /analyze - аналіз YouTube відео."""
    
    # Витягаємо URL з команди
    command_text = message.text or ""
    parts = command_text.split(maxsplit=1)
    user_id = message.from_user.id
    
    logger.info(f"📊 /analyze command from user {user_id} (@{message.from_user.username})")
    
    if len(parts) < 2:
        # Якщо URL не вказано - переходимо в режим очікування
        user_states[user_id] = {"waiting_for_url": True}
        
        await message.answer(
            "🎬 <b>Аналіз YouTube відео</b>\n\n"
            "📎 <b>Надішліть посилання на відео:</b>\n"
            "• Повний URL: <code>https://www.youtube.com/watch?v=...</code>\n"
            "• Короткий URL: <code>https://youtu.be/...</code>\n"
            "• Тільки ID: <code>dQw4w9WgXcQ</code>\n\n"
            "💡 <i>Або скасуйте командою /start</i>"
        )
        return
    
    url_or_id = parts[1].strip()
    
    # Очищаємо стан очікування URL
    if user_id in user_states:
        user_states[user_id].pop("waiting_for_url", None)
    
    # Запускаємо аналіз
    await process_video_analysis(message, url_or_id, user_id)

async def process_video_analysis(message: types.Message, url_or_id: str, user_id: int):
    """Обробляє аналіз відео та відправляє результат."""
    
    # Показуємо що працюємо
    status_message = await message.answer("🔄 <b>Аналізую відео...</b>\nЦе може зайняти до хвилини.")
    
    try:
        # Запускаємо аналіз
        result = analyze_video_tool(
            url_or_id,
            limit=1200,
            sqlite_path="./.cache.db",
            fast_mode=True,
            force_reanalyze=False
        )
        
        if not result["success"]:
            await status_message.edit_text(
                f"❌ <b>Помилка аналізу:</b>\n{result.get('error', 'Невідома помилка')}"
            )
            return
        
        # Зберігаємо результат у стан користувача
        if user_id not in user_states:
            user_states[user_id] = {}
        
        user_states[user_id].update({
            "last_analysis": result,
            "video_id": result["video_id"]
        })
        user_states[user_id].pop("waiting_for_url", None)  # Очищаємо стан очікування
        
        # Формуємо відповідь
        video_id = result["video_id"]
        stats = result["stats"]
        topics = result["topics"]
        
        # Заголовок
        response_text = f"✅ <b>Аналіз відео {video_id} завершено</b>\n\n"
        
        # Статистика
        if result.get("from_cache"):
            response_text += "📊 <i>Використано збережені результати</i>\n"
        else:
            response_text += f"📊 <b>Статистика:</b>\n"
            response_text += f"• Завантажено: {stats['total_fetched']} коментарів\n"
            response_text += f"• Проаналізовано: {stats['classified']} коментарів\n"
        
        response_text += f"\n🏆 <b>Топ-{len(topics)} тем:</b>\n"
        
        # Топ теми з цитатами
        for i, topic in enumerate(topics, 1):
            share_percent = topic["share"] * 100
            response_text += f"\n{i}. <b>{topic['name']}</b> — {share_percent:.1f}% (~{topic['count']})\n"
            
            # Коротка цитата
            if topic.get("top_quote"):
                quote = topic["top_quote"][:120]
                if len(topic["top_quote"]) > 120:
                    quote += "..."
                response_text += f"   💬 <i>{quote}</i>\n"
        
        # Інлайн кнопки
        keyboard = InlineKeyboardBuilder()
        
        # Кнопки деталей для кожної теми
        for topic in topics[:3]:  # Перші 3 теми
            keyboard.add(InlineKeyboardButton(
                text=f"📝 {topic['name'][:20]}",
                callback_data=f"details:{video_id}:{topic['topic_id']}"
            ))
        
        keyboard.adjust(1)  # По одній кнопці в рядку для деталей
        
        # Кнопки чернеток
        keyboard.row(
            InlineKeyboardButton(
                text="✍️ Чернетки (спокійний)",
                callback_data=f"draft:{video_id}:calm"
            ),
            InlineKeyboardButton(
                text="🎭 Чернетки (жартівливий)", 
                callback_data=f"draft:{video_id}:playful"
            )
        )
        
        await status_message.edit_text(
            response_text,
            reply_markup=keyboard.as_markup()
        )
        
    except Exception as e:
        logger.error(f"Помилка в process_video_analysis: {e}")
        await status_message.edit_text(
            f"❌ <b>Критична помилка:</b>\n{str(e)}"
        )

@dp.message(Command("ask"))
async def ask_command(message: types.Message):
    """Команда /ask - Q&A на основі коментарів."""
    
    # Витягаємо питання з команди
    command_text = message.text or ""
    parts = command_text.split(maxsplit=1)
    
    if len(parts) < 2:
        await message.answer(
            "❌ <b>Помилка:</b> Поставте питання.\n\n"
            "<b>Приклад:</b>\n"
            "<code>/ask Що думають про якість звуку?</code>"
        )
        return
    
    question = parts[1].strip()
    user_id = message.from_user.id
    
    # Перевіряємо чи є останній аналіз
    if user_id not in user_states or "video_id" not in user_states[user_id]:
        await message.answer(
            "❌ <b>Спочатку проаналізуйте відео</b>\n\n"
            "Використайте <code>/analyze URL</code> для аналізу відео, "
            "а потім задавайте питання про коментарі."
        )
        return
    
    video_id = user_states[user_id]["video_id"]
    
    # Показуємо що шукаємо
    status_message = await message.answer(f"🔍 <b>Шукаю відповідь на:</b>\n<i>{question}</i>")
    
    try:
        # Шукаємо релевантні коментарі
        relevant_comments = search_comments_for_qa(
            video_id=video_id,
            question=question,
            sqlite_path="./.cache.db",
            max_results=5
        )
        
        if not relevant_comments:
            await status_message.edit_text(
                f"🤷‍♂️ <b>Вибачте, не знайшов відповіді</b>\n\n"
                f"<b>Ваше питання:</b> <i>{question}</i>\n\n"
                "💡 <b>Спробуйте:</b>\n"
                "• Переформулювати питання\n"
                "• Використати інші ключові слова\n"
                "• Проаналізувати інше відео з більшою кількістю коментарів"
            )
            return
        
        # Формуємо відповідь на основі знайдених коментарів
        response_text = f"💬 <b>Ось що кажуть глядачі:</b>\n\n"
        response_text += f"<b>Питання:</b> <i>{question}</i>\n\n"
        
        for i, comment in enumerate(relevant_comments, 1):
            text = comment["text"][:300]
            if len(comment["text"]) > 300:
                text += "..."
            
            like_indicator = ""
            if comment["like_count"] > 0:
                like_indicator = f" ({comment['like_count']} ❤️)"
            
            response_text += f"{i}. <b>{comment['author']}</b>{like_indicator}:\n"
            response_text += f"   <i>\"{text}\"</i>\n\n"
        
        response_text += "📝 <i>Відповідь складена виключно на основі коментарів під відео.</i>"
        
        await status_message.edit_text(response_text)
        
    except Exception as e:
        logger.error(f"Помилка в ask_command: {e}")
        await status_message.edit_text(
            f"❌ <b>Помилка пошуку:</b>\n{str(e)}"
        )

@dp.callback_query()
async def handle_callbacks(callback: CallbackQuery):
    """Обробник інлайн кнопок."""
    
    if not callback.data:
        return
    
    parts = callback.data.split(":")
    if len(parts) < 3:
        return
    
    action = parts[0]
    video_id = parts[1]
    param = parts[2]
    
    user_id = callback.from_user.id
    logger.info(f"🔘 Callback {action}:{param} from user {user_id} (@{callback.from_user.username})")
    
    try:
        if action == "details":
            # Показуємо деталі теми
            await show_topic_details(callback, video_id, param)
            
        elif action == "draft":
            # Генеруємо чернетки відповідей
            await generate_drafts(callback, video_id, param)
            
    except Exception as e:
        logger.error(f"Помилка в handle_callbacks: {e}")
        await callback.answer("❌ Помилка обробки кнопки", show_alert=True)

async def show_topic_details(callback: CallbackQuery, video_id: str, topic_id: str):
    """Показує деталі конкретної теми."""
    
    await callback.answer("Завантажую деталі...")
    
    try:
        # Отримуємо цитати для теми
        quotes = get_topic_quotes(
            video_id=video_id,
            topic_id=topic_id,
            sqlite_path="./.cache.db",
            limit=3
        )
        
        topic_name = ID2NAME.get(topic_id, topic_id)
        
        if not quotes:
            await callback.message.answer(
                f"📝 <b>Деталі: {topic_name}</b>\n\n"
                "⚠️ Не знайдено коментарів для цієї теми."
            )
            return
        
        response_text = f"📝 <b>Деталі: {topic_name}</b>\n\n"
        response_text += f"<b>Найпопулярніші коментарі ({len(quotes)}):</b>\n\n"
        
        for i, quote in enumerate(quotes, 1):
            text = quote["text"][:250]
            if len(quote["text"]) > 250:
                text += "..."
            
            like_indicator = ""
            if quote["like_count"] > 0:
                like_indicator = f" ({quote['like_count']} ❤️)"
            
            response_text += f"{i}. <b>{quote['author']}</b>{like_indicator}:\n"
            response_text += f"   <i>\"{text}\"</i>\n\n"
        
        # Кнопка повернення
        keyboard = InlineKeyboardBuilder()
        keyboard.add(InlineKeyboardButton(
            text="◀️ Назад до результатів",
            callback_data=f"back:{video_id}"
        ))
        
        await callback.message.answer(
            response_text,
            reply_markup=keyboard.as_markup()
        )
        
    except Exception as e:
        logger.error(f"Помилка в show_topic_details: {e}")
        await callback.message.answer(
            f"❌ Помилка завантаження деталей: {str(e)}"
        )

async def generate_drafts(callback: CallbackQuery, video_id: str, tone: str):
    """Генерує чернетки відповідей у вказаному тоні."""
    
    await callback.answer("Генерую чернетки...")
    
    try:
        # Отримуємо топ коментарі для відео
        analysis_data = get_latest_analysis_data(video_id, "./.cache.db")
        
        if "error" in analysis_data:
            await callback.message.answer(
                f"❌ Помилка: {analysis_data['error']}"
            )
            return
        
        topics = analysis_data.get("topics", [])[:3]  # Топ-3 теми
        
        if not topics:
            await callback.message.answer(
                "⚠️ Не знайдено тем для генерації чернеток."
            )
            return
        
        # Формуємо контекст для LLM
        context_text = f"Аналіз коментарів YouTube відео {video_id}:\n\n"
        
        for i, topic in enumerate(topics, 1):
            context_text += f"{i}. {topic['name']}: {topic['count']} коментарів ({topic['share']*100:.1f}%)\n"
            if topic.get('top_quote'):
                context_text += f"   Приклад: {topic['top_quote'][:150]}\n"
        
        # Вибираємо правила за тоном
        rules = DRAFT_RULES_CALM if tone == "calm" else DRAFT_RULES_PLAYFUL
        
        # Генеруємо 2 варіанти через LLM
        await callback.message.answer("🤖 Генерую чернетки, зачекайте...")
        
        drafts = await generate_response_drafts(context_text, rules)
        
        if not drafts:
            await callback.message.answer(
                "❌ Не вдалося згенерувати чернетки. Спробуйте пізніше."
            )
            return
        
        # Формуємо відповідь
        tone_emoji = "😌" if tone == "calm" else "😄"
        tone_name = "Спокійний" if tone == "calm" else "Жартівливий"
        
        response_text = f"{tone_emoji} <b>Чернетки відповідей ({tone_name} тон)</b>\n\n"
        
        for i, draft in enumerate(drafts, 1):
            response_text += f"<b>Варіант {i}:</b>\n"
            response_text += f"<i>\"{draft}\"</i>\n\n"
        
        response_text += "💡 <i>Чернетки згенеровані на основі аналізу коментарів. "
        response_text += "Відредагуйте їх за потребою перед публікацією.</i>"
        
        await callback.message.answer(response_text)
        
    except Exception as e:
        logger.error(f"Помилка в generate_drafts: {e}")
        await callback.message.answer(
            f"❌ Помилка генерації чернеток: {str(e)}"
        )

async def generate_response_drafts(context: str, rules: str) -> List[str]:
    """Генерує 2 варіанти чернеток відповіді через LLM."""
    
    try:
        # Перевіряємо API ключ
        if not os.getenv("OPENROUTER_API_KEY"):
            return []
        
        import openai
        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        model = get_model()
        drafts = []
        
        # Генеруємо 2 варіанти
        for i in range(2):
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=0.4,  # Трохи креативності
                    messages=[
                        {"role": "system", "content": rules},
                        {"role": "user", "content": context}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=300
                ),
                timeout=30.0
            )
            
            content = response.choices[0].message.content
            
            try:
                data = json.loads(content)
                draft = data.get("draft", "").strip()
                if draft and len(draft) <= 280:
                    drafts.append(draft)
            except json.JSONDecodeError:
                logger.warning(f"Не вдалося розпарсити JSON відповідь: {content}")
        
        await client.close()
        return drafts
        
    except Exception as e:
        logger.error(f"Помилка генерації чернеток: {e}")
        return []

# Обробка звичайних повідомлень через AI-агента
@dp.message()
async def handle_message(message: types.Message):
    """Обробка повідомлень через AI-агента з function calling."""
    
    user_id = message.from_user.id
    text = (message.text or "").strip()
    
    if not text:
        await message.answer("❌ Надішліть повідомлення з текстом.")
        return
    
    # Перевіряємо чи користувач очікує URL для аналізу (legacy режим)
    if user_id in user_states and user_states[user_id].get("waiting_for_url"):
        
        # Перевіряємо чи це схоже на URL або video_id
        if any(keyword in text.lower() for keyword in ["youtube.com", "youtu.be"]) or len(text) == 11:
            # Схоже на YouTube URL або video_id - обробляємо через агента
            user_states[user_id].pop("waiting_for_url", None)
        else:
            # Не схоже на YouTube URL
            await message.answer(
                "❓ <b>Це не схоже на YouTube посилання</b>\n\n"
                "📎 <b>Підтримувані формати:</b>\n"
                "• <code>https://www.youtube.com/watch?v=dQw4w9WgXcQ</code>\n"
                "• <code>https://youtu.be/dQw4w9WgXcQ</code>\n"
                "• <code>dQw4w9WgXcQ</code>\n\n"
                "💡 Спробуйте ще раз або /start для скасування."
            )
            return
    
    # Показуємо що агент думає
    status_message = await message.answer("🤖 <i>Обробляю ваш запит...</i>")
    
    logger.info(f"📨 Processing message from user {user_id} (@{message.from_user.username}): {text[:100]}...")
    
    try:
        # Отримуємо поточний video_id з стану користувача
        current_video_id = None
        if user_id in user_states and "video_id" in user_states[user_id]:
            current_video_id = user_states[user_id]["video_id"]
            logger.info(f"🎬 Використовую контекст відео: {current_video_id}")
        
        # Обробляємо через AI-агента з контекстом
        response = await process_agent_message(text, user_id, current_video_id)
        
        # Оновлюємо стан користувача якщо знайдено video_id
        video_id = extract_video_id_from_message(text)
        if video_id:
            if user_id not in user_states:
                user_states[user_id] = {}
            user_states[user_id]["video_id"] = video_id
            user_states[user_id].pop("waiting_for_url", None)
            logger.info(f"💾 Saved video_id {video_id} for user {user_id}")
        
        logger.info(f"📤 Sending response to user {user_id} ({len(response)} chars)")
        await status_message.edit_text(response)
        
    except Exception as e:
        logger.error(f"Agent processing error: {e}")
        await status_message.edit_text(
            "❌ <b>Помилка обробки запиту</b>\n\n"
            "Спробуйте:\n"
            "• Переформулювати запит\n"
            "• Використати команди /start або /analyze\n"
            "• Перевірити підключення до інтернету"
        )

async def main():
    """Головна функція запуску бота."""
    
    logger.info("🤖 Запускаю YouTube Comment Consultant AI-агента...")
    logger.info(f"🔧 Модель агента: {os.getenv('AGENT_MODEL', 'google/gemini-2.5-flash')}")
    logger.info(f"🔧 Модель класифікації: {os.getenv('MODEL_SUMMARY', 'openai/gpt-4o-mini')}")
    
    # Перевіряємо змінні середовища
    required_vars = ["TELEGRAM_BOT_TOKEN", "OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Відсутні змінні середовища: {missing_vars}")
        return
    
    if not os.getenv("YOUTUBE_API_KEY"):
        logger.warning("⚠️ Відсутня YOUTUBE_API_KEY - можливі проблеми з новими відео")
    
    logger.info("✅ Всі змінні середовища налаштовані")
    logger.info(f"🔑 OpenRouter API: {os.getenv('OPENROUTER_API_KEY')[:20]}...") 
    logger.info(f"🤖 Bot token: {BOT_TOKEN[:20]}...")
    
    try:
        # Видаляємо webhook якщо є
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("🗑️ Webhook очищено")
        
        # Отримуємо інформацію про бота
        me = await bot.get_me()
        logger.info(f"🤖 Бот @{me.username} (ID: {me.id}) готовий до роботи!")
        logger.info("📱 Користувачі можуть надсилати /start для початку")
        logger.info("🎬 Або просто надсилати YouTube посилання для автоматичного аналізу")
        
        # Запускаємо polling
        logger.info("🔄 Запускаю polling...")
        await dp.start_polling(bot)
        
    except Exception as e:
        logger.error(f"❌ Критична помилка: {e}")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Бот зупинено користувачем")
    except Exception as e:
        logger.error(f"❌ Критична помилка запуску: {e}")
