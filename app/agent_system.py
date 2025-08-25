# app/agent_system.py
# -*- coding: utf-8 -*-
"""
AI-агент система з function calling для YouTube Comment Consultant.
Агент самостійно приймає рішення щодо використання інструментів.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union
import openai

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("agent_system")

# Імпорти наших інструментів
from app.tools.analyze_video_tool import analyze_video_tool, search_comments_for_qa
from app.tools.classification_db import get_latest_analysis_data, get_topic_quotes, get_filtered_comments
from app.tools.youtube import extract_video_id
from app.tools.topics_taxonomy import ID2NAME, TAXONOMY

# Мапінг українських назв до topic_id для агента
NAME_TO_ID = {
    "Похвала/подяка": "praise",
    "Критика/незадоволення": "critique", 
    "Питання/уточнення": "questions",
    "Поради/пропозиції": "suggestions",
    "Ведучий/персона": "host_persona",
    "Точність/правдивість": "content_truth",
    "Звук/відео/монтаж": "av_quality",
    "Ціни/цінність": "price_value",
    "Особисті історії": "personal_story",
    "Офтоп/жарти/меми": "offtopic_fun",
    "Токсичність/хейт": "toxicity"
}

def find_topic_id_by_name(user_input: str) -> str:
    """Знаходить topic_id за українською назвою або частиною назви."""
    user_input = user_input.lower()
    
    # Пряме співставлення
    for name, topic_id in NAME_TO_ID.items():
        if name.lower() in user_input:
            return topic_id
    
    # Пошук за ключовими словами
    keyword_map = {
        "похвал": "praise",
        "подяк": "praise", 
        "критик": "critique",
        "незадовол": "critique",
        "питанн": "questions",
        "уточнен": "questions",
        "поради": "suggestions",
        "пропозиц": "suggestions",
        "ведуч": "host_persona",
        "персон": "host_persona",
        "точн": "content_truth",
        "правдив": "content_truth",
        "звук": "av_quality",
        "відео": "av_quality",
        "монтаж": "av_quality",
        "цін": "price_value",
        "вартість": "price_value",
        "особист": "personal_story",
        "історі": "personal_story",
        "офтоп": "offtopic_fun",
        "жарт": "offtopic_fun",
        "мем": "offtopic_fun",
        "токсич": "toxicity",
        "хейт": "toxicity"
    }
    
    for keyword, topic_id in keyword_map.items():
        if keyword in user_input:
            return topic_id
    
    return None

def find_sentiment_by_name(user_input: str) -> str:
    """Знаходить sentiment за українською назвою."""
    user_input = user_input.lower()
    
    sentiment_keywords = {
        "позитивн": "positive",
        "схвален": "positive",
        "добр": "positive",
        "хорош": "positive",
        "негативн": "negative", 
        "поган": "negative",
        "критич": "negative",
        "незадовол": "negative",
        "нейтральн": "neutral",
        "спокійн": "neutral",
        "фактичн": "neutral"
    }
    
    for keyword, sentiment in sentiment_keywords.items():
        if keyword in user_input:
            return sentiment
    
    return None

def generate_category_insight(topic_id: str, topic_name: str, count: int, share: float, example: str) -> str:
    """Генерує інсайт для категорії коментарів."""
    share_percent = share * 100
    
    insights = {
        "praise": f"Аудиторія {share_percent:.0f}% позитивно сприймає контент. Це свідчить про високу якість та відповідність очікуванням глядачів.",
        "critique": f"{share_percent:.0f}% коментарів містять критику. Це може вказувати на проблемні місця, які варто покращити в майбутніх відео.",
        "questions": f"{share_percent:.0f}% глядачів мають питання. Це гарна можливість для створення FAQ або додаткових пояснювальних відео.",
        "suggestions": f"{share_percent:.0f}% коментарів містять пропозиції. Це цінний фідбек від аудиторії для покращення контенту.",
        "host_persona": f"{share_percent:.0f}% коментарів стосуються особистості автора. Це показує рівень особистого зв'язку з аудиторією.",
        "accuracy": f"{share_percent:.0f}% коментарів стосуються точності інформації. Це важливо для довіри до каналу.",
        "tech_quality": f"{share_percent:.0f}% коментарів про технічну якість. Це прямий фідбек щодо монтажу, звуку та відео.",
        "price_value": f"{share_percent:.0f}% коментарів про ціну/цінність. Важливо для монетизації та позиціонування контенту.",
        "personal_story": f"{share_percent:.0f}% глядачів діляться особистими історіями. Це показує вплив контенту на аудиторію.",
        "offtopic": f"{share_percent:.0f}% офтопічних коментарів. Висока частка може вказувати на зниження фокусу відео.",
        "toxic": f"{share_percent:.0f}% токсичних коментарів. Потрібна модерація та можливо зміна підходу до подачі контенту."
    }
    
    return insights.get(topic_id, f"{share_percent:.0f}% коментарів у категорії '{topic_name}'.")

# Системний промпт для агента
AGENT_SYSTEM_PROMPT = """Ти — YouTube Comment Consultant, ввічливий і тактовний консультант для авторів YouTube-каналів.

ТВОЯ РОЛЬ:
- Допомагаєш авторам YouTube розуміти реакцію глядачів через аналіз коментарів
- Відповідаєш ВИКЛЮЧНО на основі даних з проаналізованих коментарів
- Ніколи не вигадуєш інформацію, яка відсутня у коментарях
- ЗАПАМ'ЯТОВУЄШ останнє проаналізоване відео для подальших питань

ПРИНЦИПИ РОБОТИ:
1. АВТОНОМНІСТЬ: Самостійно вирішуєш які інструменти використати
2. ТОЧНІСТЬ: Відповіді лише на основі реальних даних з коментарів
3. ВВІЧЛИВІСТЬ: Тактовний тон, без токсичності
4. УКРАЇНСЬКА МОВА: За замовчуванням українська, але відповідаєш мовою користувача
5. ПАМ'ЯТЬ: Після аналізу відео відповідаєш на питання про це ж відео БЕЗ повторного аналізу

ДОСТУПНІ ІНСТРУМЕНТИ:
- analyze_video: Аналіз YouTube відео (парсинг + класифікація коментарів)
- search_comments: Пошук релевантних коментарів для відповіді на питання
- get_analysis_data: Отримання збережених результатів аналізу
- get_topic_details: Деталі конкретної теми з цитатами
- analyze_categories: Детальний аналіз всіх категорій з інсайтами та рекомендаціями
- get_filtered_comments: Пошук коментарів за темою та/або тональністю
- get_sentiment_analysis: Детальний аналіз тональності з прикладами коментарів

КОЛИ ВИКОРИСТОВУВАТИ ІНСТРУМЕНТИ:
- URL YouTube → analyze_video
- Питання про відео (коли є контекст) → search_comments або get_analysis_data
- Запит деталей теми → get_topic_details
- "Що думають про..." → search_comments (використай останнє відео якщо контекст зрозумілий)
- "Покажи топ теми" → get_analysis_data (для останнього відео)
- "Інсайти по категоріях", "Опиши категорії", "Які висновки" → analyze_categories
- "Позитивні коментарі", "Критика", фільтри за темою/тональністю → get_filtered_comments
- "Аналіз тональності", "Негативні коментарі", "Статистика емоцій" → get_sentiment_analysis

ВАЖЛИВО ПРО КАТЕГОРІЇ:
При використанні get_filtered_comments використовуй правильні topic_id:
- praise = Похвала/подяка
- critique = Критика/незадоволення  
- questions = Питання/уточнення
- suggestions = Поради/пропозиції
- host_persona = Ведучий/персона
- content_truth = Точність/правдивість
- av_quality = Звук/відео/монтаж
- price_value = Ціни/цінність
- personal_story = Особисті історії
- offtopic_fun = Офтоп/жарти/меми
- toxicity = Токсичність/хейт

ТОНАЛЬНІСТЬ (sentiment):
- positive = позитивні коментарі
- neutral = нейтральні коментарі
- negative = негативні коментарі

ВАЖЛИВО ПРО КОНТЕКСТ:
- Якщо користувач щойно проаналізував відео, ТИ ЗНАЄШ його video_id
- На питання про "коментарі", "теми", "що думають" відповідай про ОСТАННЄ відео
- НЕ ПРОСИ нове посилання якщо питання стосується попереднього аналізу
- Тільки для НОВОГО відео проси URL

ОБМЕЖЕННЯ:
- Не даєш медичних/фінансових порад
- Не називаєш користувачів поіменно
- При нестачі даних чесно кажеш про це
- Пропонуєш дії: проаналізувати інше відео, збільшити ліміт коментарів

СТИЛЬ ВІДПОВІДЕЙ:
- Коротко і по суті
- З емодзі помірно
- Структуровано (списки, категорії)
- З конкретними цитатами як докази"""

def get_agent_client():
    """Створює клієнт для агента з підтримкою function calling."""
    return openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

def get_agent_model():
    """Повертає модель для агента з підтримкою tools."""
    # Пріоритет: gemini-2.5-flash (найкращий price/performance)
    return os.getenv("AGENT_MODEL", "google/gemini-2.5-flash")

# Схеми інструментів для function calling
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_video",
            "description": "Аналізує YouTube відео: парсить коментарі, класифікує за темами, зберігає в БД",
            "parameters": {
                "type": "object",
                "properties": {
                    "url_or_id": {
                        "type": "string",
                        "description": "YouTube URL або video_id"
                    },
                    "limit": {
                        "type": "integer", 
                        "description": "Максимальна кількість коментарів для аналізу",
                        "default": 1200
                    }
                },
                "required": ["url_or_id"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "search_comments",
            "description": "Шукає релевантні коментарі для відповіді на питання користувача. Використовує поточне відео якщо video_id не вказано.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    },
                    "question": {
                        "type": "string",
                        "description": "Питання користувача"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Максимальна кількість результатів",
                        "default": 5
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_analysis_data", 
            "description": "Отримує збережені результати аналізу відео з топ-темами. Використовує поточне відео якщо video_id не вказано.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_topic_details",
            "description": "Отримує деталі конкретної теми з найкращими цитатами. Використовує поточне відео якщо video_id не вказано.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    },
                    "topic_id": {
                        "type": "string",
                        "description": "ID теми (praise, critique, questions, etc.)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Кількість цитат",
                        "default": 3
                    }
                },
                "required": ["topic_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_categories",
            "description": "Аналізує всі категорії коментарів та дає інсайти для кожної з них. Використовує поточне відео якщо video_id не вказано.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_filtered_comments",
            "description": "Отримує коментарі з фільтрацією за темою та/або тональністю. Корисно для глибокого аналізу конкретних категорій.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    },
                    "topic_id": {
                        "type": "string", 
                        "description": "Фільтр за темою (praise, critique, questions, etc.)"
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "Фільтр за тональністю (positive, neutral, negative)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Кількість коментарів",
                        "default": 10
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sentiment_analysis",
            "description": "Отримує детальний аналіз тональності коментарів з прикладами. Використовує поточне відео якщо video_id не вказано.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "ID YouTube відео (опціонально, використається поточне відео)"
                    }
                },
                "required": []
            }
        }
    }
]

async def execute_tool_call(tool_call, current_video_id: Optional[str] = None) -> Dict[str, Any]:
    """Виконує виклик інструменту та повертає результат."""
    
    # Обробляємо різні типи tool_call об'єктів
    if hasattr(tool_call, 'function'):
        # Новий OpenAI SDK формат
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool_call_id = tool_call.id
    else:
        # Старий dict формат
        function_name = tool_call["function"]["name"] 
        arguments = json.loads(tool_call["function"]["arguments"])
        tool_call_id = tool_call["id"]
    
    logger.info(f"🔧 Executing tool: {function_name} with args: {arguments}")
    
    try:
        if function_name == "analyze_video":
            logger.info(f"📺 Starting video analysis for: {arguments.get('url_or_id', 'unknown')[:50]}...")
            result = analyze_video_tool(
                arguments["url_or_id"],
                limit=arguments.get("limit", 1200),
                sqlite_path="./.cache.db"
            )
            if result.get("success"):
                stats = result.get("stats", {})
                topics_count = len(result.get("topics", []))
                logger.info(f"✅ Video analysis completed: {stats.get('classified', 0)} comments, {topics_count} topics")
            else:
                logger.error(f"❌ Video analysis failed: {result.get('error', 'Unknown error')}")
            return {"success": True, "data": result}
            
        elif function_name == "search_comments":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                return {"success": False, "error": "Не вказано video_id і немає поточного відео"}
            
            question = arguments["question"]
            
            # Перевіряємо чи питання стосується конкретної категорії або тональності
            detected_topic_id = find_topic_id_by_name(question)
            detected_sentiment = find_sentiment_by_name(question)
            
            if detected_topic_id or detected_sentiment:
                filter_info = []
                if detected_topic_id:
                    filter_info.append(f"category: {detected_topic_id} ({ID2NAME.get(detected_topic_id, detected_topic_id)})")
                if detected_sentiment:
                    filter_info.append(f"sentiment: {detected_sentiment}")
                    
                logger.info(f"🎯 Detected filter request: {', '.join(filter_info)}")
                
                # Використовуємо фільтровані коментарі замість пошуку
                comments = get_filtered_comments(
                    video_id=video_id,
                    sqlite_path="./.cache.db",
                    topic_id=detected_topic_id,
                    sentiment=detected_sentiment,
                    limit=arguments.get("max_results", 10)  # Більше коментарів для sentiment
                )
                result_comments = [{"text": c["text"], "likes": c["likes"], "author": c["author"], "topic": c["topic"], "sentiment": c["sentiment"]} for c in comments]
                logger.info(f"✅ Found {len(result_comments)} filtered comments")
                
                response_data = {"comments": result_comments}
                if detected_topic_id:
                    response_data["category"] = ID2NAME.get(detected_topic_id, detected_topic_id)
                if detected_sentiment:
                    response_data["sentiment"] = detected_sentiment
                    
                return {"success": True, "data": response_data}
            else:
                # Звичайний пошук по питанню
                logger.info(f"🔍 Searching comments for question: {question[:50]}... (video: {video_id})")
                comments = search_comments_for_qa(
                    video_id=video_id,
                    question=question,
                    sqlite_path="./.cache.db",
                    max_results=arguments.get("max_results", 5)
                )
                logger.info(f"✅ Found {len(comments)} relevant comments")
                return {"success": True, "data": {"comments": comments}}
            
        elif function_name == "get_analysis_data":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                return {"success": False, "error": "Не вказано video_id і немає поточного відео"}
            
            logger.info(f"📊 Getting analysis data for video: {video_id}")
            data = get_latest_analysis_data(
                video_id=video_id,
                sqlite_path="./.cache.db"
            )
            if "error" not in data:
                topics_count = len(data.get("topics", []))
                logger.info(f"✅ Analysis data loaded: {topics_count} topics found")
                # Виключаємо DataFrame для JSON серіалізації
                data_for_agent = {k: v for k, v in data.items() if k != "classified_comments"}
                return {"success": True, "data": data_for_agent}
            else:
                logger.warning(f"⚠️ No analysis data found: {data.get('error', 'Unknown error')}")
                return {"success": True, "data": data}
            
        elif function_name == "get_topic_details":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                return {"success": False, "error": "Не вказано video_id і немає поточного відео"}
            
            topic_id = arguments["topic_id"]
            logger.info(f"📝 Getting topic details for: {topic_id} (video: {video_id})")
            quotes = get_topic_quotes(
                video_id=video_id,
                topic_id=topic_id,
                sqlite_path="./.cache.db",
                limit=arguments.get("limit", 3)
            )
            topic_name = ID2NAME.get(topic_id, topic_id)
            logger.info(f"✅ Found {len(quotes)} quotes for topic: {topic_name}")
            return {"success": True, "data": {"topic_name": topic_name, "quotes": quotes}}
            
        elif function_name == "analyze_categories":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                return {"success": False, "error": "Не вказано video_id і немає поточного відео"}
            
            logger.info(f"🔍 Analyzing all categories for video: {video_id}")
            
            # Отримуємо дані аналізу
            data = get_latest_analysis_data(video_id, "./.cache.db")
            if "error" in data:
                return {"success": False, "error": data["error"]}
            
            # Формуємо аналіз категорій з інсайтами
            categories_analysis = []
            topics = data.get("topics", [])
            sentiment = data.get("sentiment", [])
            
            for topic in topics:
                topic_id = topic.get("topic_id")
                topic_name = topic.get("name")
                count = topic.get("count", 0)
                share = topic.get("share", 0.0)
                top_quote = topic.get("top_quote", "")
                
                # Генеруємо інсайт для категорії
                insight = generate_category_insight(topic_id, topic_name, count, share, top_quote)
                
                categories_analysis.append({
                    "topic_id": topic_id,
                    "name": topic_name,
                    "count": count,
                    "share": share,
                    "insight": insight,
                    "example": top_quote
                })
            
            result = {
                "total_comments": data.get("used_comments", 0),
                "categories": categories_analysis,
                "sentiment": sentiment
            }
            
            logger.info(f"✅ Generated insights for {len(categories_analysis)} categories")
            return {"success": True, "data": result}
            
        elif function_name == "get_filtered_comments":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                # Спробуємо знайти останній проаналізований відео
                try:
                    import sqlite3
                    with sqlite3.connect("./.cache.db") as conn:
                        result = conn.execute("""
                            SELECT video_id FROM analyses 
                            ORDER BY created_at DESC LIMIT 1
                        """).fetchone()
                        if result:
                            video_id = result[0]
                            logger.info(f"🎬 Використовую останнє проаналізоване відео: {video_id}")
                        else:
                            return {"success": False, "error": "Немає проаналізованих відео"}
                except Exception as e:
                    return {"success": False, "error": f"Помилка пошуку відео: {e}"}
            
            topic_id = arguments.get("topic_id")
            sentiment = arguments.get("sentiment")
            limit = arguments.get("limit", 10)
            
            # Якщо topic_id не вказано або не співпадає, спробуємо знайти за назвою
            if not topic_id or topic_id not in [t["id"] for t in TAXONOMY]:
                # Спробуємо знайти topic_id з повідомлення користувача в контексті
                if hasattr(execute_tool_call, '_user_message'):
                    found_topic_id = find_topic_id_by_name(execute_tool_call._user_message)
                    if found_topic_id:
                        topic_id = found_topic_id
                        logger.info(f"🎯 Автоматично визначено категорію: {topic_id} ({ID2NAME.get(topic_id, topic_id)})")
            
            logger.info(f"🔍 Getting filtered comments for {video_id} (topic={topic_id}, sentiment={sentiment}, limit={limit})")
            
            comments = get_filtered_comments(
                video_id=video_id,
                sqlite_path="./.cache.db",
                topic_id=topic_id,
                sentiment=sentiment,
                limit=limit
            )
            
            logger.info(f"✅ Found {len(comments)} filtered comments")
            return {"success": True, "data": {"comments": comments, "total": len(comments), "video_id": video_id}}
            
        elif function_name == "get_sentiment_analysis":
            # Використовуємо поточне відео якщо video_id не вказано
            video_id = arguments.get("video_id") or current_video_id
            if not video_id:
                # Спробуємо знайти останній проаналізований відео
                try:
                    import sqlite3
                    with sqlite3.connect("./.cache.db") as conn:
                        result = conn.execute("""
                            SELECT video_id FROM analyses 
                            ORDER BY created_at DESC LIMIT 1
                        """).fetchone()
                        if result:
                            video_id = result[0]
                        else:
                            return {"success": False, "error": "Немає проаналізованих відео"}
                except Exception as e:
                    return {"success": False, "error": f"Помилка пошуку відео: {e}"}
            
            logger.info(f"😊😐😟 Getting sentiment analysis for video: {video_id}")
            
            # Отримуємо загальні дані аналізу (включно з sentiment)
            data = get_latest_analysis_data(video_id, "./.cache.db")
            if "error" in data:
                return {"success": False, "error": data["error"]}
            
            # Формуємо детальний аналіз тональності з прикладами
            sentiment_analysis = {
                "total_comments": data.get("used_comments", 0),
                "sentiment_distribution": data.get("sentiment", []),
                "examples": {}
            }
            
            # Отримуємо приклади для кожної тональності
            for sentiment_info in data.get("sentiment", []):
                sentiment = sentiment_info["sentiment"]
                count = sentiment_info["count"]
                
                if count > 0:
                    examples = get_filtered_comments(
                        video_id=video_id,
                        sqlite_path="./.cache.db",
                        sentiment=sentiment,
                        limit=3
                    )
                    sentiment_analysis["examples"][sentiment] = examples
            
            logger.info(f"✅ Generated sentiment analysis with {len(sentiment_analysis['examples'])} sentiment categories")
            return {"success": True, "data": sentiment_analysis}
            
        else:
            return {"success": False, "error": f"Unknown function: {function_name}"}
            
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return {"success": False, "error": str(e)}

async def process_agent_message(user_message: str, user_id: int, current_video_id: Optional[str] = None) -> str:
    """
    Основна функція обробки повідомлення через агента з function calling.
    
    Args:
        user_message: Повідомлення користувача
        user_id: ID користувача
        current_video_id: ID поточного відео (якщо є)
        
    Returns:
        Відповідь агента
    """
    
    try:
        client = get_agent_client()
        model = get_agent_model()
        
        # Формуємо повідомлення з контекстом
        user_content = user_message
        if current_video_id and not extract_video_id_from_message(user_message):
            # Додаємо контекст поточного відео, якщо в повідомленні немає нового URL
            user_content = f"[КОНТЕКСТ: Поточне відео: {current_video_id}]\n\n{user_message}"
            logger.info(f"🧠 Додаю контекст поточного відео: {current_video_id}")
        
        # Початковий виклик агента
        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        logger.info(f"🤖 Agent processing message from user {user_id}: {user_message[:100]}...")
        
        # Перевіряємо чи є YouTube URL
        video_id = extract_video_id_from_message(user_message)
        if video_id:
            logger.info(f"🎬 Detected YouTube video: {video_id}")
        
        # Перший виклик агента
        logger.info(f"🧠 Calling {model} for initial processing...")
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2000
            ),
            timeout=60.0
        )
        
        assistant_message = response.choices[0].message
        logger.info(f"🎯 Agent decided: {'use tools' if assistant_message.tool_calls else 'direct response'}")
        
        messages.append({
            "role": "assistant", 
            "content": assistant_message.content,
            "tool_calls": assistant_message.tool_calls
        })
        
        # Обробляємо tool calls якщо є
        if assistant_message.tool_calls:
            logger.info(f"🔧 Agent requested {len(assistant_message.tool_calls)} tool calls")
            
            # Виконуємо всі tool calls
            for i, tool_call in enumerate(assistant_message.tool_calls, 1):
                logger.info(f"⚙️ Executing tool {i}/{len(assistant_message.tool_calls)}...")
                result = await execute_tool_call(tool_call, current_video_id)
                
                # Отримуємо tool_call_id правильно
                if hasattr(tool_call, 'id'):
                    tool_call_id = tool_call.id
                else:
                    tool_call_id = tool_call["id"]
                
                # Додаємо результат до повідомлень
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(result, ensure_ascii=False)
                })
            
            # Другий виклик агента з результатами tools
            logger.info(f"🧠 Calling {model} for final response generation...")
            final_response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                ),
                timeout=60.0
            )
            
            final_content = final_response.choices[0].message.content
        else:
            # Якщо tool calls не потрібні
            final_content = assistant_message.content
        
        await client.close()
        
        response_length = len(final_content) if final_content else 0
        logger.info(f"✅ Agent response generated successfully ({response_length} chars)")
        return final_content or "Вибачте, не зміг сформувати відповідь."
        
    except asyncio.TimeoutError:
        logger.error("Agent timeout")
        return "⏰ Вибачте, обробка зайняла занадто багато часу. Спробуйте ще раз."
        
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"❌ Помилка агента: {str(e)}"

def extract_video_id_from_message(message: str) -> Optional[str]:
    """Витягає video_id з повідомлення користувача."""
    
    # Шукаємо YouTube URL в повідомленні
    import re
    
    # Патерни для YouTube URLs
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'\b([a-zA-Z0-9_-]{11})\b'  # Просто video_id
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            potential_id = match.group(1)
            # Перевіряємо через наш існуючий екстрактор
            video_id = extract_video_id(potential_id)
            if video_id:
                return video_id
    
    return None

async def is_youtube_related_message(message: str) -> bool:
    """Перевіряє чи повідомлення стосується YouTube."""
    
    youtube_keywords = [
        'youtube.com', 'youtu.be', 'відео', 'video', 'коментар', 'comment',
        'глядач', 'viewer', 'канал', 'channel', 'ролик'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in youtube_keywords) or extract_video_id_from_message(message) is not None
