# app/tools/analyze_video_tool.py
# -*- coding: utf-8 -*-
"""
Високорівневий інструмент для аналізу YouTube відео для Telegram бота.
Об'єднує весь пайплайн: завантаження → препроцесинг → класифікація → збереження.
"""

from __future__ import annotations
import os
import time
from typing import Dict, Any, Optional, List
import pandas as pd

# Імпорти нашого пайплайну
from .youtube import fetch_comments, extract_video_id
from .preprocess import select_fast_batch, preprocess_comments_df
from .topics_taxonomy import TAXONOMY, ID2NAME
from .topics_llm import classify_llm_full, aggregate_topics
from .classification_db import save_analysis_to_db, get_latest_analysis_data

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("analyze_video_tool")

def aggregate_sentiment(df_classified: pd.DataFrame) -> pd.DataFrame:
    """Обчислює статистику тональності."""
    if df_classified.empty or "sentiment" not in df_classified.columns:
        return pd.DataFrame(columns=["sentiment", "count", "share"])
    
    # Рахуємо кількість кожної тональності
    sentiment_counts = df_classified["sentiment"].value_counts()
    total = len(df_classified)
    
    # Створюємо DataFrame з результатами
    result = []
    for sentiment in ["positive", "neutral", "negative"]:
        count = sentiment_counts.get(sentiment, 0)
        share = count / total if total > 0 else 0.0
        result.append({
            "sentiment": sentiment,
            "count": count,
            "share": share
        })
    
    return pd.DataFrame(result)

def analyze_video_tool(
    url_or_id: str,
    *,
    limit: int = 1200,
    sqlite_path: str = "./.cache.db",
    fast_mode: bool = True,
    force_reanalyze: bool = False
) -> Dict[str, Any]:
    """
    Головний інструмент для аналізу YouTube відео через LLM.
    
    Args:
        url_or_id: YouTube URL або video_id
        limit: Максимальна кількість коментарів для аналізу
        sqlite_path: Шлях до SQLite кешу
        fast_mode: Використовувати швидкий режим (топ коментарі за лайками)
        force_reanalyze: Примусово переаналізувати навіть якщо є збережені дані
        
    Returns:
        {
            "success": bool,
            "error": str | None,
            "video_id": str,
            "analysis_id": int,
            "stats": {
                "total_fetched": int,
                "used_for_analysis": int,
                "classified": int
            },
            "topics": [
                {
                    "topic_id": str,
                    "name": str, 
                    "count": int,
                    "share": float,
                    "top_quote": str
                }
            ],
            "processing_time": float
        }
    """
    start_time = time.perf_counter()
    
    try:
        # 1. Витягаємо video_id
        video_id = extract_video_id(url_or_id)
        if not video_id:
            return {
                "success": False,
                "error": "Не вдалося витягти video_id з URL",
                "video_id": None
            }
        
        logger.info(f"🎬 Починаємо аналіз відео: {video_id} (limit={limit}, fast_mode={fast_mode})")
        
        # 2. Перевіряємо чи є збережені результати
        if not force_reanalyze:
            logger.info("🔍 Перевіряю чи є збережені результати...")
            existing_data = get_latest_analysis_data(video_id, sqlite_path)
            if "error" not in existing_data and existing_data.get("topics"):
                logger.info(f"📊 Знайдено збережені результати для {video_id}")
                processing_time = time.perf_counter() - start_time
                
                return {
                    "success": True,
                    "error": None,
                    "video_id": video_id,
                    "analysis_id": existing_data["analysis_id"],
                    "stats": {
                        "total_fetched": existing_data["total_comments"],
                        "used_for_analysis": existing_data["used_comments"],
                        "classified": existing_data["used_comments"]
                    },
                    "topics": existing_data["topics"][:5],  # Топ-5 тем
                    "sentiment": existing_data.get("sentiment", []),
                    "processing_time": processing_time,
                    "from_cache": True
                }
        
        # 3. Завантажуємо коментарі
        logger.info("📥 Завантаження коментарів...")
        df_all = fetch_comments(
            url_or_id,
            sqlite_path=sqlite_path,
            include_replies=True,
            max_comments=min(5000, limit * 4)  # Завантажуємо більше для відбору
        )
        
        if df_all.empty:
            return {
                "success": False,
                "error": "Не вдалося завантажити коментарі. Можливо, відео приватне або відсутні коментарі.",
                "video_id": video_id
            }
        
        logger.info(f"   Завантажено: {len(df_all)} коментарів")
        
        # 4. Швидкий режим + препроцесинг
        logger.info("🔧 Препроцесинг коментарів...")
        if fast_mode:
            df_selected = select_fast_batch(
                df_all, 
                mode="top_likes", 
                limit=limit, 
                include_replies=False
            )
        else:
            df_selected = df_all.head(limit)
        
        logger.info(f"   Відібрано для аналізу: {len(df_selected)} коментарів")
        
        df_processed = preprocess_comments_df(
            df_selected,
            min_chars=12,
            keep_langs=None  # Залишаємо всі мови для LLM
        )
        
        if df_processed.empty:
            return {
                "success": False,
                "error": "Після препроцесингу не залишилося коментарів для аналізу",
                "video_id": video_id
            }
        
        logger.info(f"   Після препроцесингу: {len(df_processed)} коментарів")
        
        # 5. LLM класифікація
        logger.info("🤖 LLM класифікація через OpenRouter...")
        
        # Перевіряємо API ключ
        if not os.getenv("OPENROUTER_API_KEY"):
            return {
                "success": False,
                "error": "Відсутній OPENROUTER_API_KEY в змінних середовища",
                "video_id": video_id
            }
        
        # Обрізаємо тексти до 500 символів для економії токенів
        df_for_llm = df_processed.copy()
        df_for_llm["text_clean"] = df_for_llm["text_clean"].apply(
            lambda x: str(x)[:500] if isinstance(x, str) else ""
        )
        
        df_classified = classify_llm_full(
            df_for_llm, 
            TAXONOMY, 
            text_col="text_clean", 
            batch_size=20
        )
        
        logger.info(f"   Класифіковано: {len(df_classified)} коментарів")
        
        # 6. Агрегація тем
        logger.info("📊 Агрегація результатів...")
        topics_summary = aggregate_topics(df_classified)
        sentiment_summary = aggregate_sentiment(df_classified)
        
        if topics_summary.empty:
            return {
                "success": False,
                "error": "Не вдалося створити зведення тем",
                "video_id": video_id
            }
        
        # 7. Збереження в БД
        logger.info("💾 Збереження результатів...")
        model_name = os.getenv("MODEL_SUMMARY", "openai/gpt-4o-mini")
        
        analysis_id = save_analysis_to_db(
            video_id=video_id,
            total_comments=len(df_all),
            used_comments=len(df_classified),
            model_name=model_name,
            df_classified=df_classified,
            topics_summary=topics_summary,
            sqlite_path=sqlite_path,
            fast_mode=fast_mode,
            sentiment_summary=sentiment_summary
        )
        
        if analysis_id == -1:
            logger.warning("⚠️ Не вдалося зберегти результати в БД, але аналіз виконано")
        
        # 8. Формуємо фінальну відповідь
        processing_time = time.perf_counter() - start_time
        
        # Готуємо топ-теми з цитатами
        topics_with_quotes = []
        for _, topic_row in topics_summary.head(5).iterrows():
            topic_id = topic_row["topic_id"]
            topic_name = ID2NAME.get(topic_id, topic_id)
            
            # Знаходимо найкращу цитату для цієї теми
            topic_comments = df_classified[
                df_classified["topic_labels_llm"].apply(
                    lambda labels: isinstance(labels, list) and topic_id in labels
                )
            ]
            
            top_quote = ""
            if not topic_comments.empty:
                best_comment = topic_comments.sort_values(
                    ["like_count", "published_at"], 
                    ascending=[False, True]
                ).iloc[0]
                top_quote = str(best_comment.get("text_clean", ""))[:200]
            
            topics_with_quotes.append({
                "topic_id": topic_id,
                "name": topic_name,
                "count": int(topic_row["count"]),
                "share": float(topic_row["share"]),
                "top_quote": top_quote
            })
        
        logger.info(f"✅ Аналіз завершено за {processing_time:.2f} сек")
        
        return {
            "success": True,
            "error": None,
            "video_id": video_id,
            "analysis_id": analysis_id,
            "stats": {
                "total_fetched": len(df_all),
                "used_for_analysis": len(df_processed),
                "classified": len(df_classified)
            },
            "topics": topics_with_quotes,
            "sentiment": saved_data.get("sentiment", []),
            "processing_time": processing_time,
            "from_cache": False
        }
        
    except Exception as e:
        processing_time = time.perf_counter() - start_time
        logger.error(f"❌ Помилка аналізу: {e}")
        
        return {
            "success": False,
            "error": f"Помилка аналізу: {str(e)}",
            "video_id": video_id if 'video_id' in locals() else None,
            "processing_time": processing_time
        }

def search_comments_for_qa(
    video_id: str,
    question: str,
    sqlite_path: str = "./.cache.db",
    max_results: int = 5
) -> List[Dict[str, Any]]:
    """
    Наївний пошук коментарів для Q&A функціоналу.
    Шукає коментарі, що містять ключові слова з питання.
    
    Args:
        video_id: ID YouTube відео
        question: Питання користувача
        sqlite_path: Шлях до SQLite кешу
        max_results: Максимальна кількість результатів
        
    Returns:
        [
            {
                "comment_id": str,
                "text": str,
                "author": str,
                "like_count": int,
                "relevance_score": float
            }
        ]
    """
    try:
        import sqlite3
        import re
        from collections import Counter
        
        # Витягаємо ключові слова з питання
        question_lower = question.lower()
        # Прибираємо стоп-слова та короткі слова
        stop_words = {"що", "як", "де", "коли", "чому", "чи", "і", "в", "на", "з", "для", "про", "або", "та", "але"}
        words = re.findall(r'\b\w{3,}\b', question_lower)
        keywords = [w for w in words if w not in stop_words][:10]  # Максимум 10 ключових слів
        
        if not keywords:
            return []
        
        logger.info(f"🔍 Пошук коментарів для питання: {question[:50]}...")
        logger.info(f"   Ключові слова: {keywords}")
        
        with sqlite3.connect(sqlite_path) as conn:
            # Отримуємо всі коментарі для відео
            query = """
                SELECT 
                    c.comment_id, c.text, c.author, c.like_count, c.published_at
                FROM comments c
                WHERE c.video_id = ?
                ORDER BY c.like_count DESC
            """
            
            results = []
            cursor = conn.execute(query, [video_id])
            
            for row in cursor.fetchall():
                comment_id, text, author, like_count, published_at = row
                text_lower = str(text or "").lower()
                
                # Рахуємо релевантність
                relevance_score = 0
                for keyword in keywords:
                    if keyword in text_lower:
                        relevance_score += 1
                        # Бонус за точний збіг
                        if keyword == text_lower or f" {keyword} " in text_lower:
                            relevance_score += 0.5
                
                # Додаємо бонус за кількість лайків
                like_bonus = min(like_count / 100, 1.0) if like_count > 0 else 0
                relevance_score += like_bonus
                
                if relevance_score > 0:
                    results.append({
                        "comment_id": comment_id,
                        "text": str(text or ""),
                        "author": str(author or ""),
                        "like_count": int(like_count or 0),
                        "relevance_score": relevance_score
                    })
            
            # Сортуємо за релевантністю та лайками
            results.sort(key=lambda x: (x["relevance_score"], x["like_count"]), reverse=True)
            
            logger.info(f"   Знайдено {len(results)} релевантних коментарів")
            return results[:max_results]
        
    except Exception as e:
        logger.error(f"❌ Помилка пошуку коментарів: {e}")
        return []
