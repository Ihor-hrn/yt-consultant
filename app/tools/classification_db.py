# app/tools/classification_db.py
# -*- coding: utf-8 -*-
"""
Модуль для збереження результатів класифікації коментарів в SQLite.
Потрібно для агента-персони, щоб він міг використовувати історичні дані.
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("classification_db")

def _ensure_database_schema(conn: sqlite3.Connection) -> None:
    """Створює всі необхідні таблиці для бота при необхідності."""
    
    # Таблиця analyses - інформація про проведені аналізи
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            total_comments INTEGER NOT NULL,
            used_comments INTEGER NOT NULL,
            fast_mode INTEGER NOT NULL DEFAULT 1
        )
    """)
    
    # Таблиця comment_labels - результати класифікації для кожного коментаря
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comment_labels (
            comment_id TEXT NOT NULL,
            video_id TEXT NOT NULL,
            labels_json TEXT NOT NULL,  -- JSON array: ["praise", "suggestions"]
            top_label TEXT,             -- Top category: "praise"
            sentiment TEXT,             -- Sentiment: "positive", "neutral", "negative"
            analysis_id INTEGER,
            PRIMARY KEY (comment_id, video_id),
            FOREIGN KEY (comment_id) REFERENCES comments (comment_id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
        )
    """)
    
    # Таблиця topics_summary - зведення тем для кожного відео
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topics_summary (
            video_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            count INTEGER NOT NULL,
            share REAL NOT NULL,
            analysis_id INTEGER,
            PRIMARY KEY (video_id, topic_id),
            FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
        )
    """)
    
    # Таблиця sentiment_summary - зведення тональності для кожного відео
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_summary (
            video_id TEXT NOT NULL,
            sentiment TEXT NOT NULL,    -- "positive", "neutral", "negative"
            count INTEGER NOT NULL,
            share REAL NOT NULL,        -- 0.0 - 1.0
            analysis_id INTEGER,
            PRIMARY KEY (video_id, sentiment),
            FOREIGN KEY (analysis_id) REFERENCES analyses (analysis_id)
        )
    """)
    
    # Оновлена таблиця classification_results для зворотної сумісності
    conn.execute("""
        CREATE TABLE IF NOT EXISTS classification_results (
            comment_id TEXT PRIMARY KEY,
            video_id TEXT NOT NULL,
            topic_labels TEXT,  -- JSON array: ["praise", "suggestions"]
            topic_top TEXT,     -- Top category: "praise"
            confidence REAL,    -- Confidence score (0-1)
            classified_at TEXT, -- ISO timestamp
            model_used TEXT,    -- e.g. "openai/gpt-4o-mini"
            batch_size INTEGER, -- Batch size used for classification
            FOREIGN KEY (comment_id) REFERENCES comments (comment_id)
        )
    """)
    
    # Індекси для швидкого пошуку
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_video_id ON analyses(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_comment_labels_video_id ON comment_labels(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_comment_labels_top_label ON comment_labels(top_label)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_summary_video_id ON topics_summary(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_classification_video_id ON classification_results(video_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_classification_topic_top ON classification_results(topic_top)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_classification_classified_at ON classification_results(classified_at)")
    
    conn.commit()

def save_classification_results(
    df: pd.DataFrame, 
    sqlite_path: str,
    model_name: str = "openai/gpt-4o-mini",
    batch_size: int = 20
) -> bool:
    """
    Зберігає результати класифікації в SQLite.
    
    Args:
        df: DataFrame з колонками comment_id, video_id, topic_labels_llm, topic_top_llm
        sqlite_path: Шлях до SQLite файлу
        model_name: Назва використаної моделі
        batch_size: Розмір батчу, що використовувався
        
    Returns:
        bool: True якщо успішно збережено
    """
    if df.empty:
        logger.warning("Немає даних для збереження класифікації")
        return False
    
    try:
        with sqlite3.connect(sqlite_path) as conn:
            _ensure_database_schema(conn)
            
            # Підготовка даних
            rows = []
            for _, row in df.iterrows():
                labels = row.get("topic_labels_llm", [])
                labels_json = json.dumps(labels) if isinstance(labels, list) else "[]"
                
                rows.append({
                    "comment_id": str(row["comment_id"]),
                    "video_id": str(row.get("video_id", "")),
                    "topic_labels": labels_json,
                    "topic_top": row.get("topic_top_llm"),
                    "confidence": 1.0,  # TODO: додати реальну confidence від LLM
                    "classified_at": pd.Timestamp.utcnow().isoformat(),
                    "model_used": model_name,
                    "batch_size": batch_size
                })
            
            # Upsert в БД
            conn.executemany("""
                INSERT INTO classification_results 
                (comment_id, video_id, topic_labels, topic_top, confidence, classified_at, model_used, batch_size)
                VALUES 
                (:comment_id, :video_id, :topic_labels, :topic_top, :confidence, :classified_at, :model_used, :batch_size)
                ON CONFLICT(comment_id) DO UPDATE SET
                    topic_labels=excluded.topic_labels,
                    topic_top=excluded.topic_top,
                    confidence=excluded.confidence,
                    classified_at=excluded.classified_at,
                    model_used=excluded.model_used,
                    batch_size=excluded.batch_size
            """, rows)
            
            conn.commit()
            logger.info(f"💾 Збережено результати класифікації для {len(rows)} коментарів в {sqlite_path}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Помилка збереження класифікації: {e}")
        return False

def load_classification_results(
    video_id: str, 
    sqlite_path: str
) -> pd.DataFrame:
    """
    Завантажує збережені результати класифікації для відео.
    
    Args:
        video_id: ID YouTube відео
        sqlite_path: Шлях до SQLite файлу
        
    Returns:
        DataFrame з результатами або порожній DataFrame
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            query = """
                SELECT 
                    c.comment_id, c.video_id, c.text, c.like_count, c.published_at,
                    c.author, c.lang,
                    cr.topic_labels, cr.topic_top, cr.confidence, 
                    cr.classified_at, cr.model_used, cr.batch_size
                FROM comments c
                LEFT JOIN classification_results cr ON c.comment_id = cr.comment_id
                WHERE c.video_id = ?
                ORDER BY c.like_count DESC
            """
            df = pd.read_sql_query(query, conn, params=[video_id])
            
            # Парсинг JSON для topic_labels
            if not df.empty and "topic_labels" in df.columns:
                df["topic_labels_llm"] = df["topic_labels"].apply(
                    lambda x: json.loads(x) if x and x != 'null' else []
                )
                df["topic_top_llm"] = df["topic_top"]
                
            logger.info(f"📊 Завантажено {len(df)} коментарів з результатами класифікації для відео {video_id}")
            return df
            
    except Exception as e:
        logger.error(f"❌ Помилка завантаження класифікації: {e}")
        return pd.DataFrame()

def get_topic_statistics(
    video_id: str, 
    sqlite_path: str
) -> Dict[str, Any]:
    """
    Повертає статистику тем для відео.
    
    Args:
        video_id: ID YouTube відео
        sqlite_path: Шлях до SQLite файлу
        
    Returns:
        {"total_comments": int, "topics": [{"topic": str, "count": int, "share": float}]}
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            # Загальна кількість коментарів
            total_query = "SELECT COUNT(*) as total FROM classification_results WHERE video_id = ?"
            total_result = conn.execute(total_query, [video_id]).fetchone()
            total_comments = total_result[0] if total_result else 0
            
            if total_comments == 0:
                return {"total_comments": 0, "topics": [], "video_id": video_id}
            
            # Статистика тем
            topics_query = """
                SELECT 
                    topic_top as topic,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 1.0 / ? * 100, 1) as share_percent
                FROM classification_results 
                WHERE video_id = ? AND topic_top IS NOT NULL
                GROUP BY topic_top
                ORDER BY count DESC
            """
            topics_df = pd.read_sql_query(
                topics_query, 
                conn, 
                params=[total_comments, video_id]
            )
            
            topics = topics_df.to_dict('records')
            
            # Додаємо інформацію про відео
            video_info_query = """
                SELECT COUNT(*) as total_comments_in_db
                FROM comments 
                WHERE video_id = ?
            """
            video_info = conn.execute(video_info_query, [video_id]).fetchone()
            total_in_db = video_info[0] if video_info else 0
            
            return {
                "video_id": video_id,
                "total_comments": total_comments,
                "total_in_db": total_in_db,
                "classification_coverage": round(total_comments / total_in_db * 100, 1) if total_in_db > 0 else 0,
                "topics": topics
            }
            
    except Exception as e:
        logger.error(f"❌ Помилка статистики: {e}")
        return {"total_comments": 0, "topics": [], "video_id": video_id}

def get_video_list_with_classification(sqlite_path: str) -> pd.DataFrame:
    """
    Повертає список всіх відео з інформацією про класифікацію.
    
    Returns:
        DataFrame з колонками: video_id, total_comments, classified_comments, last_classified_at
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            query = """
                SELECT 
                    c.video_id,
                    COUNT(c.comment_id) as total_comments,
                    COUNT(cr.comment_id) as classified_comments,
                    MAX(cr.classified_at) as last_classified_at,
                    cr.model_used
                FROM comments c
                LEFT JOIN classification_results cr ON c.comment_id = cr.comment_id
                GROUP BY c.video_id
                ORDER BY last_classified_at DESC NULLS LAST
            """
            df = pd.read_sql_query(query, conn)
            
            if not df.empty:
                df["classification_coverage"] = round(
                    df["classified_comments"] / df["total_comments"] * 100, 1
                )
            
            logger.info(f"📋 Знайдено {len(df)} відео в базі даних")
            return df
            
    except Exception as e:
        logger.error(f"❌ Помилка отримання списку відео: {e}")
        return pd.DataFrame()

def delete_classification_results(
    video_id: Optional[str] = None, 
    sqlite_path: str = "./.cache.db"
) -> bool:
    """
    Видаляє результати класифікації.
    
    Args:
        video_id: ID відео для видалення (якщо None - видаляє всі)
        sqlite_path: Шлях до SQLite файлу
        
    Returns:
        bool: True якщо успішно видалено
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            if video_id:
                conn.execute("DELETE FROM classification_results WHERE video_id = ?", [video_id])
                logger.info(f"🗑️ Видалено результати класифікації для відео {video_id}")
            else:
                conn.execute("DELETE FROM classification_results")
                logger.info("🗑️ Видалено всі результати класифікації")
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"❌ Помилка видалення: {e}")
        return False

def save_analysis_to_db(
    video_id: str,
    total_comments: int, 
    used_comments: int,
    model_name: str,
    df_classified: pd.DataFrame,
    topics_summary: pd.DataFrame,
    sqlite_path: str,
    fast_mode: bool = True,
    sentiment_summary: Optional[pd.DataFrame] = None
) -> int:
    """
    Зберігає повний аналіз відео в нових таблицях.
    
    Args:
        video_id: ID YouTube відео
        total_comments: Загальна кількість коментарів
        used_comments: Кількість коментарів використаних для аналізу
        model_name: Назва LLM моделі
        df_classified: DataFrame з класифікованими коментарями (має містити 'sentiment')
        topics_summary: DataFrame зі зведенням тем
        sqlite_path: Шлях до SQLite файлу
        fast_mode: Чи використовувався швидкий режим
        sentiment_summary: DataFrame з підсумком тональності
        
    Returns:
        analysis_id: ID створеного аналізу
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            _ensure_database_schema(conn)
            
            # 1. Створюємо запис аналізу
            created_at = pd.Timestamp.utcnow().isoformat()
            cursor = conn.execute("""
                INSERT INTO analyses (video_id, created_at, model, total_comments, used_comments, fast_mode)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [video_id, created_at, model_name, total_comments, used_comments, int(fast_mode)])
            
            analysis_id = cursor.lastrowid
            
            # 2. Зберігаємо мітки коментарів
            comment_labels = []
            for _, row in df_classified.iterrows():
                labels = row.get("topic_labels_llm", [])
                labels_json = json.dumps(labels) if isinstance(labels, list) else "[]"
                
                comment_labels.append({
                    "comment_id": str(row["comment_id"]),
                    "video_id": video_id,
                    "labels_json": labels_json,
                    "top_label": row.get("topic_top_llm"),
                    "sentiment": row.get("sentiment", "neutral"),  # За замовчуванням neutral
                    "analysis_id": analysis_id
                })
            
            if comment_labels:
                conn.executemany("""
                    INSERT OR REPLACE INTO comment_labels 
                    (comment_id, video_id, labels_json, top_label, sentiment, analysis_id)
                    VALUES (:comment_id, :video_id, :labels_json, :top_label, :sentiment, :analysis_id)
                """, comment_labels)
            
            # 3. Зберігаємо зведення тем
            topic_rows = []
            for _, row in topics_summary.iterrows():
                topic_rows.append({
                    "video_id": video_id,
                    "topic_id": row["topic_id"], 
                    "count": int(row["count"]),
                    "share": float(row["share"]),
                    "analysis_id": analysis_id
                })
            
            if topic_rows:
                # Спочатку видаляємо старі дані для цього відео
                conn.execute("DELETE FROM topics_summary WHERE video_id = ?", [video_id])
                
                conn.executemany("""
                    INSERT INTO topics_summary (video_id, topic_id, count, share, analysis_id)
                    VALUES (:video_id, :topic_id, :count, :share, :analysis_id)
                """, topic_rows)
            
            # 4. Зберігаємо зведення тональності (якщо є)
            sentiment_rows = []
            if sentiment_summary is not None:
                for _, row in sentiment_summary.iterrows():
                    sentiment_rows.append({
                        "video_id": video_id,
                        "sentiment": row["sentiment"],
                        "count": int(row["count"]),
                        "share": float(row["share"]),
                        "analysis_id": analysis_id
                    })
                
                if sentiment_rows:
                    conn.executemany("""
                        INSERT OR REPLACE INTO sentiment_summary 
                        (video_id, sentiment, count, share, analysis_id)
                        VALUES (:video_id, :sentiment, :count, :share, :analysis_id)
                    """, sentiment_rows)
            
            conn.commit()
            logger.info(f"💾 Збережено аналіз #{analysis_id} для відео {video_id}: {len(comment_labels)} коментарів, {len(topic_rows)} тем, {len(sentiment_rows)} тональностей")
            return analysis_id
            
    except Exception as e:
        logger.error(f"❌ Помилка збереження аналізу: {e}")
        return -1

def get_latest_analysis_data(video_id: str, sqlite_path: str) -> Dict[str, Any]:
    """
    Повертає дані останнього аналізу для відео у форматі для Telegram бота.
    
    Returns:
        {
            "analysis_id": int,
            "video_id": str,
            "total_comments": int,
            "used_comments": int,
            "model": str,
            "created_at": str,
            "topics": [{"topic_id": str, "name": str, "count": int, "share": float, "top_quote": str}],
            "classified_comments": DataFrame
        }
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            # Отримуємо останній аналіз
            analysis_query = """
                SELECT * FROM analyses 
                WHERE video_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            analysis_df = pd.read_sql_query(analysis_query, conn, params=[video_id])
            
            if analysis_df.empty:
                return {"error": "Немає збережених аналізів для цього відео"}
            
            analysis = analysis_df.iloc[0].to_dict()
            analysis_id = analysis["analysis_id"]
            
            # Отримуємо топ тем з зведення
            topics_query = """
                SELECT * FROM topics_summary 
                WHERE video_id = ? 
                ORDER BY count DESC
            """
            topics_df = pd.read_sql_query(topics_query, conn, params=[video_id])
            
            # Отримуємо статистику тональності
            sentiment_query = """
                SELECT * FROM sentiment_summary 
                WHERE video_id = ? 
                ORDER BY count DESC
            """
            sentiment_df = pd.read_sql_query(sentiment_query, conn, params=[video_id])
            
            # Отримуємо класифіковані коментарі з цитатами
            comments_query = """
                SELECT 
                    c.comment_id, c.text, c.like_count, c.published_at, c.author,
                    cl.labels_json, cl.top_label
                FROM comments c
                JOIN comment_labels cl ON c.comment_id = cl.comment_id
                WHERE cl.video_id = ?
                ORDER BY c.like_count DESC
            """
            comments_df = pd.read_sql_query(comments_query, conn, params=[video_id])
            
            # Обробляємо теми з цитатами
            from .topics_taxonomy import ID2NAME
            topics_with_quotes = []
            
            for _, topic_row in topics_df.iterrows():
                topic_id = topic_row["topic_id"]
                topic_name = ID2NAME.get(topic_id, topic_id)
                
                # Знаходимо найкращу цитату для цієї теми
                topic_comments = comments_df[comments_df["top_label"] == topic_id]
                top_quote = ""
                if not topic_comments.empty:
                    best_comment = topic_comments.iloc[0]
                    top_quote = str(best_comment["text"] or "")[:200]
                
                topics_with_quotes.append({
                    "topic_id": topic_id,
                    "name": topic_name,
                    "count": int(topic_row["count"]),
                    "share": float(topic_row["share"]),
                    "top_quote": top_quote
                })
            
            # Обробляємо JSON мітки у коментарях
            if not comments_df.empty and "labels_json" in comments_df.columns:
                comments_df["topic_labels_llm"] = comments_df["labels_json"].apply(
                    lambda x: json.loads(x) if x and x != 'null' else []
                )
            
            # Створюємо список тональностей
            sentiment_names = {"positive": "Позитивна", "neutral": "Нейтральна", "negative": "Негативна"}
            sentiment_emojis = {"positive": "😊", "neutral": "😐", "negative": "😟"}
            sentiment_list = []
            
            for _, row in sentiment_df.iterrows():
                sentiment_id = row["sentiment"]
                sentiment_list.append({
                    "sentiment": sentiment_id,
                    "name": sentiment_names.get(sentiment_id, sentiment_id),
                    "emoji": sentiment_emojis.get(sentiment_id, ""),
                    "count": int(row["count"]),
                    "share": float(row["share"])
                })
            
            return {
                "analysis_id": analysis_id,
                "video_id": video_id,
                "total_comments": analysis["total_comments"],
                "used_comments": analysis["used_comments"],
                "model": analysis["model"],
                "created_at": analysis["created_at"],
                "topics": topics_with_quotes,
                "sentiment": sentiment_list,
                "classified_comments": comments_df
            }
            
    except Exception as e:
        logger.error(f"❌ Помилка отримання даних аналізу: {e}")
        return {"error": f"Помилка: {e}"}

def get_topic_quotes(video_id: str, topic_id: str, sqlite_path: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Повертає найкращі цитати для конкретної теми.
    
    Returns:
        [{"comment_id": str, "text": str, "author": str, "like_count": int}, ...]
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            query = """
                SELECT 
                    c.comment_id, c.text, c.author, c.like_count, c.published_at
                FROM comments c
                JOIN comment_labels cl ON c.comment_id = cl.comment_id
                WHERE cl.video_id = ? AND cl.top_label = ?
                ORDER BY c.like_count DESC, c.published_at DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=[video_id, topic_id, limit])
            
            quotes = []
            for _, row in df.iterrows():
                quotes.append({
                    "comment_id": row["comment_id"],
                    "text": str(row["text"] or ""),
                    "author": str(row["author"] or ""),
                    "like_count": int(row["like_count"] or 0)
                })
            
            return quotes
            
    except Exception as e:
        logger.error(f"❌ Помилка отримання цитат: {e}")
        return []

def get_filtered_comments(
    video_id: str,
    sqlite_path: str,
    topic_id: Optional[str] = None,
    sentiment: Optional[str] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Отримує коментарі з фільтрацією за темою та/або тональністю.
    
    Args:
        video_id: ID YouTube відео
        sqlite_path: Шлях до SQLite БД  
        topic_id: Фільтр за темою (опціонально)
        sentiment: Фільтр за тональністю (positive/neutral/negative)
        limit: Максимальна кількість результатів
        
    Returns:
        Список коментарів з метаданими
    """
    try:
        with sqlite3.connect(sqlite_path) as conn:
            # Будуємо WHERE умови
            where_conditions = ["cl.video_id = ?"]
            params = [video_id]
            
            if topic_id:
                where_conditions.append("cl.top_label = ?")
                params.append(topic_id)
                
            if sentiment:
                where_conditions.append("cl.sentiment = ?")
                params.append(sentiment)
            
            where_clause = " AND ".join(where_conditions)
            
            query = f"""
                SELECT 
                    c.text, c.like_count, c.published_at, c.author,
                    cl.top_label, cl.sentiment, cl.labels_json
                FROM comments c
                JOIN comment_labels cl ON c.comment_id = cl.comment_id AND c.video_id = cl.video_id  
                WHERE {where_clause}
                ORDER BY c.like_count DESC, c.published_at DESC
                LIMIT ?
            """
            params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            result = []
            for _, row in df.iterrows():
                # Безпечна обробка likes
                try:
                    likes = int(row["like_count"] or 0)
                except (ValueError, TypeError):
                    likes = 0
                    
                result.append({
                    "text": row["text"],
                    "likes": likes,
                    "author": row["author"] or "Unknown",
                    "date": row["published_at"],
                    "topic": row["top_label"],
                    "sentiment": row["sentiment"] or "neutral"
                })
            
            logger.info(f"🔍 Found {len(result)} filtered comments (topic={topic_id}, sentiment={sentiment})")
            return result
            
    except Exception as e:
        logger.error(f"❌ Помилка отримання коментарів: {e}")
        return []
