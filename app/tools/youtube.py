# app/tools/youtube.py
# -*- coding: utf-8 -*-
"""
YouTube comments fetcher (M1: Integration, P0)
- Отримує коментарі (включно з відповідями) через YouTube Data API v3
- Підтримує пагінацію, "швидкий режим" через max_comments/max_pages
- Повертає pandas.DataFrame і (опційно) кешує у SQLite
"""

from __future__ import annotations
import os
import re
import time
import sqlite3
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ---------- Завантаження змінних середовища з .env ----------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Якщо python-dotenv не встановлено, спробуємо завантажити .env вручну
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# ---------- Логер (використаємо ваш logger, якщо є) ----------
try:
    from logger import logger  # ваш існуючий логер
except Exception:  # fallback — базовий
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("yt")

# ---------- Допоміжні функції ----------
VIDEO_ID_RE = re.compile(
    r"""
    (?:youtu\.be/|youtube\.com/(?:watch\?v=|embed/|shorts/|live/))
    ([0-9A-Za-z_-]{11})
    """,
    re.VERBOSE,
)

def extract_video_id(url_or_id: str) -> Optional[str]:
    """
    Витягує 11-символьний video_id з повного URL або повертає рядок, якщо він уже схожий на id.
    Підтримка: youtu.be/, watch?v=, embed/, shorts/, live/
    """
    s = (url_or_id or "").strip()
    if not s:
        return None
    # Якщо вже схоже на video_id
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s

    # Спершу спробуємо через query параметр v=
    if "watch?" in s and "v=" in s:
        from urllib.parse import urlparse, parse_qs
        q = parse_qs(urlparse(s).query)
        v = q.get("v", [None])[0]
        if v and re.fullmatch(r"[0-9A-Za-z_-]{11}", v):
            return v

    # Інакше — через загальну регулярку
    m = VIDEO_ID_RE.search(s)
    return m.group(1) if m else None


def _ensure_sqlite(conn: sqlite3.Connection) -> None:
    """Створює таблицю comments при необхідності."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comments (
            video_id TEXT NOT NULL,
            comment_id TEXT PRIMARY KEY,
            parent_id TEXT,
            author TEXT,
            author_channel_id TEXT,
            text TEXT,
            like_count INTEGER,
            reply_count INTEGER,
            published_at TEXT,
            updated_at TEXT,
            is_reply INTEGER,
            fetched_at TEXT
        )
        """
    )
    conn.commit()


def _upsert_comments(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    """UPSERT у SQLite (ON CONFLICT REPLACE за comment_id)."""
    if not rows:
        return
    _ensure_sqlite(conn)
    conn.executemany(
        """
        INSERT INTO comments (
            video_id, comment_id, parent_id, author, author_channel_id, text,
            like_count, reply_count, published_at, updated_at, is_reply, fetched_at
        )
        VALUES (
            :video_id, :comment_id, :parent_id, :author, :author_channel_id, :text,
            :like_count, :reply_count, :published_at, :updated_at, :is_reply, :fetched_at
        )
        ON CONFLICT(comment_id) DO UPDATE SET
            parent_id=excluded.parent_id,
            author=excluded.author,
            author_channel_id=excluded.author_channel_id,
            text=excluded.text,
            like_count=excluded.like_count,
            reply_count=excluded.reply_count,
            published_at=excluded.published_at,
            updated_at=excluded.updated_at,
            is_reply=excluded.is_reply,
            fetched_at=excluded.fetched_at
        """,
        rows,
    )
    conn.commit()


# ---------- Основна функція ----------
def fetch_comments(
    url_or_id: str,
    *,
    api_key: Optional[str] = None,
    order: str = "relevance",           # 'relevance' | 'time'
    include_replies: bool = True,
    max_results_per_page: int = 100,    # 1..100
    max_pages: Optional[int] = None,    # обмеження сторінок
    max_comments: Optional[int] = None, # обмеження загальної кількості
    text_format: str = "plainText",     # 'plainText' | 'html'
    sleep_between_pages: float = 0.0,   # сек. між сторінками (якщо боїшся квот)
    sqlite_path: Optional[str] = None,  # якщо задано — збережемо кеш
    quota_user: Optional[str] = None,   # ідентифікатор для quota групування
) -> pd.DataFrame:
    """
    Стягує коментарі для відео. Повертає DataFrame зі стовпчиками:
    ['video_id','comment_id','parent_id','author','author_channel_id','text',
     'like_count','reply_count','published_at','updated_at','is_reply']
    """

    t0 = time.perf_counter()
    video_id = extract_video_id(url_or_id)
    if not video_id:
        raise ValueError("Не вдалося визначити video_id. Перевірте посилання або ID.")

    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("Не задано api_key і змінну середовища YOUTUBE_API_KEY.")

    logger.info(f"📥 Fetch comments for video_id={video_id} (order={order})")

    # Ініціалізація клієнта
    client_kwargs = {"developerKey": api_key}
    if quota_user:
        client_kwargs["quotaUser"] = quota_user  # корисно для квот
    youtube = build("youtube", "v3", **client_kwargs)

    rows: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None
    page = 0

    try:
        while True:
            if max_pages is not None and page >= max_pages:
                logger.info(f"⏹️ max_pages={max_pages} досягнуто")
                break

            page += 1
            t_page = time.perf_counter()
            request = youtube.commentThreads().list(
                part="snippet,replies",
                videoId=video_id,
                maxResults=max_results_per_page,
                order=order,
                pageToken=next_page_token,
                textFormat=text_format,  # повертає plainText у textDisplay
            )
            response = request.execute()
            items = response.get("items", [])
            fetched_this_page = 0

            for item in items:
                thread_id = item.get("id")
                snip = item["snippet"]
                top = snip["topLevelComment"]["snippet"]

                # top-level
                rows.append({
                    "video_id": video_id,
                    "comment_id": item["snippet"]["topLevelComment"]["id"],
                    "parent_id": None,
                    "author": top.get("authorDisplayName"),
                    "author_channel_id": (top.get("authorChannelId") or {}).get("value"),
                    "text": top.get("textDisplay") or top.get("textOriginal"),
                    "like_count": int(top.get("likeCount", 0) or 0),
                    "reply_count": int(snip.get("totalReplyCount", 0) or 0),
                    "published_at": top.get("publishedAt"),
                    "updated_at": top.get("updatedAt"),
                    "is_reply": 0,
                    "fetched_at": pd.Timestamp.utcnow().isoformat(),
                })
                fetched_this_page += 1

                if include_replies and "replies" in item:
                    for reply in item["replies"].get("comments", []):
                        rs = reply["snippet"]
                        rows.append({
                            "video_id": video_id,
                            "comment_id": reply["id"],
                            "parent_id": thread_id,
                            "author": rs.get("authorDisplayName"),
                            "author_channel_id": (rs.get("authorChannelId") or {}).get("value"),
                            "text": rs.get("textDisplay") or rs.get("textOriginal"),
                            "like_count": int(rs.get("likeCount", 0) or 0),
                            "reply_count": 0,
                            "published_at": rs.get("publishedAt"),
                            "updated_at": rs.get("updatedAt"),
                            "is_reply": 1,
                            "fetched_at": pd.Timestamp.utcnow().isoformat(),
                        })
                        fetched_this_page += 1

                # Обмеження по кількості
                if max_comments is not None and len(rows) >= max_comments:
                    logger.info(f"⏹️ Досягнуто max_comments={max_comments}")
                    break

            # лог по сторінці
            dt_page = time.perf_counter() - t_page
            logger.info(f"📄 Page {page}: +{fetched_this_page} comments, {dt_page:.2f}s")

            if max_comments is not None and len(rows) >= max_comments:
                break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

            if sleep_between_pages > 0:
                time.sleep(sleep_between_pages)

    except HttpError as e:
        logger.error(f"❌ YouTube API error: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")

    # До DataFrame
    df = pd.DataFrame(rows, columns=[
        "video_id","comment_id","parent_id","author","author_channel_id","text",
        "like_count","reply_count","published_at","updated_at","is_reply","fetched_at"
    ])

    # (Опціонально) кеш у SQLite
    if sqlite_path:
        try:
            with sqlite3.connect(sqlite_path) as conn:
                _ensure_sqlite(conn)
                _upsert_comments(conn, df.to_dict(orient="records"))
            logger.info(f"💾 Saved {len(df)} comments into SQLite: {sqlite_path}")
        except Exception as e:
            logger.error(f"⚠️ Failed to write SQLite cache: {e}")

    total_dt = time.perf_counter() - t0
    logger.info(f"✅ Done. fetched={len(df)} in {total_dt:.2f}s (video_id={video_id})")
    return df


# ---------- Простий CLI-тест ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch YouTube comments")
    parser.add_argument("url_or_id", help="YouTube URL or video_id")
    parser.add_argument("--api_key", default=os.getenv("YOUTUBE_API_KEY"))
    parser.add_argument("--order", default="relevance", choices=["relevance", "time"])
    parser.add_argument("--include_replies", action="store_true")
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument("--max_comments", type=int, default=2000)
    parser.add_argument("--sqlite", type=str, default=None)
    args = parser.parse_args()

    df = fetch_comments(
        args.url_or_id,
        api_key=args.api_key,
        order=args.order,
        include_replies=args.include_replies,
        max_comments=args.max_comments,
        max_pages=args.max_pages,
        sqlite_path=args.sqlite,
    )
    print(df.head())
    print(f"Total: {len(df)}")
