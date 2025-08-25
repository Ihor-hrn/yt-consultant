# app/analyze_llm.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, pandas as pd

from app.tools.youtube import fetch_comments, extract_video_id
from app.tools.preprocess import select_fast_batch, preprocess_comments_df
from app.tools.topics_taxonomy import TAXONOMY, ID2NAME
from app.tools.topics_llm import classify_llm_full, aggregate_topics, sample_quotes
from app.tools.classification_db import (
    load_classification_results, 
    get_topic_statistics, 
    get_video_list_with_classification,
    delete_classification_results
)

def run(url: str, sqlite_path: str, limit: int = 1200):
    """
    Головна функція для аналізу YouTube коментарів через LLM класифікацію.
    
    Args:
        url: YouTube URL або video_id
        sqlite_path: Шлях до SQLite кешу
        limit: Максимальна кількість коментарів для обробки
    
    Returns:
        tuple: (df_cls, top) - класифіковані коментарі та топ тем
    """
    print(f"🎬 Аналіз YouTube відео: {url}")
    print(f"📊 Обробка до {limit} коментарів...")
    
    # 1) Завантажуємо/кеш
    print("\n1️⃣ Завантаження коментарів...")
    df_all = fetch_comments(url, sqlite_path=sqlite_path, include_replies=True, max_comments=5000)
    print(f"   Завантажено: {len(df_all)} коментарів")
    
    # 2) Швидкий режим + препроцесинг
    print("\n2️⃣ Препроцесинг...")
    df_fast = select_fast_batch(df_all, mode="top_likes", limit=limit, include_replies=False)
    print(f"   Відібрано для швидкої обробки: {len(df_fast)} коментарів")
    
    df_pre = preprocess_comments_df(
        df_fast, 
        min_chars=12
        # keep_langs=None - залишаємо всі мови для LLM
    )
    print(f"   Після препроцесингу: {len(df_pre)} коментарів")
    
    if len(df_pre) == 0:
        print("⚠️  Немає коментарів для класифікації після препроцесингу!")
        return pd.DataFrame(), pd.DataFrame()
    
    # 3) Повна LLM-класифікація
    print("\n3️⃣ LLM класифікація через OpenRouter...")
    df_cls = classify_llm_full(df_pre, TAXONOMY, text_col="text_clean", batch_size=20)
    print(f"   Класифіковано: {len(df_cls)} коментарів")
    
    # 4) Агрегація: Top-5
    print("\n4️⃣ Аналіз результатів...")
    top = aggregate_topics(df_cls).head(5)
    
    print(f"\n🏆 Топ-{len(top)} тем:")
    print("=" * 60)
    
    for i, (_, row) in enumerate(top.iterrows(), 1):
        topic_name = ID2NAME.get(row['topic_id'], row['topic_id'])
        print(f"{i}. {topic_name}: {int(row['count'])} ({row['share']*100:.1f}%)")
        
        # Показуємо 2 найкращі цитати для цієї теми
        quotes = sample_quotes(df_cls, row["topic_id"], k=2)
        for j, quote in enumerate(quotes, 1):
            text = (quote["text"] or "")[:160].replace("\n", " ")
            print(f"   {j}) {quote['comment_id']}: {text}")
        
        if i < len(top):
            print()  # Порожній рядок між темами
    
    print("=" * 60)
    print(f"✅ Аналіз завершено! Обробано {len(df_cls)} коментарів")
    
    return df_cls, top

def show_saved_results(video_url: str, sqlite_path: str):
    """Показати збережені результати класифікації для відео."""
    video_id = extract_video_id(video_url)
    if not video_id:
        print(f"❌ Не вдалося витягти video_id з {video_url}")
        return
    
    print(f"📊 Завантажуємо збережені результати для відео {video_id}...")
    
    # Завантажуємо результати з БД
    df_saved = load_classification_results(video_id, sqlite_path)
    
    if df_saved.empty:
        print("⚠️  Збережених результатів не знайдено. Спочатку запустіть аналіз.")
        return
    
    # Статистика
    stats = get_topic_statistics(video_id, sqlite_path)
    
    print(f"\n✅ Знайдено збережені результати:")
    print(f"   Всього коментарів в БД: {stats['total_in_db']}")
    print(f"   Класифіковано: {stats['total_comments']} ({stats['classification_coverage']}%)")
    
    if stats['topics']:
        print(f"\n🏆 Топ тем (з БД):")
        for i, topic in enumerate(stats['topics'][:5], 1):
            topic_name = ID2NAME.get(topic['topic'], topic['topic'])
            print(f"{i}. {topic_name}: {topic['count']} ({topic['share_percent']}%)")
    
    # Показуємо зразки коментарів
    classified_df = df_saved[df_saved['topic_top'].notna()]
    if not classified_df.empty:
        print(f"\n💬 Зразки класифікованих коментарів:")
        for topic in stats['topics'][:3]:
            topic_comments = classified_df[classified_df['topic_top'] == topic['topic']].head(2)
            topic_name = ID2NAME.get(topic['topic'], topic['topic'])
            print(f"\n   {topic_name}:")
            for _, comment in topic_comments.iterrows():
                text = (comment['text'] or "")[:150].replace("\n", " ")
                print(f"   • {text}...")

def list_analyzed_videos(sqlite_path: str):
    """Показати список всіх проаналізованих відео."""
    print("📋 Список відео в базі даних:")
    
    videos_df = get_video_list_with_classification(sqlite_path)
    
    if videos_df.empty:
        print("⚠️  База даних порожня. Проаналізуйте якесь відео спочатку.")
        return
    
    print(f"\n{'#':<3} {'Video ID':<15} {'Коментарів':<12} {'Класифіковано':<15} {'Покриття':<10} {'Останній аналіз':<20}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(videos_df.iterrows(), 1):
        coverage = f"{row['classification_coverage']:.1f}%" if row['classification_coverage'] > 0 else "0%"
        last_analysis = row['last_classified_at'][:19] if row['last_classified_at'] else "Ніколи"
        
        print(f"{i:<3} {row['video_id']:<15} {row['total_comments']:<12} {row['classified_comments']:<15} {coverage:<10} {last_analysis:<20}")

def clear_results(video_url: str = None, sqlite_path: str = "./.cache.db"):
    """Очистити результати класифікації."""
    if video_url:
        video_id = extract_video_id(video_url)
        if not video_id:
            print(f"❌ Не вдалося витягти video_id з {video_url}")
            return
        
        success = delete_classification_results(video_id, sqlite_path)
        if success:
            print(f"✅ Результати для відео {video_id} очищено")
        else:
            print(f"❌ Помилка очищення результатів для {video_id}")
    else:
        success = delete_classification_results(None, sqlite_path)
        if success:
            print("✅ Всі результати класифікації очищено")
        else:
            print("❌ Помилка очищення всіх результатів")

def main():
    """CLI точка входу."""
    parser = argparse.ArgumentParser(
        description="Аналіз YouTube коментарів через LLM класифікацію",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  # Новий аналіз
  python -m app.analyze_llm analyze "https://www.youtube.com/watch?v=26riTPNOJbc"
  
  # Показати збережені результати
  python -m app.analyze_llm show "26riTPNOJbc"
  
  # Список проаналізованих відео
  python -m app.analyze_llm list
  
  # Очистити результати
  python -m app.analyze_llm clear --video "26riTPNOJbc"
  python -m app.analyze_llm clear --all
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Доступні команди")
    
    # Команда analyze
    analyze_parser = subparsers.add_parser("analyze", help="Провести новий аналіз відео")
    analyze_parser.add_argument("url", help="YouTube URL або video_id")
    analyze_parser.add_argument("--sqlite", default="./.cache.db", help="Шлях до SQLite кешу")
    analyze_parser.add_argument("--limit", type=int, default=1200, help="Максимальна кількість коментарів")
    
    # Команда show
    show_parser = subparsers.add_parser("show", help="Показати збережені результати")
    show_parser.add_argument("url", help="YouTube URL або video_id")
    show_parser.add_argument("--sqlite", default="./.cache.db", help="Шлях до SQLite кешу")
    
    # Команда list
    list_parser = subparsers.add_parser("list", help="Показати список проаналізованих відео")
    list_parser.add_argument("--sqlite", default="./.cache.db", help="Шлях до SQLite кешу")
    
    # Команда clear
    clear_parser = subparsers.add_parser("clear", help="Очистити результати")
    clear_parser.add_argument("--video", help="URL або video_id для очищення (якщо не вказано - очищає все)")
    clear_parser.add_argument("--all", action="store_true", help="Очистити всі результати")
    clear_parser.add_argument("--sqlite", default="./.cache.db", help="Шлях до SQLite кешу")
    
    args = parser.parse_args()
    
    # Якщо команда не вказана, показуємо help
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "analyze":
            # Перевірка API ключів для аналізу
            if not os.getenv("OPENROUTER_API_KEY"):
                print("❌ Помилка: відсутня змінна середовища OPENROUTER_API_KEY")
                print("💡 Додайте її в .env файл або встановіть через export OPENROUTER_API_KEY=your_key")
                return
            
            if not os.getenv("YOUTUBE_API_KEY"):
                print("⚠️  Попередження: відсутня змінна середовища YOUTUBE_API_KEY")
                print("💡 Можливо, знадобиться для завантаження нових відео")
            
            run(args.url, sqlite_path=args.sqlite, limit=args.limit)
            
        elif args.command == "show":
            show_saved_results(args.url, args.sqlite)
            
        elif args.command == "list":
            list_analyzed_videos(args.sqlite)
            
        elif args.command == "clear":
            if args.all:
                clear_results(None, args.sqlite)
            elif args.video:
                clear_results(args.video, args.sqlite)
            else:
                print("❌ Вкажіть --video <URL> або --all для очищення")
                
    except KeyboardInterrupt:
        print("\n⏹️  Зупинено користувачем")
    except Exception as e:
        print(f"\n❌ Помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
