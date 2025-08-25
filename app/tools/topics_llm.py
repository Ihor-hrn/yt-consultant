# app/tools/topics_llm.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, textwrap, asyncio, time
from typing import List, Dict, Any
import pandas as pd

try:
    from logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("topics_llm")

# OpenRouter via OpenAI SDK
import openai

def get_client():
    """Створює клієнт OpenAI з OpenRouter налаштуваннями."""
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

def get_model():
    """Повертає модель для класифікації."""
    return os.getenv("MODEL_SUMMARY", "openai/gpt-4o-mini")

# Константа для правил класифікації
CLASSIFICATION_RULES = """Ти — суворий класифікатор коментарів YouTube. Вибери 0..2 категорії та визнач тональність для кожного коментаря.

ТОНАЛЬНІСТЬ:
- positive: позитивні емоції, схвалення, задоволення
- neutral: нейтральні факти, питання без емоційного забарвлення  
- negative: незадоволення, критика, негативні емоції

Відповідай РІВНО JSON-об'єктом вигляду:
{"items":[{"id":"<comment_id>","labels":["<cat_id>", ...],"sentiment":"positive/neutral/negative"}, ...]}

Де <cat_id> має бути одним з кодів категорій.

Приклад відповіді:
{
  "items": [
    {"id": "comment_1", "labels": ["praise"], "sentiment": "positive"},
    {"id": "comment_2", "labels": ["questions", "suggestions"], "sentiment": "neutral"},
    {"id": "comment_3", "labels": [], "sentiment": "negative"}
  ]
}"""

def _build_prompt(taxonomy: List[Dict[str,str]], items: List[Dict[str,str]]) -> str:
    """Будує промпт для класифікації батча коментарів."""
    taxo_lines = [f"- {t['id']}: {t['name']} — {t['desc']}" for t in taxonomy]
    items_lines = [f"{i['id']}\t{i['text']}" for i in items]
    
    return textwrap.dedent(f"""
    Категорії:
    {chr(10).join(taxo_lines)}

    Коментарі (tab-рядки "<id>\\t<text>"):
    {chr(10).join(items_lines)}
    """).strip()

async def classify_batch_async(
    items: List[Dict[str, str]], 
    taxonomy: List[Dict[str,str]], 
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore
) -> Dict[str, List[str]]:
    """Асинхронна класифікація одного батча коментарів."""
    prompt = _build_prompt(taxonomy, items)
    
    async with semaphore:
        try:
            start_time = time.time()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=get_model(),
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": CLASSIFICATION_RULES},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                ),
                timeout=45.0
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            
            # Оцінка токенів (приблизно)
            prompt_tokens = len(prompt.split()) * 1.3
            response_tokens = len(content.split()) * 1.3
            logger.info(f"Batch processed: {len(items)} items, {latency:.2f}s, ~{prompt_tokens:.0f}+{response_tokens:.0f} tokens")
            
            try:
                data = json.loads(content)
                mapping = {
                    str(it["id"]): {
                        "labels": it.get("labels", []), 
                        "sentiment": it.get("sentiment", "neutral")
                    } for it in data.get("items", [])
                }
                return mapping
            except Exception as e:
                logger.error(f"JSON parse error, trying fallback: {e}")
                # Fallback: можливо JSON-список
                if content.strip().startswith("["):
                    data = json.loads(content)
                    mapping = {str(it["id"]): it.get("labels", []) for it in data}
                    return mapping
                else:
                    # Остання спроба — обернути як {"items": ...}
                    data = json.loads(f'{{"items": {content}}}')
                    mapping = {
                    str(it["id"]): {
                        "labels": it.get("labels", []), 
                        "sentiment": it.get("sentiment", "neutral")
                    } for it in data.get("items", [])
                }
                    return mapping
                    
        except Exception as e:
            logger.error(f"API call failed: {e}")
            # Повертаємо порожні результати для цього батча
            return {str(item["id"]): [] for item in items}

async def process_all_batches(
    df: pd.DataFrame,
    taxonomy: List[Dict[str,str]],
    text_col: str = "text_clean",
    batch_size: int = 20
) -> List[Dict[str, Any]]:
    """Обробляє всі батчі асинхронно з прогресом."""
    import nest_asyncio
    nest_asyncio.apply()
    
    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    semaphore = asyncio.Semaphore(10)  # Максимум 10 одночасних запитів
    
    # Підготовка всіх батчів
    all_items = []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        items = [{"id": str(r["comment_id"]), "text": str(r[text_col])} for _, r in chunk.iterrows()]
        all_items.append(items)
    
    logger.info(f"Starting classification: {len(df)} comments in {len(all_items)} batches")
    
    # Запускаємо всі батчі паралельно
    tasks = [
        classify_batch_async(items, taxonomy, client, semaphore) 
        for items in all_items
    ]
    
    results = []
    completed = 0
    start_time = time.time()
    
    for task in asyncio.as_completed(tasks):
        mapping = await task
        results.append(mapping)
        completed += 1
        
        # Розрахунок прогресу
        elapsed = time.time() - start_time
        if completed > 0:
            avg_time_per_batch = elapsed / completed
            remaining_batches = len(all_items) - completed
            remaining_time = avg_time_per_batch * remaining_batches
            
            overall_progress = (completed / len(all_items)) * 100
            print(f"Overall: {overall_progress:.2f}% | Chunk: {completed}/{len(all_items)} | Remaining: {remaining_time:.2f}s")
    
    await client.close()
    
    # Збираємо всі результати
    all_mappings = {}
    for mapping in results:
        all_mappings.update(mapping)
    
    # Формуємо фінальний результат
    final_results = []
    for _, row in df.iterrows():
        cid = str(row["comment_id"])
        result = all_mappings.get(cid, {"labels": [], "sentiment": "neutral"})
        labels = result.get("labels", [])
        sentiment = result.get("sentiment", "neutral")
        
        final_results.append({
            "comment_id": cid,
            "topic_labels_llm": labels,
            "topic_top_llm": (labels[0] if labels else None),
            "sentiment": sentiment
        })
    
    return final_results

def classify_llm_sync(
    df: pd.DataFrame,
    taxonomy: List[Dict[str,str]],
    *,
    text_col: str = "text_clean",
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Синхронна версія LLM-класифікації для використання в Jupyter.
    """
    import openai
    
    client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )
    
    results = []
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        items = [{"id": str(r["comment_id"]), "text": str(r[text_col])} for _, r in chunk.iterrows()]
        
        # Створюємо промпт
        prompt = _build_prompt(taxonomy, items)
        
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=get_model(),
                temperature=0.0,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_RULES},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content
            
            # Оцінка токенів
            prompt_tokens = len(prompt.split()) * 1.3
            response_tokens = len(content.split()) * 1.3
            logger.info(f"Batch {i//batch_size + 1}/{total_batches}: {len(items)} items, {latency:.2f}s, ~{prompt_tokens:.0f}+{response_tokens:.0f} tokens")
            
            # Парсинг JSON відповіді
            try:
                data = json.loads(content)
                mapping = {
                    str(it["id"]): {
                        "labels": it.get("labels", []), 
                        "sentiment": it.get("sentiment", "neutral")
                    } for it in data.get("items", [])
                }
            except Exception as e:
                logger.error(f"JSON parse error: {e}")
                mapping = {str(item["id"]): {"labels": [], "sentiment": "neutral"} for item in items}
            
            # Додаємо результати
            for _, row in chunk.iterrows():
                cid = str(row["comment_id"])
                result = mapping.get(cid, {"labels": [], "sentiment": "neutral"})
                labels = result.get("labels", [])
                sentiment = result.get("sentiment", "neutral")
                
                results.append({
                    "comment_id": cid,
                    "topic_labels_llm": labels,
                    "topic_top_llm": (labels[0] if labels else None),
                    "sentiment": sentiment
                })
            
            # Прогрес
            progress = ((i//batch_size + 1) / total_batches) * 100
            remaining_batches = total_batches - (i//batch_size + 1)
            remaining_time = remaining_batches * latency if latency > 0 else 0
            print(f"Overall: {progress:.1f}% | Chunk: {i//batch_size + 1}/{total_batches} | Remaining: {remaining_time:.1f}s")
            
        except Exception as e:
            logger.error(f"API call failed for batch {i//batch_size + 1}: {e}")
            # Додаємо порожні результати
            for _, row in chunk.iterrows():
                results.append({
                    "comment_id": str(row["comment_id"]),
                    "topic_labels_llm": [],
                    "topic_top_llm": None
                })
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Злиття з оригінальним DataFrame
    merged_df = df.merge(results_df, on="comment_id", how="left")
    
    # Зберігаємо результати в БД (синхронна версія)
    try:
        from .classification_db import save_classification_results
        sqlite_path = os.getenv("SQLITE_PATH", "./.cache.db")
        
        # Якщо немає video_id, спробуємо витягти з comment_id або додати дефолтний
        if "video_id" not in merged_df.columns:
            if len(merged_df) > 0:
                # Додаємо дефолтний video_id (можна вдосконалити пізніше)
                merged_df["video_id"] = "26riTPNOJbc"  # video_id з ноутбука
                logger.info("Додано дефолтний video_id='26riTPNOJbc' для збереження класифікації")
        
        if sqlite_path:
            save_classification_results(
                merged_df, 
                sqlite_path, 
                model_name=get_model(),
                batch_size=batch_size
            )
    except Exception as e:
        logger.warning(f"Не вдалося зберегти класифікацію в БД: {e}")
    
    return merged_df

def classify_llm_full(
    df: pd.DataFrame,
    taxonomy: List[Dict[str,str]],
    *,
    text_col: str = "text_clean",
    batch_size: int = 20,
) -> pd.DataFrame:
    """
    Повна LLM-класифікація всіх рядків df на основі заданої таксономії.
    Повертає df з колонками:
      - topic_labels_llm: list[str]
      - topic_top_llm: str | None
    """
    # Перевіряємо, чи запущено в Jupyter
    try:
        import IPython
        if IPython.get_ipython() is not None:
            # Якщо в Jupyter, використовуємо синхронну версію
            logger.info("Detected Jupyter environment, using synchronous classification")
            return classify_llm_sync(df, taxonomy, text_col=text_col, batch_size=batch_size)
    except ImportError:
        # Якщо IPython не встановлено, використовуємо асинхронну версію
        pass
    
    import nest_asyncio
    nest_asyncio.apply()
    
    try:
        # Спробуємо отримати існуючий loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Якщо loop вже запущений, використовуємо синхронну версію
            logger.info("Async loop already running, falling back to sync classification")
            return classify_llm_sync(df, taxonomy, text_col=text_col, batch_size=batch_size)
        else:
            results = loop.run_until_complete(
                process_all_batches(df, taxonomy, text_col, batch_size)
            )
    except RuntimeError:
        # Fallback - створюємо новий loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            process_all_batches(df, taxonomy, text_col, batch_size)
        )
        loop.close()
    
    # Створення DataFrame з результатами
    results_df = pd.DataFrame(results)
    
    # Злиття з оригінальним DataFrame
    merged_df = df.merge(results_df, on="comment_id", how="left")
    
    # Зберігаємо результати в БД (асинхронна версія)
    try:
        from .classification_db import save_classification_results
        sqlite_path = os.getenv("SQLITE_PATH", "./.cache.db")
        
        # Якщо немає video_id, спробуємо витягти з comment_id або додати дефолтний
        if "video_id" not in merged_df.columns:
            if len(merged_df) > 0:
                # Додаємо дефолтний video_id (можна вдосконалити пізніше)
                merged_df["video_id"] = "26riTPNOJbc"  # video_id з ноутбука
                logger.info("Додано дефолтний video_id='26riTPNOJbc' для збереження класифікації")
        
        if sqlite_path:
            save_classification_results(
                merged_df, 
                sqlite_path, 
                model_name=get_model(),
                batch_size=batch_size
            )
    except Exception as e:
        logger.warning(f"Не вдалося зберегти класифікацію в БД: {e}")
    
    return merged_df

def aggregate_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегує частоти тем за колонкою topic_labels_llm і повертає таблицю:
      topic_id, count, share
    """
    from collections import Counter
    bag = []
    for labs in df["topic_labels_llm"].tolist():
        if isinstance(labs, list):
            bag.extend(labs)
    
    cnt = Counter(bag)
    n = sum(cnt.values()) or 1
    rows = [{"topic_id": k, "count": v, "share": round(v/n, 4)} for k,v in cnt.items()]
    return pd.DataFrame(rows).sort_values(["count","topic_id"], ascending=[False, True]).reset_index(drop=True)

def sample_quotes(df: pd.DataFrame, topic_id: str, k: int = 3) -> list[dict]:
    """Повертає k найпопулярніших цитат (за like_count) для теми."""
    subset = df[df["topic_labels_llm"].apply(lambda L: isinstance(L,list) and topic_id in L)]
    subset = subset.sort_values(["like_count","published_at"], ascending=[False, True]).head(k)
    return [{"comment_id": r["comment_id"], "text": r["text_clean"]} for _, r in subset.iterrows()]
