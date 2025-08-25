# 🤖 YouTube Comment Consultant

**AI-агент з function calling для аналізу коментарів YouTube**

Персона: автономний, ввічливий консультант з підтримкою function calling.

## 🎯 Опис

YouTube Comment Consultant — це **автономний AI-агент** у Telegram, який використовує **function calling** для самостійного прийняття рішень. Агент аналізує коментарі YouTube, відповідає **виключно на основі коментарів** (як NotebookLM) і самостійно вирішує які інструменти використовувати.

## 🚀 Швидкий старт

### 1. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 2. Налаштування .env

Створіть `.env` файл на основі `.env.example`:

```bash
# YouTube API для завантаження коментарів
YOUTUBE_API_KEY=your_youtube_api_key_here

# OpenRouter API для LLM класифікації  
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Модель для класифікації (за замовчуванням: openai/gpt-4o-mini)
MODEL_SUMMARY=openai/gpt-4o-mini

# Модель для AI-агента з function calling (підтримує tools)
AGENT_MODEL=google/gemini-2.5-flash

# Telegram Bot Token (отримайте у @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
```

### 3. Запуск Telegram бота

```bash
# Запуск бота
python run_bot.py

# Або альтернативно
python -m app.telegram_bot
```

### 4. Використання AI-агента

1. **Знайдіть вашого бота в Telegram** за username
2. **Напишіть `/start`** для знайомства з агентом
3. **Просто спілкуйтесь природно:**
   - Надішліть YouTube URL → агент автоматично проаналізує
   - "Що думають про звук?" → знайде відповідь у коментарях  
   - "Покажи топ теми" → дасть статистику
   - "Деталі про критику" → покаже цитати

**Особливість:** Агент самостійно вирішує які інструменти використати!

### Приклад роботи в терміналі:

```
🤖 Запускаю YouTube Comment Consultant AI-агента...
🔧 Модель агента: google/gemini-2.5-flash
🤖 Бот @OnlyFansAI111bot (ID: 7981366461) готовий до роботи!

📨 Processing message from user 12345: https://www.youtube.com/watch?v=dQw4w9WgXcQ
🎬 Detected YouTube video: dQw4w9WgXcQ
🧠 Calling google/gemini-2.5-flash for initial processing...
🎯 Agent decided: use tools
🔧 Agent requested 1 tool calls
⚙️ Executing tool 1/1...
🔧 Executing tool: analyze_video with args: {'url_or_id': 'dQw4w9WgXcQ'}
📺 Starting video analysis for: dQw4w9WgXcQ...
✅ Video analysis completed: 850 comments, 5 topics
💾 Saved video_id dQw4w9WgXcQ for user 12345

[Наступне повідомлення: "Що думають про якість звуку?"]
📨 Processing message from user 12345: Що думають про якість звуку?
🎬 Використовую контекст відео: dQw4w9WgXcQ
🧠 Додаю контекст поточного відео: dQw4w9WgXcQ
🧠 Calling google/gemini-2.5-flash for initial processing...
🎯 Agent decided: use tools
🔧 Agent requested 1 tool calls
🔍 Searching comments for question: Що думають про якість звуку?... (video: dQw4w9WgXcQ)
✅ Found 3 relevant comments
```

### 5. CLI для розробників (опціонально)

```bash
# Новий аналіз відео
python -m app.analyze_llm analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# З налаштуваннями
python -m app.analyze_llm analyze "VIDEO_ID" --limit 800 --sqlite ./cache.db

# Показати збережені результати
python -m app.analyze_llm show "VIDEO_ID"

# Список проаналізованих відео
python -m app.analyze_llm list

# Очищення результатів
python -m app.analyze_llm clear --video "VIDEO_ID"  # одне відео
python -m app.analyze_llm clear --all              # всі результати
```

## 📊 Таксономія тем

Система класифікує коментарі за 11 категоріями:

- **Похвала/подяка** - Схвальні відгуки, компліменти
- **Критика/незадоволення** - Негативні реакції, претензії
- **Питання/уточнення** - Прохання пояснити, дати лінк
- **Поради/пропозиції** - Ідеї для покращення, рекомендації
- **Ведучий/персона** - Коментарі про ведучого, манеру подачі
- **Точність/правдивість** - Сумніви в достовірності, фактичні зауваги
- **Звук/відео/монтаж** - Технічна якість контенту
- **Ціни/цінність** - Обговорення ціни, вигоди, окупності
- **Особисті історії** - Досвід глядача, приватні приклади
- **Офтоп/жарти/меми** - Гумор, меми, флуд не по темі
- **Токсичність/хейт** - Образи, мова ворожнечі

## 🧠 AI-агент з Function Calling

### Автономні можливості

- **🎯 Самостійне прийняття рішень:** агент аналізує запит і вибирає потрібні інструменти
- **📊 Автоматичний аналіз:** YouTube URL → парсинг → класифікація → збереження  
- **🔍 Інтелектуальний пошук:** питання → пошук у коментарях → структурована відповідь
- **📈 Контекстні деталі:** автоматично показує статистику, цитати, тренди

### Доступні інструменти агента

1. **`analyze_video`** — повний аналіз YouTube відео
2. **`search_comments`** — пошук релевантних коментарів  
3. **`get_analysis_data`** — отримання збережених результатів
4. **`get_topic_details`** — деталі конкретних тем з цитатами

### Підтримувані моделі (з function calling)

- **`google/gemini-2.5-flash`** ⭐ — рекомендується (швидко + дешево)
- **`openai/gpt-4o`** — максимальна якість
- **`openai/gpt-4.1`** — альтернатива

### Персона агента

- **Принцип NotebookLM:** відповіді **виключно на основі коментарів**
- **Чесність:** якщо даних немає — прямо про це каже
- **Автономність:** самостійно вирішує які дії потрібні  
- **Багатомовність:** українська за замовчуванням, адаптується до користувача
- **🧠 Пам'ять:** запам'ятовує останнє відео і відповідає на питання без повторного аналізу

### Контекстна пам'ять

**Новинка!** Агент тепер запам'ятовує останнє проаналізоване відео:

✅ **Після аналізу відео:**
- "Що думають про звук?" → автоматично шукає у коментарях цього відео
- "Покажи топ теми" → показує статистику для цього відео  
- "Деталі про критику" → знаходить цитати з критикою

❌ **НЕ просить повторно URL** для питань про те ж відео

🔄 **Новий аналіз:** тільки коли ви надсилаєте нове YouTube посилання

## 🔧 Структура проєкту

```
app/
├── tools/
│   ├── youtube.py               # Завантаження коментарів з YouTube API
│   ├── preprocess.py            # Препроцесинг тексту
│   ├── topics_taxonomy.py       # Фіксована таксономія 11 тем
│   ├── topics_llm.py            # LLM-класифікація через OpenRouter
│   ├── classification_db.py     # Збереження результатів в SQLite
│   └── analyze_video_tool.py    # Високорівневий інструмент
├── agent_system.py              # 🆕 AI-агент з function calling
├── telegram_bot.py              # 🆕 Telegram бот з агентом
└── analyze_llm.py               # CLI для розробників
run_bot.py                       # 🆕 Точка входу для запуску бота
```

## 🔧 Налаштування Telegram бота

### 1. Створення бота

1. Знайдіть [@BotFather](https://t.me/BotFather) в Telegram
2. Напишіть `/newbot`
3. Вкажіть назву бота (наприклад, "YouTube Comment Consultant")
4. Вкажіть username бота (має закінчуватись на `bot`)
5. Скопіюйте отриманий **Bot Token**

### 2. Налаштування команд (опціонально)

Відправте [@BotFather](https://t.me/BotFather) команду `/setcommands` і вкажіть:

```
start - Привітання та інструкції
analyze - Проаналізувати YouTube відео
ask - Знайти відповідь у коментарях
```

## 💾 Збереження результатів

Усі результати аналізу автоматично зберігаються в SQLite БД:

### Таблиці

- **`analyses`** — інформація про проведені аналізи
- **`comment_labels`** — результати класифікації та тональності для кожного коментаря  
- **`topics_summary`** — зведення тем з кількістю та частками
- **`sentiment_summary`** — 🎭 статистика тональності (позитивна/нейтральна/негативна)
- **`comments`** — кеш завантажених коментарів (з youtube.py)
- **`classification_results`** — зворотна сумісність

### Особливості

- **Автоматичне кешування**: повторні аналізи використовують збережені дані
- **Історія аналізів**: бот "бачить" результати попередніх класифікацій
- **Швидкий доступ**: Q&A та деталі тем працюють з БД без повторних запитів до LLM

## 💻 Використання в коді

```python
from app.tools.youtube import fetch_comments
from app.tools.preprocess import select_fast_batch, preprocess_comments_df
from app.tools.topics_taxonomy import TAXONOMY, ID2NAME
from app.tools.topics_llm import classify_llm_full, aggregate_topics, sample_quotes
from app.tools.analyze_video_tool import analyze_video_tool, aggregate_sentiment
from app.tools.classification_db import load_classification_results, get_topic_statistics

# Завантаження коментарів (автоматично кешується)
df_all = fetch_comments("VIDEO_URL", sqlite_path="./.cache.db")

# Препроцесинг (залишаємо ВСІ мови для LLM)
df_fast = select_fast_batch(df_all, mode="top_likes", limit=1200)
df_pre = preprocess_comments_df(df_fast, min_chars=12)  # keep_langs=None

# LLM-класифікація (автоматично зберігається в БД)
df_classified = classify_llm_full(df_pre, TAXONOMY)

# Аналіз результатів
top_topics = aggregate_topics(df_classified).head(5)
sentiment_stats = aggregate_sentiment(df_classified)  # 🎭 Нова функція тональності!
quotes = sample_quotes(df_classified, topic_id="praise", k=3)

# 🆕 Робота зі збереженими результатами
video_id = "26riTPNOJbc"

# Завантаження збережених результатів
df_saved = load_classification_results(video_id, "./.cache.db")

# Статистика по відео
stats = get_topic_statistics(video_id, "./.cache.db")
print(f"Класифіковано: {stats['total_comments']} коментарів")
print(f"Топ тема: {stats['topics'][0]['topic']}")
```

## 🧪 Jupyter Notebook

Демонстрацію роботи можна побачити в `.ipynb` файлі в корені проекту.

## ⚡ Оптимізація

- **Batch розмір**: 10-20 коментарів на запит до LLM
- **Concurrency**: До 10 одночасних запитів до OpenRouter
- **Кешування**: SQLite для збереження завантажених коментарів
- **Фільтрація**: Обробка лише найпопулярніших коментарів

## 🛡️ Безпека та обмеження

### Безпека

- **Ніяких медичних/фінансових порад**: бот уникає надання консультацій в сенситивних сферах
- **Без особистих даних**: користувачі не називаються поіменно
- **Нейтралізація токсичності**: при токсичних темах бот пом'якшує тон у чернетках
- **Обмеження джерел**: відповіді тільки на основі коментарів YouTube

### Обмеження

- **Приватні відео**: якщо відео приватне або без коментарів — зрозумілі повідомлення про помилку
- **Квоти API**: при перевищенні квот YouTube/OpenRouter — діагностика та корисні повідомлення
- **Мова**: за замовчуванням українська, але відповідає мовою користувача
- **Розмір**: аналіз до ~1200 коментарів в швидкому режимі (для економії токенів)

## ⚡ Продуктивність

### LLM виклики

- **Batch розмір**: 10-25 коментарів на запит
- **Temperature**: 0.0 для класифікації, 0.4-0.5 для чернеток
- **Токени**: коментарі обрізаються до ~500 символів
- **Паралельність**: до 10 одночасних запитів до OpenRouter
- **JSON формат**: строгий `response_format={"type":"json_object"}`

### Логування

Бот логує кожен батч: кількість прикладів, орієнтовні токени, latency (сек).
У відповідь користувачу — стисла зведена строка: скільки коментарів зібрано/використано.

## 🎓 Переваги AI-агента

- 🧠 **Function Calling**: агент самостійно вибирає потрібні інструменти
- 🎯 **Автономність**: не потребує команд — просто спілкуйтесь природно
- 📚 **Принцип NotebookLM**: відповіді виключно на основі коментарів
- 🚫 **Не вигадує**: чесно каже коли даних недостатньо
- 💬 **Природна комунікація**: YouTube URL → автоматичний аналіз
- 🇺🇦 **Багатомовність**: підтримка української та інших мов
- ⚡ **Швидкість**: Gemini 2.5 Flash для оптимальної ціни/якості
- 🛡️ **Безпека**: обмеження на медичні/фінансові поради
- 🔄 **Кешування**: повторні запити використовують збережені дані
- 📊 **Структуровані відповіді**: статистика, цитати, категорії
