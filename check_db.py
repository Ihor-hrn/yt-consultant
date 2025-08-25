import sqlite3

conn = sqlite3.connect('.cache.db')

# Перевіряємо таблиці
tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("Tables:", tables)

# Коментарі
if 'comments' in tables:
    count = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]
    print(f"Comments: {count}")
    
    if count > 0:
        sample = conn.execute("SELECT video_id, comment_id, text FROM comments LIMIT 3").fetchall()
        for row in sample:
            print(f"  {row[0]}: {row[2][:50]}...")

# Класифікація
if 'classification_results' in tables:
    count = conn.execute("SELECT COUNT(*) FROM classification_results").fetchone()[0]
    print(f"Classification results: {count}")
else:
    print("classification_results table not found")

conn.close()
