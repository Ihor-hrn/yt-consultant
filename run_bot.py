#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входу для запуску YouTube Comment Consultant Telegram бота.
"""

import sys
import os

# Додаємо поточну директорію до Python шляху
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    try:
        from app.telegram_bot import main
        import asyncio
        asyncio.run(main())
    except KeyboardInterrupt:
        print("👋 Бот зупинено користувачем")
    except ImportError as e:
        print(f"❌ Помилка імпорту: {e}")
        print("💡 Переконайтесь що встановлені всі залежності: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        import traceback
        traceback.print_exc()
