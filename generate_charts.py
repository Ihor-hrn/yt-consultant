#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерація графіків для презентації курсової роботи
YouTube Comment Consultant - аналіз коментарів з використанням GenAI
"""

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
from datetime import datetime
import json

# Налаштування matplotlib для українського тексту та гарного вигляду
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_latest_video_data(db_path=".cache.db"):
    """Завантажує дані останнього проаналізованого відео."""
    
    with sqlite3.connect(db_path) as conn:
        # Знаходимо відео з даними тональності
        video_query = """
            SELECT cl.video_id, COUNT(cl.comment_id) as comment_count
            FROM comment_labels cl
            INNER JOIN sentiment_summary ss ON cl.video_id = ss.video_id
            GROUP BY cl.video_id 
            ORDER BY comment_count DESC 
            LIMIT 1
        """
        
        video_result = pd.read_sql_query(video_query, conn)
        if video_result.empty:
            raise ValueError("Немає даних для аналізу в БД")
            
        video_id = video_result.iloc[0]['video_id']
        print(f"📊 Аналізуємо відео: {video_id} ({video_result.iloc[0]['comment_count']} коментарів)")
        
        # Завантажуємо дані коментарів
        comments_query = """
            SELECT 
                cl.comment_id,
                cl.labels_json,
                cl.sentiment,
                cl.top_label,
                c.text as comment_text,
                c.like_count,
                c.published_at
            FROM comment_labels cl
            LEFT JOIN comments c ON cl.comment_id = c.comment_id
            WHERE cl.video_id = ?
        """
        
        comments_df = pd.read_sql_query(comments_query, conn, params=[video_id])
        
        # Завантажуємо статистику тем
        topics_query = """
            SELECT topic_id, count, share
            FROM topics_summary 
            WHERE video_id = ?
            ORDER BY count DESC
        """
        
        topics_df = pd.read_sql_query(topics_query, conn, params=[video_id])
        
        # Завантажуємо статистику тональності
        sentiment_query = """
            SELECT sentiment, count, share
            FROM sentiment_summary 
            WHERE video_id = ?
        """
        
        sentiment_df = pd.read_sql_query(sentiment_query, conn, params=[video_id])
        
    return video_id, comments_df, topics_df, sentiment_df

def create_topics_distribution_chart(topics_df, video_id):
    """Створює круговий графік розподілу категорій."""
    
    # Мапінг назв категорій
    topic_names = {
        'praise': 'Похвала/подяка',
        'critique': 'Критика/незадоволення', 
        'questions': 'Питання/уточнення',
        'suggestions': 'Поради/пропозиції',
        'host_persona': 'Ведучий/персона',
        'content_truth': 'Точність/правдивість',
        'av_quality': 'Звук/відео/монтаж',
        'price_value': 'Ціни/цінність',
        'personal_story': 'Особисті історії',
        'offtopic_fun': 'Офтоп/жарти/меми',
        'toxicity': 'Токсичність/хейт'
    }
    
    # Підготовка даних
    if topics_df.empty:
        print("⚠️ Немає даних про теми")
        return
        
    # Обчислюємо відсотки і беремо топ-8 тем для читабельності
    topics_df['share_percent'] = topics_df['share'] * 100
    top_topics = topics_df.head(8).copy()
    top_topics['topic_name'] = top_topics['topic_id'].map(topic_names)
    top_topics['topic_name'] = top_topics['topic_name'].fillna(top_topics['topic_id'])
    
    # Якщо є інші теми, групуємо їх
    if len(topics_df) > 8:
        others_count = topics_df.iloc[8:]['count'].sum()
        others_row = pd.DataFrame({
            'topic_id': ['others'],
            'topic_name': ['Інше'],
            'count': [others_count],
            'share_percent': [others_count / topics_df['count'].sum() * 100]
        })
        top_topics = pd.concat([top_topics, others_row], ignore_index=True)
    
    # Створення графіку
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_topics)))
    
    wedges, texts, autotexts = ax.pie(
        top_topics['count'], 
        labels=top_topics['topic_name'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 10}
    )
    
    # Покращення вигляду
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    for text in texts:
        text.set_fontsize(9)
    
    ax.set_title(f'Розподіл категорій коментарів\nВідео: {video_id}', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Додаємо легенду з кількістю
    legend_labels = [f'{name} ({count})' for name, count in 
                    zip(top_topics['topic_name'], top_topics['count'])]
    ax.legend(wedges, legend_labels, title="Категорії", 
             loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig('topics_distribution.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Збережено: topics_distribution.png")

def create_sentiment_analysis_chart(sentiment_df, comments_df, video_id):
    """Створює графік аналізу тональності."""
    
    if sentiment_df.empty:
        print("⚠️ Немає даних про тональність")
        return
    
    # Мапінг назв тональності
    sentiment_names = {
        'positive': 'Позитивна',
        'neutral': 'Нейтральна', 
        'negative': 'Негативна'
    }
    
    sentiment_colors = {
        'positive': '#2ecc71',   # Зелений
        'neutral': '#95a5a6',    # Сірий
        'negative': '#e74c3c'    # Червоний
    }
    
    # Підготовка даних
    sentiment_df = sentiment_df.copy()
    sentiment_df['sentiment_name'] = sentiment_df['sentiment'].map(sentiment_names)
    sentiment_df['color'] = sentiment_df['sentiment'].map(sentiment_colors)
    sentiment_df['percentage'] = sentiment_df['share'] * 100
    
    # Створення субплотів
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Аналіз тональності коментарів\nВідео: {video_id}', 
                fontsize=18, fontweight='bold')
    
    # 1. Круговий графік тональності
    wedges, texts, autotexts = ax1.pie(
        sentiment_df['count'],
        labels=sentiment_df['sentiment_name'],
        autopct='%1.1f%%',
        colors=sentiment_df['color'],
        startangle=90,
        textprops={'fontsize': 11}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax1.set_title('Розподіл тональності', fontsize=14, fontweight='bold')
    
    # 2. Стовпчикова діаграма
    bars = ax2.bar(sentiment_df['sentiment_name'], sentiment_df['count'], 
                   color=sentiment_df['color'], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Додаємо значення на стовпчики
    for bar, count in zip(bars, sentiment_df['count']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_title('Кількість коментарів за тональністю', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Кількість коментарів')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Горизонтальна діаграма з відсотками
    y_pos = np.arange(len(sentiment_df))
    bars_h = ax3.barh(y_pos, sentiment_df['percentage'], 
                      color=sentiment_df['color'], alpha=0.8, edgecolor='black', linewidth=1)
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(sentiment_df['sentiment_name'])
    ax3.set_xlabel('Відсоток (%)')
    ax3.set_title('Відсоткові частки тональності', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Додаємо відсотки
    for i, (bar, pct) in enumerate(zip(bars_h, sentiment_df['percentage'])):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 4. Таблиця зі статистикою
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    total_comments = sentiment_df['count'].sum()
    
    for _, row in sentiment_df.iterrows():
        table_data.append([
            row['sentiment_name'],
            f"{row['count']}",
            f"{row['percentage']:.1f}%"
        ])
    
    # Додаємо загальну суму
    table_data.append(['Всього', f"{total_comments}", "100.0%"])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Тональність', 'Кількість', 'Відсоток'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Стилізація таблиці
    for i in range(len(table_data)):
        for j in range(3):
            cell = table[(i+1, j)]
            if i < len(sentiment_df):
                color = sentiment_df.iloc[i]['color']
                cell.set_facecolor(color)
                cell.set_alpha(0.3)
            else:  # Загальна сума
                cell.set_facecolor('#34495e')
                cell.set_alpha(0.7)
                cell.set_text_props(weight='bold', color='white')
    
    # Заголовки таблиці
    for j in range(3):
        cell = table[(0, j)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')
    
    ax4.set_title('Детальна статистика', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('sentiment_analysis.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Збережено: sentiment_analysis.png")

def create_combined_overview_chart(topics_df, sentiment_df, video_id):
    """Створює комбінований огляд для презентації."""
    
    if topics_df.empty or sentiment_df.empty:
        print("⚠️ Недостатньо даних для комбінованого графіку")
        return
    
    # Мапінги
    topic_names = {
        'praise': 'Похвала/подяка',
        'critique': 'Критика/незадоволення', 
        'questions': 'Питання/уточнення',
        'suggestions': 'Поради/пропозиції',
        'host_persona': 'Ведучий/персона',
        'content_truth': 'Точність/правдивість',
        'av_quality': 'Звук/відео/монтаж',
        'price_value': 'Ціни/цінність',
        'personal_story': 'Особисті історії',
        'offtopic_fun': 'Офтоп/жарти/меми',
        'toxicity': 'Токсичність/хейт'
    }
    
    sentiment_names = {
        'positive': 'Позитивна',
        'neutral': 'Нейтральна', 
        'negative': 'Негативна'
    }
    
    sentiment_colors = {
        'positive': '#2ecc71',
        'neutral': '#95a5a6',
        'negative': '#e74c3c'
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Огляд аналізу коментарів - YouTube Comment Consultant\nВідео: {video_id}', 
                fontsize=16, fontweight='bold')
    
    # 1. Топ-6 категорій
    top_topics = topics_df.head(6).copy()
    top_topics['topic_name'] = top_topics['topic_id'].map(topic_names)
    top_topics['topic_name'] = top_topics['topic_name'].fillna(top_topics['topic_id'])
    
    # Скорочуємо довгі назви для кращого відображення
    top_topics['short_name'] = top_topics['topic_name'].str.replace('/', '/\n')
    
    bars1 = ax1.bar(range(len(top_topics)), top_topics['count'], 
                   color=plt.cm.Set3(np.linspace(0, 1, len(top_topics))),
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xticks(range(len(top_topics)))
    ax1.set_xticklabels(top_topics['short_name'], rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Кількість коментарів')
    ax1.set_title('Топ-6 категорій коментарів', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Додаємо значення на стовпчики
    for bar, count in zip(bars1, top_topics['count']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Тональність
    sentiment_df_copy = sentiment_df.copy()
    sentiment_df_copy['sentiment_name'] = sentiment_df_copy['sentiment'].map(sentiment_names)
    sentiment_df_copy['color'] = sentiment_df_copy['sentiment'].map(sentiment_colors)
    sentiment_df_copy['percentage'] = sentiment_df_copy['share'] * 100
    
    wedges, texts, autotexts = ax2.pie(
        sentiment_df_copy['count'],
        labels=sentiment_df_copy['sentiment_name'],
        autopct='%1.1f%%',
        colors=sentiment_df_copy['color'],
        startangle=90,
        textprops={'fontsize': 12}
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax2.set_title('Розподіл тональності', fontsize=14, fontweight='bold')
    
    # Додаємо інформацію про проєкт
    total_comments = sentiment_df['count'].sum()
    info_text = f"""GenAI Technologies Used:
• OpenRouter API (GPT-4o-mini, Gemini-2.5-Flash)  
• Function Calling для автономного агента
• Асинхронна класифікація тем і тональності
• Telegram Bot з AI-агентом

Всього проаналізовано: {total_comments} коментарів"""
    
    fig.text(0.02, 0.02, info_text, fontsize=9, va='bottom', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('overview_presentation.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ Збережено: overview_presentation.png")

def main():
    """Головна функція для генерації всіх графіків."""
    
    print("🎨 Генерація графіків для презентації курсової роботи...")
    print("📊 YouTube Comment Consultant - GenAI для аналізу коментарів\n")
    
    try:
        # Завантажуємо дані
        video_id, comments_df, topics_df, sentiment_df = load_latest_video_data()
        
        print(f"📈 Дані завантажено:")
        print(f"   Коментарі: {len(comments_df)}")
        print(f"   Категорії: {len(topics_df)}")
        print(f"   Тональності: {len(sentiment_df)}\n")
        
        # Генеруємо графіки
        print("🎨 Створення графіків...")
        
        create_topics_distribution_chart(topics_df, video_id)
        create_sentiment_analysis_chart(sentiment_df, comments_df, video_id)
        create_combined_overview_chart(topics_df, sentiment_df, video_id)
        
        print(f"\n🎉 Готово! Створено 3 графіки:")
        print("   📊 topics_distribution.png - Розподіл категорій")
        print("   😊 sentiment_analysis.png - Аналіз тональності") 
        print("   📈 overview_presentation.png - Загальний огляд")
        print(f"\n💡 Графіки готові для вставки в презентацію!")
        
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
