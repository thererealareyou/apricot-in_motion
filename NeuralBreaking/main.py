import config
import telebot
import os
import sqlite3
from sqlite3 import Error
from datetime import datetime
import listener, rephraser
from pydub import AudioSegment

bot = telebot.TeleBot(config.token)
db = sqlite3.connect('articles.db', check_same_thread=False)
curr = db.cursor()


def create_table():
    """Создает таблицу articles, если она не существует."""
    try:
        curr.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_appearance TEXT NOT NULL,
                file_link TEXT NOT NULL,
                text TEXT NOT NULL,
                short_text TEXT NOT NULL
            )
        ''')
        db.commit()
        print("Таблица 'articles' успешно создана или уже существует.")
    except Error as e:
        print(f'Ошибка при создании таблицы: {e}')

def clear_table():
    try:
        curr.execute('DELETE FROM articles')
        db.commit()
        print("Таблица 'articles' успешно очищена.")
    except Error as e:
        print(f'Ошибка при очистке таблицы: {e}')

@bot.message_handler(content_types=["text"])
def handle_text_message(message):
    print('Получено текстовое сообщение!')

@bot.message_handler(content_types=["voice"])
def handle_voice_message(message):
    try:
        print('Получил голосовое сообщение!')
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        if not os.path.exists('voice_messages'):
            os.makedirs('voice_messages')

        ogg_file_path = f'voice_messages/{message.voice.file_id}.ogg'
        with open(ogg_file_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        wav_file_path = f'voice_messages/{message.voice.file_id}.wav'
        audio = AudioSegment.from_ogg(ogg_file_path)
        audio.export(wav_file_path, format='wav')

        os.remove(ogg_file_path)

        text = listener.audio_to_text(wav_file_path)
        print('Запускаю модель...')
        rephrased_text = rephraser.summarize_text(text)

        date_appearance = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        sql = '''INSERT INTO articles (date_appearance, file_link, text, short_text)
                 VALUES (?, ?, ?, ?)'''
        curr.execute(sql, (date_appearance, wav_file_path, text, rephrased_text))
        db.commit()
        print('Данные успешно записаны в базу данных.')

    except Exception as e:
        print(f'Произошла ошибка: {e}')


if __name__ == '__main__':
    if True :
        clear_table()
    create_table()
    try:
        bot.infinity_polling()
    except KeyboardInterrupt:
        print("Бот остановлен.")
    finally:
        db.close()