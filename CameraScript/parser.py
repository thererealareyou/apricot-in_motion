import requests
from bs4 import BeautifulSoup
import sqlite3

header = {
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
    'referer': 'https://www.google.com/'
}

def create_database():
    conn = sqlite3.connect('events.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            date TEXT,
            place TEXT,
            link TEXT UNIQUE  -- Уникальный индекс на поле link
        )
    ''')

    conn.commit()
    return conn


def insert_event(conn, name, date, place, link):
    cursor = conn.cursor()

    cursor.execute('SELECT COUNT(*) FROM events WHERE link = ?', (link,))
    exists = cursor.fetchone()[0]

    if exists == 0:
        cursor.execute('''
            INSERT INTO events (name, date, place, link) VALUES (?, ?, ?, ?)
        ''', (name, date, place, link))
        conn.commit()
        print(f"Inserted: {name} - {link}")
    else:
        print(f"Already exists: {link}")


def parse_yandex_afisha():
    conn = create_database()

    link = 'https://afisha.yandex.ru/krasnodar/sport?source=menu'
    r = requests.get(link, headers=header)
    print(r.status_code)
    soup = BeautifulSoup(r.text, features="html.parser")

    for tag in soup.find_all('div', class_="event events-list__item yandex-sans"):
        texts = [li.get_text() for li in tag.find_all('li')]

        name = tag.find('h2').text
        date = texts[0]
        place = texts[1]
        link = tag.find('a')['href']

        insert_event(conn, name, date, place, link)

    conn.close()

if __name__ == '__main__':
    parse_yandex_afisha()