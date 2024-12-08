import numpy as np
import pandas as pd
import datetime
import sqlite3
import csv

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

X_sc = MinMaxScaler()
y_sc = MinMaxScaler()
X, y = 0, 0
X_t, y_t = (['temp', 'rain_1h', 'snow_1h', 'clouds_all',
            'last_1_hour_traffic', 'last_2_hour_traffic', 'last_3_hour_traffic',
            'last_4_hour_traffic', 'last_5_hour_traffic', 'last_6_hour_traffic',
            'weekday', 'hour', 'day', 'month', 'year', 'weather', 'weather_desc'],
            ['traffic_volume'])

Regr = MLPRegressor(random_state=42, max_iter=1000,
                    activation="relu",
                    solver='adam', alpha=0.001,
                    batch_size='auto', learning_rate="constant",
                    learning_rate_init=0.001, tol=1e-4)


def open_db():
    connection = sqlite3.connect('traffic_lights.db')
    cursor = connection.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS TrafficLights (
        id INTEGER PRIMARY KEY,
        duration_red INTEGER NOT NULL,
        duration_green INTEGER NOT NULL
    )
    ''')

    with open('traffic_lights.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print('a')
            cursor.execute('''
            INSERT INTO TrafficLights (id, duration_red, duration_green) 
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            ''', (row['id'], row['duration_red'], row['duration_green']))

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Programs (
        id INTEGER PRIMARY KEY,
        hour INTEGER NOT NULL,
        red_duration INTEGER NOT NULL,
        green_duration INTEGER NOT NULL,
        FOREIGN KEY (traffic_light_id) REFERENCES TrafficLights(id)
    )
    ''')

    connection.commit()
    connection.close()


def prepare_data():
    data = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)
    last_hours = [1, 2, 3, 4, 5, 6]
    for hour in last_hours:
        data[f'last_{hour}_hour_traffic'] = data['traffic_volume'].shift(hour)
    data.loc[data['holiday'] == 'None', 'holiday'] = 1
    data.loc[data['holiday'] != 'None', 'holiday'] = 0

    data['date_time'] = pd.to_datetime(data['date_time'])
    data['weekday'] = data['date_time'].map(lambda x: x.weekday()+1)
    data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
    data['day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
    data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
    data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))
    weather_f = ['Clear' 'Clouds' 'Rain' 'Drizzle' 'Mist' 'Haze' 'Fog' 'Thunderstorm'
         'Snow' 'Squall' 'Smoke']
    weather_op_f = ['scattered clouds' 'broken clouds' 'overcast clouds' 'sky is clear'
         'few clouds' 'light rain' 'light intensity drizzle' 'mist' 'haze' 'fog'
         'proximity shower rain' 'drizzle' 'moderate rain' 'heavy intensity rain'
         'proximity thunderstorm' 'thunderstorm with light rain'
         'proximity thunderstorm with rain' 'heavy snow' 'heavy intensity drizzle'
         'snow' 'thunderstorm with heavy rain' 'freezing rain' 'shower snow'
         'light rain and snow' 'light intensity shower rain' 'SQUALLS'
         'thunderstorm with rain' 'proximity thunderstorm with drizzle'
         'thunderstorm' 'Sky is Clear' 'very heavy rain'
         'thunderstorm with light drizzle' 'light snow'
         'thunderstorm with drizzle' 'smoke' 'shower drizzle' 'light shower snow'
         'sleet']
    label_encoder = LabelEncoder()
    data['weather'] = label_encoder.fit_transform(data['weather_main'])
    data['weather_desc'] = label_encoder.fit_transform(data['weather_description'])
    data = data.drop(['holiday', 'date_time', 'weather_main', 'weather_description'], axis=1)
    data = data.dropna().reset_index(drop=True)
    data.to_csv('Traffic_data_processed.csv')


def load_data(shuffle=False):
    global X, y

    data = pd.read_csv('Traffic_data_processed.csv')
    X = data[X_t]
    y = data[y_t]
    print(X)
    print(y)


def scale_data():
    global X, y
    global X_sc, y_sc
    X = X_sc.fit_transform(X)
    y = y_sc.fit_transform(y).flatten()


def train_model():
    global Regr
    global X, y
    Regr.fit(X, y)


def predict_model(amount=10, info=False):
    pred = Regr.predict(X[:amount])
    y_pred = y_sc.inverse_transform([pred])
    if info:
        mse = mean_squared_error(y[:amount], Regr.predict(X[:amount]))
        print("Volume: ", y_pred)
        print("Predicted: ", Regr.predict(X[:amount]))
        print("Actual: ", y[:amount])
        print("Mean Squared Error: ", mse)


def predict_model_exact_data(data, id) -> []:
    connection = sqlite3.connect('traffic_lights.db')
    cursor = connection.cursor()
    cursor.execute('DELETE FROM Programs WHERE traffic_light_id = ?', (id,))
    print("Информация по светофору №", id)
    pred = Regr.predict(data)
    y_pred = y_sc.inverse_transform([pred])
    data = X_sc.inverse_transform(data)
    print("Volume: ", y_pred)
    for i in range(len(data)):
        print("В {:02d}:00 ожидается, что через перекресток проедет {:.0f} автомобилей.".format(round(data[i][-6]), y_pred[0][i]), end=' ')
        if y_pred[0][i] < 2000:
            message = ("Небольшое количество автомобилей. Рекомендуем поставить режим уменьшеного зелёного света для "
                       "автомобилей.")
            cursor.execute('''
                        INSERT INTO Programs (traffic_light_id, hour, red_duration, green_duration)
                        VALUES (?, ?, ?, ?)
                    ''', (id, round(data[i][-6]), -1, 1))
        elif y_pred[0][i] < 4000:
            message = ("Умеренное количество автомобилей. Рекомендуем поставить режим обычного зелёного света для "
                       "автомобилей.")
            cursor.execute('''
                        INSERT INTO Programs (traffic_light_id, hour, red_duration, green_duration)
                        VALUES (?, ?, ?, ?)
                    ''', (id, round(data[i][-6]), 0, 0))
        else:
            message = ("Большое количество автомобилей. Рекомендуем поставить режим увеличенного зелёного света для "
                       "автомобилей.")
            cursor.execute('''
                        INSERT INTO Programs (traffic_light_id, hour, red_duration, green_duration)
                        VALUES (?, ?, ?, ?)
                    ''', (id, round(data[i][-6]), 1, -1))
        print(message)

    connection.commit()
    connection.close()


def main():
    global X
    open_db()
    prepare_data()
    load_data()
    scale_data()
    train_model()
    predict_model()
    print(type(X))
    print(X)
    predict_model_exact_data(X[30:51], 1)
    predict_model_exact_data(X[1168:1191], 2)
    predict_model_exact_data(X[2884:2904], 3)


if __name__ == '__main__':
    main()
