# -*- coding: utf-8 -*-
"""
«ДермаСкрин» — веб-прототип помощника по первичной оценке риска
кожных новообразований. Flask + TensorFlow (модель skin_model.keras).

ВАЖНО: это учебный прототип, НЕ медицинский прибор. Он не ставит диагноз,
а лишь оценивает визуальное сходство с опасными образованиями и советует
обратиться к дерматологу.

«Риск-скор» (0–100) — процентильный ранг вероятности «высокого риска»
относительно эталонного распределения на валидационной выборке:
значение 80 означает «фото выглядит подозрительнее, чем 80 % эталонных снимков».

Запуск:  python app.py           (откроется http://127.0.0.1:5000)
"""

import os
import io
import json
import base64
import numpy as np
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, render_template
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'skin_model.keras')
CONFIG_PATH = os.path.join(BASE_DIR, 'model', 'config.json')

with open(CONFIG_PATH, encoding='utf-8') as f:
    CONFIG = json.load(f)

IMG_SIZE = tuple(CONFIG['img_size'])
REF_PROBS = np.array(CONFIG.get('reference_probs', []), dtype=np.float32)

# Границы зон на шкале «риск-скор» (процентили)
LOW_MAX = 40.0
MED_MAX = 70.0

RECOMMENDATIONS = {
    'low': ('Выраженных признаков опасного образования не обнаружено. '
            'Рекомендуется обычное наблюдение и самопроверка по правилу ABCDE.'),
    'medium': ('Пограничный случай. Рекомендуется показать образование '
               'дерматологу в ближайшее время.'),
    'high': ('Высокая визуальная схожесть с опасным образованием. '
             'Настоятельно рекомендуется записаться к дерматологу без промедления.'),
}

ABCDE = [
    ('A', 'Asymmetry — асимметрия', 'половинки родинки несимметричны'),
    ('B', 'Border — края', 'края неровные, «рваные», нечёткие'),
    ('C', 'Color — цвет', 'цвет неоднородный, несколько оттенков'),
    ('D', 'Diameter — диаметр', 'диаметр больше 6 мм'),
    ('E', 'Evolution — изменения', 'родинка меняет форму, цвет или размер'),
]

print('Загрузка модели...')
model = keras.models.load_model(MODEL_PATH)
print('Модель загружена.')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # максимум 16 МБ


def preprocess_for_model(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def make_thumbnail_b64(image_bytes, size=256):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=85)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def risk_score(prob):
    """Процентильный ранг вероятности (0–100) относительно эталона."""
    if REF_PROBS.size == 0:
        return float(prob) * 100.0
    return float((REF_PROBS <= prob).mean()) * 100.0


def classify(score):
    if score < LOW_MAX:
        zone = 'low'
    elif score < MED_MAX:
        zone = 'medium'
    else:
        zone = 'high'
    label = {'low': 'Низкий риск', 'medium': 'Средний риск (пограничная зона)',
             'high': 'Высокий риск'}[zone]
    color = {'low': 'green', 'medium': 'yellow', 'high': 'red'}[zone]
    return zone, label, color, RECOMMENDATIONS[zone]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    f = request.files.get('photo')
    if f is None or f.filename == '':
        return render_template('index.html', error='Пожалуйста, выберите фото.')

    try:
        image_bytes = f.read()
        x = preprocess_for_model(image_bytes)
        prob = float(model.predict(x, verbose=0)[0][0])
    except Exception as e:  # noqa: BLE001
        return render_template('index.html', error=f'Не удалось обработать файл: {e}')

    score = round(risk_score(prob), 1)
    zone, label, color, recommendation = classify(score)
    thumb = make_thumbnail_b64(image_bytes)

    return render_template(
        'result.html',
        thumb=thumb,
        score=score,
        label=label,
        color=color,
        recommendation=recommendation,
        abcde=ABCDE,
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
