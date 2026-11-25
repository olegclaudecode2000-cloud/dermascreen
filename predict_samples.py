# -*- coding: utf-8 -*-
"""Быстрая проверка модели на тестовых снимках из папки samples/."""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, 'model', 'config.json'), encoding='utf-8'))
model = keras.models.load_model(os.path.join(BASE, 'model', 'skin_model.keras'))
ref = np.array(cfg['reference_probs'], dtype=np.float32)
sz = tuple(cfg['img_size'])

NAMES = {
    'nv': 'невус (доброкач.)', 'mel': 'меланома (опасно)',
    'bkl': 'кератоз (доброкач.)', 'bcc': 'базалиома (опасно)',
    'akiec': 'актинический кератоз (опасно)', 'vasc': 'сосуд. (доброкач.)',
    'df': 'дерматофиброма (доброкач.)',
}


def risk_score(path):
    img = Image.open(path).convert('RGB').resize(sz)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    p = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    return float((ref <= p).mean()) * 100.0


print(f"{'класс':<8}{'риск-скор':>12}   описание")
for f in sorted(os.listdir(os.path.join(BASE, 'samples'))):
    code = f.split('_')[0]
    s = risk_score(os.path.join(BASE, 'samples', f))
    print(f"{code:<8}{s:>10.1f}%   {NAMES.get(code, code)}")
