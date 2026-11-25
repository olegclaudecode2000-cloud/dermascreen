# -*- coding: utf-8 -*-
"""
Подбирает показательные примеры для демонстрации: для опасных классов
(mel, bcc, akiec) — снимок с НАИБОЛЬШИМ риск-скором, для доброкачественных
(nv, bkl, df, vasc) — с НАИМЕНЬШИМ. Копирует в папку demo/.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(BASE, 'model', 'config.json'), encoding='utf-8'))
model = keras.models.load_model(os.path.join(BASE, 'model', 'skin_model.keras'))
ref = np.array(cfg['reference_probs'], dtype=np.float32)
sz = tuple(cfg['img_size'])
HIGH = {'mel', 'bcc', 'akiec'}
LOW = {'nv', 'bkl', 'df', 'vasc'}

df = pd.read_csv(os.path.join(BASE, 'data', 'HAM10000_metadata.csv'))
os.makedirs(os.path.join(BASE, 'demo'), exist_ok=True)


def score(path):
    img = Image.open(path).convert('RGB').resize(sz)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = keras.applications.mobilenet_v2.preprocess_input(arr)
    p = float(model.predict(np.expand_dims(arr, 0), verbose=0)[0][0])
    return float((ref <= p).mean()) * 100.0


for code in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']:
    subset = df[df['dx'] == code].head(40)
    best = None
    for _, row in subset.iterrows():
        p = os.path.join(BASE, 'data', 'images', row['image_id'] + '.jpg')
        if not os.path.exists(p):
            continue
        s = score(p)
        if best is None:
            best = (s, p, row['image_id'])
            continue
        if code in HIGH and s > best[0]:
            best = (s, p, row['image_id'])
        if code in LOW and s < best[0]:
            best = (s, p, row['image_id'])
    if best:
        dst = os.path.join(BASE, 'demo', f"{code}_{best[2]}.jpg")
        shutil.copy(best[1], dst)
        print(f"{code:<6} риск-скор={best[0]:5.1f}%  ->  demo/{os.path.basename(dst)}")
print('Готово: папка demo/ содержит показательные примеры.')
