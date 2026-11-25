# -*- coding: utf-8 -*-
"""
Калибровка зон риска и финальная оценка модели.
Пересчитывает зоны (green/yellow/red) по процентилям вероятностей на валидации,
сохраняет эталонное распределение для шкалы «риск-скор», печатает метрики теста.
"""
import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

BASE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(BASE, 'model', 'config.json')
MODEL = os.path.join(BASE, 'model', 'skin_model.keras')
META = os.path.join(BASE, 'data', 'HAM10000_metadata.csv')
IMG = os.path.join(BASE, 'data', 'images')

with open(CFG, encoding='utf-8') as f:
    cfg = json.load(f)
model = keras.models.load_model(MODEL)

df = pd.read_csv(META)
df['label'] = df['dx'].apply(lambda x: 1 if x in {'mel', 'bcc', 'akiec'} else 0)
df['filepath'] = df['image_id'].apply(lambda i: os.path.join(IMG, i + '.jpg'))
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=42)

IMG_SIZE = tuple(cfg['img_size'])


def preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return keras.applications.mobilenet_v2.preprocess_input(img)


def predict_all(data_df):
    out = []
    for p in data_df['filepath'].values:
        x = tf.expand_dims(preprocess(p), 0)
        out.append(float(model.predict(x, verbose=0)[0][0]))
    return np.array(out)


# --- зоны по процентилям вероятностей на валидации ---
val_probs = predict_all(val_df)
q40, q70 = np.percentile(val_probs, [40, 70])
cfg['zones'] = {
    'low': {'max': round(float(q40), 6), 'label': 'Низкий риск', 'color': 'green'},
    'medium': {'max': round(float(q70), 6), 'label': 'Средний риск (пограничная зона)', 'color': 'yellow'},
    'high': {'max': 1.0, 'label': 'Высокий риск', 'color': 'red'},
}
cfg['reference_probs'] = sorted(round(float(x), 6) for x in val_probs)
with open(CFG, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)
print('Зоны риска (процентили 40/70):', cfg['zones'])

# --- метрики на тесте ---
test_probs = predict_all(test_df)
test_labels = test_df['label'].values
auc = roc_auc_score(test_labels, test_probs)
print(f'\nROC-AUC (test) = {auc:.4f}')

preds = (test_probs >= cfg['threshold']).astype(int)
print('\nConfusion matrix (строки=факт, столбцы=прогноз):')
print(confusion_matrix(test_labels, preds))
print('\nClassification report (test):')
print(classification_report(test_labels, preds,
                            target_names=['low_risk', 'high_risk'], zero_division=0))
