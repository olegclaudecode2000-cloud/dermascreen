# -*- coding: utf-8 -*-
"""
Обучение нейросети для первичной оценки риска кожных новообразований.
====================================================================
Датасет: HAM10000 (10 015 дерматоскопических снимков).
Задача: бинарная классификация «низкий риск» (0) vs «высокий риск» (1).

Двухэтапное обучение (transfer learning, MobileNetV2):
  Этап 1 — обучаем только «голову» (база заморожена), LR=1e-4.
  Этап 2 — fine-tuning: размораживаем базу и дообучаем с малым LR=1e-5.

Классы риска: ВЫСОКИЙ = {mel, bcc, akiec}, НИЗКИЙ = {nv, bkl, df, vasc}.

Запуск:  python train.py
На выходе:  model/skin_model.keras  +  model/config.json
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)

# ----------------------------------------------------------------------
# Конфигурация
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
METADATA_CSV = os.path.join(DATA_DIR, 'HAM10000_metadata.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_model.keras')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.json')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

HEAD_EPOCHS = 8          # этап 1 (голова)
HEAD_LR = 1e-4
FT_EPOCHS = 5            # этап 2 (fine-tuning)
FT_LR = 1e-5
MAX_CLASS_WEIGHT = 2.0   # ограничиваем вес редкого класса (против «схлопывания»)

HIGH_RISK = {'mel', 'bcc', 'akiec'}
CLASS_NAMES = {0: 'low_risk', 1: 'high_risk'}

os.makedirs(MODEL_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Загрузка разметки
# ----------------------------------------------------------------------
df = pd.read_csv(METADATA_CSV)
df['label'] = df['dx'].apply(lambda x: 1 if x in HIGH_RISK else 0)
df['filepath'] = df['image_id'].apply(
    lambda i: os.path.join(IMAGES_DIR, i + '.jpg'))
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)
print('Снимков на диске:', len(df))
print('Распределение групп риска:\n', df['label'].value_counts().to_dict())

# ----------------------------------------------------------------------
# 2. Стратифицированное разбиение
# ----------------------------------------------------------------------
train_df, test_df = train_test_split(
    df, test_size=0.20, stratify=df['label'], random_state=SEED)
train_df, val_df = train_test_split(
    train_df, test_size=0.15, stratify=train_df['label'], random_state=SEED)
print(f'train={len(train_df)}  val={len(val_df)}  test={len(test_df)}')

# Мягкие веса классов (против дисбаланса, но без «схлопывания» модели)
n_neg = int((train_df['label'] == 0).sum())
n_pos = int((train_df['label'] == 1).sum())
cw = min(n_neg / n_pos, MAX_CLASS_WEIGHT)
class_weight = {0: 1.0, 1: cw}
print(f'class_weight = {class_weight}')

# ----------------------------------------------------------------------
# 3. tf.data конвейер
# ----------------------------------------------------------------------
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0          # -> [0, 1]


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img


def make_dataset(data_df, shuffle=False, use_augment=False):
    paths = data_df['filepath'].values
    labels = data_df['label'].values.astype(np.float32)

    def _proc(p, l):
        img = load_image(p)
        if use_augment:
            img = augment(img)
        img = keras.applications.mobilenet_v2.preprocess_input(img)
        return img, l

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_proc, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


train_ds = make_dataset(train_df, shuffle=True, use_augment=True)
val_ds = make_dataset(val_df, shuffle=False)
test_ds = make_dataset(test_df, shuffle=False)

# ----------------------------------------------------------------------
# 4. Модель
# ----------------------------------------------------------------------
base = keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet',
)
base.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)          # BN в inference-режиме
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

METRICS = ['accuracy',
           keras.metrics.Precision(name='precision'),
           keras.metrics.Recall(name='recall'),
           keras.metrics.AUC(name='auc')]


def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=3, mode='max', verbose=1,
            restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ]


# ----------------------------------------------------------------------
# 5. Этап 1: обучение головы (база заморожена)
# ----------------------------------------------------------------------
print('\n=== ЭТАП 1: обучение головы ===')
model.compile(optimizer=keras.optimizers.Adam(HEAD_LR),
              loss='binary_crossentropy', metrics=METRICS)
model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
          class_weight=class_weight, callbacks=make_callbacks(), verbose=1)

# ----------------------------------------------------------------------
# 6. Этап 2: fine-tuning (размораживаем базу, малый LR)
# ----------------------------------------------------------------------
print('\n=== ЭТАП 2: fine-tuning базы ===')
base.trainable = True
model.compile(optimizer=keras.optimizers.Adam(FT_LR),
              loss='binary_crossentropy', metrics=METRICS)
model.fit(train_ds, validation_data=val_ds, epochs=FT_EPOCHS,
          class_weight=class_weight, callbacks=make_callbacks(), verbose=1)

# ----------------------------------------------------------------------
# 7. Оценка: порог по валидации, метрики по тесту
# ----------------------------------------------------------------------
val_labels = val_df['label'].values
val_probs = model.predict(val_ds, verbose=0).ravel()
prec, rec, thresh = precision_recall_curve(val_labels, val_probs)
f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
best_i = int(np.argmax(f1))
best_threshold = float(thresh[best_i]) if best_i < len(thresh) else 0.5
print(f'\nЛучший порог (max F1, на валидации) = {best_threshold:.3f}')

test_labels = test_df['label'].values
probs = model.predict(test_ds, verbose=0).ravel()
auc = roc_auc_score(test_labels, probs)
print(f'ROC-AUC (test) = {auc:.4f}')

preds = (probs >= best_threshold).astype(int)
print('\nConfusion matrix (строки=факт, столбцы=прогноз):')
print(confusion_matrix(test_labels, preds))
print('\nClassification report (test):')
print(classification_report(test_labels, preds,
                            target_names=['low_risk', 'high_risk'],
                            zero_division=0))

# ----------------------------------------------------------------------
# 8. Сохранение
# ----------------------------------------------------------------------
model.save(MODEL_PATH)
config = {
    'img_size': list(IMG_SIZE),
    'class_names': CLASS_NAMES,
    'threshold': best_threshold,
    'preprocess': 'mobilenet_v2',
    'roc_auc': float(auc),
    'high_risk_codes': sorted(HIGH_RISK),
    'low_risk_codes': sorted({'nv', 'bkl', 'df', 'vasc'}),
    'zones': {
        'low': {'max': 0.35, 'label': 'Низкий риск', 'color': 'green'},
        'medium': {'max': 0.65, 'label': 'Средний риск (пограничная зона)', 'color': 'yellow'},
        'high': {'max': 1.0, 'label': 'Высокий риск', 'color': 'red'},
    },
}
with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print('\nГотово. ROC-AUC:', round(auc, 4), '| Порог:', round(best_threshold, 3))
print('Модель:', MODEL_PATH)
