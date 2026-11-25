# -*- coding: utf-8 -*-
"""
Продолжение fine-tuning существующей модели с более высоким LR (1e-4).
Начинаем с уже обученной модели (skin_model.keras), размораживаем базу
и дообучаем с агрессивным LR, чтобы признаки MobileNetV2 адаптировались
к дерматоскопическим снимкам.
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, classification_report,
                             confusion_matrix, precision_recall_curve)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
META = os.path.join(DATA, 'HAM10000_metadata.csv')
IMG = os.path.join(DATA, 'images')
MODEL_PATH = os.path.join(BASE, 'model', 'skin_model.keras')
CFG = os.path.join(BASE, 'model', 'config.json')

IMG_SIZE = (224, 224)
BATCH = 32
SEED = 42
HIGH = {'mel', 'bcc', 'akiec'}
EPOCHS = 8
LR = 1e-4
MAX_CW = 2.0

df = pd.read_csv(META)
df['label'] = df['dx'].apply(lambda x: 1 if x in HIGH else 0)
df['filepath'] = df['image_id'].apply(lambda i: os.path.join(IMG, i + '.jpg'))
df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
train_df, val_df = train_test_split(train_df, test_size=0.15, stratify=train_df['label'], random_state=SEED)

n0 = int((train_df['label'] == 0).sum())
n1 = int((train_df['label'] == 1).sum())
class_weight = {0: 1.0, 1: min(n0 / n1, MAX_CW)}


def load_image(p):
    img = tf.io.read_file(p)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    return tf.cast(img, tf.float32) / 255.0


def augment(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    return img


def make_ds(d, shuffle=False, aug=False):
    paths = d['filepath'].values
    labels = d['label'].values.astype(np.float32)

    def proc(p, l):
        im = load_image(p)
        if aug:
            im = augment(im)
        im = keras.applications.mobilenet_v2.preprocess_input(im)
        return im, l

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(proc, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2048)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


train_ds = make_ds(train_df, True, True)
val_ds = make_ds(val_df)
test_ds = make_ds(test_df)

model = keras.models.load_model(MODEL_PATH)
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=keras.optimizers.Adam(LR),
              loss='binary_crossentropy',
              metrics=['accuracy',
                       keras.metrics.Precision(name='precision'),
                       keras.metrics.Recall(name='recall'),
                       keras.metrics.AUC(name='auc')])

callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode='max',
                                  verbose=1, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          class_weight=class_weight, callbacks=callbacks, verbose=1)

# --- оценка ---
val_labels = val_df['label'].values
val_probs = model.predict(val_ds, verbose=0).ravel()
prec, rec, thr = precision_recall_curve(val_labels, val_probs)
f1 = 2 * prec * rec / (prec + rec + 1e-9)
best_i = int(np.argmax(f1))
thr_best = float(thr[best_i]) if best_i < len(thr) else 0.5

test_labels = test_df['label'].values
probs = model.predict(test_ds, verbose=0).ravel()
auc = roc_auc_score(test_labels, probs)
print(f'\nROC-AUC (test) = {auc:.4f}  |  порог = {thr_best:.4f}')
preds = (probs >= thr_best).astype(int)
print('\nConfusion matrix:\n', confusion_matrix(test_labels, preds))
print('\nClassification report (test):')
print(classification_report(test_labels, preds,
                            target_names=['low_risk', 'high_risk'], zero_division=0))

# --- сохранить и обновить конфиг ---
model.save(MODEL_PATH)
cfg = json.load(open(CFG, encoding='utf-8'))
cfg['threshold'] = thr_best
cfg['roc_auc'] = float(auc)
q40, q70 = np.percentile(val_probs, [40, 70])
cfg['zones'] = {
    'low': {'max': round(float(q40), 6), 'label': 'Низкий риск', 'color': 'green'},
    'medium': {'max': round(float(q70), 6), 'label': 'Средний риск (пограничная зона)', 'color': 'yellow'},
    'high': {'max': 1.0, 'label': 'Высокий риск', 'color': 'red'},
}
cfg['reference_probs'] = sorted(round(float(x), 6) for x in val_probs)
json.dump(cfg, open(CFG, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print(f'Сохранено. ROC-AUC={auc:.4f}, порог={thr_best:.4f}')
