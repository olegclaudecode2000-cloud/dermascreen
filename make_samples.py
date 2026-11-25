# -*- coding: utf-8 -*-
"""Копирует по одному примеру каждого класса в папку samples/ для теста сайта."""
import os
import shutil
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE, 'data', 'HAM10000_metadata.csv'))
out = os.path.join(BASE, 'samples')
os.makedirs(out, exist_ok=True)

for code in ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']:
    row = df[df['dx'] == code].iloc[0]
    src = os.path.join(BASE, 'data', 'images', row['image_id'] + '.jpg')
    dst = os.path.join(out, f"{code}_{row['image_id']}.jpg")
    shutil.copy(src, dst)
    print(f'{code} -> samples/{os.path.basename(dst)}')
print('Готово: папка samples/ содержит примеры всех 7 классов.')
