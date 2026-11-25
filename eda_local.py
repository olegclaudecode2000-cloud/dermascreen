# -*- coding: utf-8 -*-
"""
Локальный EDA датасета HAM10000.
Строит и сохраняет графики для отчёта/презентации:
  - eda_output/class_distribution.png  (распределение классов с подписями)
  - eda_output/sample_images_grid.png  (сетка 16 случайных снимков 4x4)

Запуск:  venv\Scripts\python.exe eda_local.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # без интерактивного окна
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
CSV = os.path.join(DATA, 'HAM10000_metadata.csv')
IMG = os.path.join(DATA, 'images')
OUT = os.path.join(BASE, 'eda_output')
os.makedirs(OUT, exist_ok=True)

DX_NAMES = {
    'nv': 'Меланоцитарный невус (родинка)',
    'mel': 'Меланома',
    'bkl': 'Доброкачественный кератоз',
    'bcc': 'Базальноклеточная карцинома',
    'akiec': 'Актинический кератоз / Болезнь Боуэна',
    'vasc': 'Сосудистые образования',
    'df': 'Дерматофиброма',
}
HIGH_RISK = {'mel', 'bcc', 'akiec'}

# ---------------------------------------------------------------
df = pd.read_csv(CSV)
print('Размер таблицы:', df.shape)
print('\nПропуски (NaN) по столбцам:\n', df.isna().sum())

class_counts = df['dx'].value_counts()
print('\nКоличество снимков каждого класса (dx):')
for code in class_counts.index:
    print(f'  {code:<6} {DX_NAMES[code]:<45} {class_counts[code]}')

df['risk_group'] = df['dx'].apply(
    lambda x: 'Высокий риск' if x in HIGH_RISK else 'Низкий риск')
print('\nГруппы риска:\n', df['risk_group'].value_counts().to_dict())
print('\nДоля групп (%):\n',
      (df['risk_group'].value_counts(normalize=True) * 100).round(1).to_dict())

# ---------------------------------------------------------------
# 1) Столбчатая диаграмма
# ---------------------------------------------------------------
colors = ['#C44E52' if c in HIGH_RISK else '#4C72B0' for c in class_counts.index]
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(class_counts.index, class_counts.values,
              color=colors, edgecolor='black', linewidth=0.6)
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
            str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Распределение классов в датасете HAM10000 (n = 10 015)',
             fontsize=15, fontweight='bold')
ax.set_xlabel('Код диагноза (dx)', fontsize=13)
ax.set_ylabel('Количество изображений', fontsize=13)
ax.set_ylim(0, class_counts.max() * 1.12)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(handles=[
    Patch(facecolor='#4C72B0', label='Низкий риск (доброкачественные)'),
    Patch(facecolor='#C44E52', label='Высокий риск (требуют врача)'),
], loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'class_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print('\nСохранено: eda_output/class_distribution.png')

# ---------------------------------------------------------------
# 2) Сетка 16 случайных снимков (4x4)
# ---------------------------------------------------------------
sample = df.sample(n=16, random_state=42)
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()
for i, (_, row) in enumerate(sample.iterrows()):
    p = os.path.join(IMG, row['image_id'] + '.jpg')
    if os.path.exists(p):
        axes[i].imshow(Image.open(p))
        age = f"{int(row['age'])} лет" if not pd.isna(row['age']) else 'возраст неизв.'
        sex = 'М' if row['sex'] == 'male' else 'Ж'
        axes[i].set_title(f"{row['dx']} | {sex}, {age}", fontsize=11)
    else:
        axes[i].set_title('файл не найден', fontsize=10)
    axes[i].axis('off')
plt.suptitle('Случайные 16 снимков из HAM10000 (4×4)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sample_images_grid.png'), dpi=150, bbox_inches='tight')
plt.close()
print('Сохранено: eda_output/sample_images_grid.png')
print('\nEDA завершён.')
