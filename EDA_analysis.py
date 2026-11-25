# -*- coding: utf-8 -*-
"""
EDA-анализ датасета HAM10000 (Skin Cancer MNIST)
================================================
Скрипт для Google Colab. Выполняет 4 шага:
  1) загрузка HAM10000_metadata.csv и базовая статистика;
  2) подсчёт классов (dx) и объединённых групп риска;
  3) столбчатая диаграмма с подписями значений (сохраняется в PNG 300 dpi);
  4) сетка 16 случайных снимков (4x4) с подписью диагноза, пола и возраста.

Перед запуском убедитесь, что в Colab есть:
  - /content/HAM10000_metadata.csv
  - папка /content/ham10000_images/ с файлами <image_id>.jpg
При необходимости поменяйте пути CSV_PATH и IMG_DIR ниже.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------
# Настройки отображения
# ------------------------------------------------------------------
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 200)

# Пути к данным (подставьте свои, если они отличаются)
CSV_PATH = '/content/HAM10000_metadata.csv'   # таблица разметки
IMG_DIR = '/content/ham10000_images'          # папка с изображениями

# Полные названия диагнозов (расшифровка кодов dx)
DX_NAMES = {
    'nv':    'Меланоцитарный невус (родинка)',
    'mel':   'Меланома',
    'bkl':   'Доброкачественный кератоз',
    'bcc':   'Базальноклеточная карцинома',
    'akiec': 'Актинический кератоз / Болезнь Боуэна',
    'vasc':  'Сосудистые образования',
    'df':    'Дерматофиброма',
}

# Группы риска для бинарной классификации
HIGH_RISK = {'mel', 'bcc', 'akiec'}                 # требуют внимания врача
LOW_RISK = {'nv', 'bkl', 'df', 'vasc'}              # доброкачественные


# ==================================================================
# ШАГ 1. Загрузка и базовая статистика
# ==================================================================
df = pd.read_csv(CSV_PATH)

print('Размер таблицы (строки, столбцы):', df.shape)
print('\nПервые 5 строк:')
print(df.head())

print('\nТипы данных столбцов:')
print(df.dtypes)

print('\nКоличество пропусков (NaN) по столбцам:')
print(df.isna().sum())

print('\nУникальные значения ключевых столбцов:')
print('dx (диагнозы):        ', sorted(df['dx'].unique()))
print('dx_type (способ):     ', sorted(df['dx_type'].unique()))
print('sex (пол):            ', sorted(df['sex'].unique()))
print('localization (место): ', sorted(df['localization'].unique()))


# ==================================================================
# ШАГ 2. Подсчёт классов и групп риска
# ==================================================================
print('\n=== Количество снимков каждого класса (dx) ===')
class_counts = df['dx'].value_counts()
for code in class_counts.index:
    print(f'  {code:<6} {DX_NAMES[code]:<45} {class_counts[code]}')

# Объединение в 2 группы риска
df['risk_group'] = df['dx'].apply(
    lambda x: 'Высокий риск' if x in HIGH_RISK else 'Низкий риск'
)

print('\n=== Объединённые группы риска ===')
group_counts = df['risk_group'].value_counts()
print(group_counts)
print('\nДоля групп (в %):')
print((df['risk_group'].value_counts(normalize=True) * 100).round(1))


# ==================================================================
# ШАГ 3. Столбчатая диаграмма распределения классов
# ==================================================================
# Цвет: синий = низкий риск, красный = высокий риск
colors = ['#C44E52' if code in HIGH_RISK else '#4C72B0'
          for code in class_counts.index]

fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(class_counts.index, class_counts.values,
              color=colors, edgecolor='black', linewidth=0.6)

# Подписи значений над столбцами
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 60,
            str(val),
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title('Распределение классов в датасете HAM10000 (n = 10 015)',
             fontsize=15, fontweight='bold')
ax.set_xlabel('Код диагноза (dx)', fontsize=13)
ax.set_ylabel('Количество изображений', fontsize=13)
ax.set_ylim(0, class_counts.max() * 1.12)
ax.grid(axis='y', linestyle='--', alpha=0.4)

# Легенда (цветовая индикация риска)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4C72B0', label='Низкий риск (доброкачественные)'),
    Patch(facecolor='#C44E52', label='Высокий риск (требуют врача)'),
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
# Сохранение в высоком качестве для презентации
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
print('Диаграмма сохранена в class_distribution.png (300 dpi)')


# ==================================================================
# ШАГ 4. Сетка 16 случайных снимков (4x4)
# ==================================================================
# Случайная выборка 16 строк (random_state — для воспроизводимости)
sample = df.sample(n=16, random_state=42)

fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.ravel()

for i, (_, row) in enumerate(sample.iterrows()):
    img_path = os.path.join(IMG_DIR, row['image_id'] + '.jpg')

    # Пропускаем, если файла нет (защита от опечаток в пути)
    if not os.path.exists(img_path):
        axes[i].set_title(f"{row['image_id']} — файл не найден", fontsize=10)
        axes[i].axis('off')
        continue

    img = Image.open(img_path)
    axes[i].imshow(img)

    # Подпись: диагноз | пол, возраст
    age = f"{int(row['age'])} лет" if not pd.isna(row['age']) else 'возраст неизв.'
    sex = 'М' if row['sex'] == 'male' else 'Ж'
    label = f"{row['dx']} ({DX_NAMES[row['dx']]})\n{sex}, {age}"
    axes[i].set_title(label, fontsize=11)
    axes[i].axis('off')

plt.suptitle('Случайные 16 снимков из HAM10000 (4×4)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_images_grid.png', dpi=150, bbox_inches='tight')
plt.show()
print('Сетка снимков сохранена в sample_images_grid.png')
