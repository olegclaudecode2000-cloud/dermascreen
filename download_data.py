# -*- coding: utf-8 -*-
"""
Скачивание изображений HAM10000 с зеркала Hugging Face
(полный датасет: HAM10000_images_part_1 + HAM10000_images_part_2).

Используется huggingface_hub (snapshot_download) — скачивает в несколько
потоков, с докачкой при обрыве.
"""
import time
from huggingface_hub import snapshot_download

t0 = time.time()
snapshot_download(
    repo_id="ShiroOnigami23/skin-cancer-ham10000-dataset",
    repo_type="dataset",
    local_dir=r"C:\Users\Олег\Desktop\Родинки\data\ham10000",
    allow_patterns=[
        "HAM10000_images_part_1/*.jpg",
        "HAM10000_images_part_2/*.jpg",
        "HAM10000_metadata.csv",
    ],
    max_workers=16,
)
print("HF_DOWNLOAD_DONE in", round(time.time() - t0, 1), "s")
