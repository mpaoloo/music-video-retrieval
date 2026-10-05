# Автоматический подбор музыки для видео

По видеоролику модель ранжирует аудиодорожки из заданного набора.
Видео описывается признаками ResNet50. Для аудио сравниваются mel-статистики и AST.
Обучаются линейные проекции в общее пространство с L2-нормализацией.

## Запуск

Python 3.11–3.12, 16 GB RAM, рекомендуется CUDA GPU. В Colab выбрать T4.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.revision.smoke
python -u -m src.revision.run
```

Colab: `notebooks/music_video_retrieval.ipynb`. Ноутбук содержит исходники,
устанавливает зависимости и запускает весь пайплайн. Результаты нужно скачать
до отключения runtime.

Параметры находятся в `configs/revision.json`. Этапы: загрузка HarmonySet,
проверка файлов, извлечение признаков, фильтрация Music/Speech, обучение,
оценка и подготовка примеров для прослушивания. Выбор checkpoint — по validation MRR.
Сравниваются пять вариантов на одинаковой галерее, с seed 42, 43, 44.

## Продолжение

```bash
python -u -m src.revision.run --start features
python -u -m src.revision.run --start train
python -u -m src.revision.run --start review
```

Готовые видео и признаки переиспользуются. Причина сбоя записывается
в `runs/revision/status.json`; ошибки загрузки — в `data/revision/downloads.csv`.

## Результаты

`validation.csv`, `test_metrics.csv`, конфигурации, веса и графики сохраняются
в `runs/revision`. Ручная оценка: `runs/revision/review/index.html`.
Отчёт, графики и ручные оценки находятся в `reports`.
Исходные эксперименты воспроизводятся скриптами `src/train.py`,
`src/evaluate.py` и `src/infer.py` с параметрами `configs/config.yaml`.
Новые эксперименты запускаются через `src.revision.run`.
Обучение завершено: 380 видео, 272 train, 55 validation, 53 test.
Основная галерея содержит 49 музыкальных тестовых дорожек. Средний MRR по трём
seed: mel — 0.128; AST — 0.204; AST с аудиофильтром — 0.159;
AST с полной очисткой — 0.163; AST на случайных 98 парах — 0.165.
Ridge AST: MRR 0.209 (один запуск). Выбранный по validation вариант — AST без фильтра.

Признаки, веса и результаты включены в репозиторий. Видео всех 380 пар
не включены; для ручной оценки сохранены медиа выбранных примеров.
Повторное обучение можно запустить с этапа train без скачивания видео.
Ручная оценка: 65 пар для семи видео, все запросы размечены полностью.
Один пропущенный балл восстановлен как 0 по отрицательному комментарию.
На них средняя оценка top-3 AST — 0.905, mel — 0.714, случайного выбора — 0.714.
Субъективное преимущество AST не подтверждено; один оценщик и малая выборка.
Оценки и сводка находятся в `reports/human_*`.

```bash
python -m src.revision.human --ratings runs/revision/review/human_ratings.csv
```

## Ограничения

Hit@k и MRR измеряют поиск исходной аудиодорожки, а не субъективную уместность музыки.
Автоматический фильтр требует ручной проверки. Одинаковая музыка в разных видео
не группируется автоматически; усреднение признаков не учитывает монтаж и ритм.

## Источники

- HarmonySet: https://arxiv.org/abs/2503.01725
- AST: https://arxiv.org/abs/2104.01778
- Веса AST: https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
- VMCML: https://openaccess.thecvf.com/content/CVPR2024W/MULA/html/Lee_VMCML_Video_and_Music_Matching_via_Cross-Modality_Lifting_CVPRW_2024_paper.html
