# IAEML — Assignments 1 и 2

## Материалы для сдачи

- [Отчёт Assignment 1: модель автомобиля](REPORT_ASSIGNMENT1.md).
- [Отчёт Assignment 2: PPO с задним ходом](REPORT_ASSIGNMENT2.md).
- [Финальные веса](results/assignment2/reverse_seed1/best_validation.npz).
- [Конфигурация обучения](results/assignment2/reverse_seed1/train/config.json).
- [Итоговый тест](results/assignment2/reverse_seed1/test.json).
- [Выбор контрольной точки](results/assignment2/reverse_seed1/selection.json).

В отчётах уже включена личная рефлексия на основе текста автора.
**Перед отправкой остаётся проверить формулировки и заполнить ФИО, группу
и подпись в обоих отчётах.**

Преподавательские инструкции сохранены в `instructions/`. Преподавательские
файлы среды, физика автомобиля и `requirements.txt` при очистке не менялись.

## Результаты и видео

| Карта | Назначение | Успехи на 64 новых эпизодах | Видео исходного сценария |
| --- | --- | --- | --- |
| 0 | обучающая геометрия | 55/64 | [успех, 8,9 с](results/assignment2/reverse_seed1/videos/map0.mp4) |
| 1 | обучающая геометрия | 46/64 | [успех, 8,3 с](results/assignment2/reverse_seed1/videos/map1.mp4) |
| 2 | геометрия вне обучения | 29/64 | [столкновение, 19,8 с](results/assignment2/reverse_seed1/videos/map2.mp4) |

Для первого задания: [демонстрация разгона, торможения и заднего хода](results/reverse/reverse_demo.mp4),
[измерения](results/reverse/demo_summary.json).

[График обучения](results/assignment2/reverse_seed1/review/plots/training.png) и
[сравнение тестов](results/assignment2/reverse_seed1/review/plots/evaluation.png)
включают контрольный запуск без заднего хода. Для него оставлены численные
результаты и журнал в `results/assignment2/strict_seed1/`, но старые веса не нужны.
Результат одного seed не доказывает устойчивость по множеству запусков;
перенос на третью карту остаётся ограниченным.

## Окружение

Нужно существующее Python-окружение с зависимостями исходного `requirements.txt`.
Пакеты при очистке не устанавливались и не обновлялись. Зафиксированное окружение
финального запуска описано в
[train/runtime.json](results/assignment2/reverse_seed1/train/runtime.json).
В нём использовались Python 3.12.15, JAX 0.8.0, Equinox 0.13.2, Optax 0.2.6,
NumPy 2.3.4, CPU macOS arm64. Для проекта достаточно одного из указанных
в исходной заготовке фреймворков нейросетей — здесь используется Equinox.

Все команды ниже выполняются из корня репозитория. Задайте интерпретатор:

```bash
IAEML_PYTHON=python
```
Matplotlib нужен лишь для перестроения дополнительных графиков, ffmpeg или
imageio-ffmpeg — лишь для записи новых видео. Для запуска модели они не нужны.

## Ручное управление и проверка физики

```bash
JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment1.py --map 1

# Фиксированная демонстрация, без обучения:
JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment1.py --reverse-demo --map 1 --steps 420
```

↑ — вперёд, ↓ — назад. Противоположная команда сначала тормозит до нуля. ←/→ поворачивают колёса; на месте машина не вращается. Максимальная
скорость вперёд 3 м/с, назад 1,5 м/с. При отпускании клавиш машина постепенно
замедляется, руль возвращается к нулю. После завершения эпизода перезапустите
команду. Исходный старт находится у стены: перед движением назад отъедьте вперёд.

Для воспроизведения старого режима без заднего хода предусмотрен `--forward-only`.
Прежние форматы checkpoint также загружаются с исходной физикой.

## Просмотр обученного агента

```bash
JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment2.py play \
  --checkpoint results/assignment2/reverse_seed1/best_validation.npz --map 0
```

Меняйте `--map` на 1 или 2. Режим заднего хода автоматически читается из весов.
При первом запуске JAX компилирует вычисления. Политика при просмотре
детерминированная; руль и ускорение вручную не корректируются.

Точно повторить первый вариативный эпизод итогового теста карты 0:

```bash
JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment2.py play \
  --checkpoint results/assignment2/reverse_seed1/best_validation.npz \
  --map 0 --split test --seed 30000 --evaluation-episode 1 --evaluation-episodes 64
```

Для дополнительных эпизодов можно использовать `--map 0 --varied --seed 30001`.
Они тоже берутся из тестового пространственного пула, но не обязательно совпадают
с нумерацией эпизодов сохранённой оценки.

## Разделение обучения и проверки

Обучаются только карты **0 и 1**, seed 1. Свободные точки полуметровой сетки
с зазором больше 2,1 м разделены фиксированным хешем на train/validation/test
с учётом отражений. Между наборами начальных точек нет пересечений. Проезжать
через тестовые координаты во время обучающих траекторий агент мог.

Около 35% обучающих вариантов используют исходный старт с разбросом курса
примерно ±27°, остальные — свободные старты с курсом из полного круга.
При каждом сбросе курс выбирается заново. Варианты отражаются вместе с целью
и скоростями препятствий; фаза движущихся препятствий выбирается из 0–12 с.
Начальные столкновения исключаются. Старты ограничены достаточно свободными
участками и не покрывают все возможные позиции возле стен.

Для validation используются только карты 0/1, seed 20000, 32 вариации и один
исходный сценарий на карту. Checkpoints 200/400/600/800/1000 сравниваются по
половине среднего успеха исходных сценариев плюс половине успеха вариаций;
при равенстве выбирается меньшая доля столкновений. Итоговый тест выполняется
после фиксации весов: карты 0/1/2, seed 30000, 64 вариации и исходный сценарий.
Исходный сценарий не называется новым стартом. Курсы и фазы получают независимые
выборки с другими seeds; пространственные пулы разделены явно.

[protocol.json](results/assignment2/reverse_seed1/protocol.json) содержит аудит
пересечений. Указанные в нём геометрические пулы карты 2 не использовались
для обучения или выбора весов. Её геометрия ранее диагностировалась, поэтому
она не называется впервые открытым слепым тестом. Агент получает опорный маршрут
текущей карты: проверяется перенос управления при навигационной подсказке.

## Повторная оценка и тесты

```bash
JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment2.py evaluate \
  --checkpoint results/assignment2/reverse_seed1/best_validation.npz \
  --maps 0 1 2 --split test --seed 30000 --episodes 64 \
  --out results/assignment2/recheck_submission.json

PYTHONPATH=src JAX_PLATFORMS=cpu "$IAEML_PYTHON" -m unittest discover -s tests -v
```

Для проверки только validation: `--maps 0 1 --split validation --seed 20000 --episodes 32`.
Выбор весов защищён от использования тестового пула и карты 2. Старые веса,
обученные без пространственного разделения, не допускаются к инициализации
строгого эксперимента.

## Полное повторение обучения

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu "$IAEML_PYTHON" -m rl.protocol \
  --out results/assignment2/reproduced_reverse --reverse \
  --seed 1 --updates 1000 --checkpoint-every 200 \
  --validation-seed 20000 --test-seed 30000 \
  --validation-episodes 32 --test-episodes 64
```

Каталог вывода должен быть новым. Команда последовательно выполняет обучение,
validation, выбор и итоговый тест. Промежуточные checkpoints генерируются
при новом обучении, но в комплекте сдачи оставлен только выбранный
`best_validation.npz`. Пути промежуточных весов в JSON результатов validation
являются записью исходного эксперимента, а не командами для просмотра.
Checkpoint содержит модель и конфигурацию, но не состояние Adam для точного
возобновления прерванного обучения. Журналы, версии и метрики сохранены.

## Запись видео

```bash
for IAEML_MAP_ID in 0 1 2; do
  JAX_PLATFORMS=cpu "$IAEML_PYTHON" src/assignment2.py play \
    --checkpoint results/assignment2/reverse_seed1/best_validation.npz \
    --map "$IAEML_MAP_ID" --headless \
    --record "results/assignment2/new_videos/map${IAEML_MAP_ID}.mp4"
done
```

При обычном просмотре видео не записывается. Папки для `--record` создаются
автоматически. Неудачный эпизод тоже записывается. `--headless` использует
SDL dummy display и не ждёт реального времени.

## Дополнительные проверки и графики

```bash
# Перестроить сравнение с контрольным запуском (нужен matplotlib):
PYTHONPATH=src "$IAEML_PYTHON" -m rl.plot_results \
  results/assignment2/strict_seed1 results/assignment2/reverse_seed1 \
  --out results/reproduced_plots

# Проверить скорость отображения среды и совпадение результатов:
PYTHONPATH=src JAX_PLATFORMS=cpu "$IAEML_PYTHON" -m rl.benchmark --out results/benchmark_recheck.json
```
