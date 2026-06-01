# Порівняльний аналіз методів машинного навчання з підкріпленням і еволюційних підходів для задач автоматичного управління в динамічному середовищі


![Demo](screenshots/demo.gif)

> 2D-середовище для навчання автономного автомобіля за допомогою алгоритмів **PPO (Proximal Policy Optimization)** та **NEAT (NeuroEvolution of Augmenting Topologies)**. Агент керує машиною на кастомних трасах, використовуючи LiDAR-сенсори для орієнтації.

---

## Автор

- **ПІБ**: Пелех Роман
- **Група**: ФЕІ-43
- **Керівник**: Дуфанець Марта, кандидат фізико-математичних наук, доцент кафедри оптоелектроніки та інформаційних технологій
- **Дата виконання**: 30.05.2026
- **GitHub**: [romanpelekh136](https://github.com/romanpelekh136)

---

## Загальна інформація

| | |
|---|---|
| **Тип проєкту** | Десктопне середовище + навчання RL-агентів |
| **Мова програмування** | Python 3.10+ |
| **Основні бібліотеки** | Gymnasium, Stable-Baselines3, NEAT-Python, Pygame, Optuna, TensorBoard |
| **Алгоритми** | PPO (gradient-based RL), NEAT (neuroevolution) |
| **Середовище** | Кастомне 2D Gymnasium з LiDAR-спостереженнями та чекпоінтами |

---

## Опис функціоналу

- **Кастомне 2D середовище** – автомобіль з bicycle-моделлю фізики, 21 LiDAR-промінь, система чекпоінтів та кіл
- **PPO-тренування** – навчання через Stable-Baselines3 з паралельними середовищами, Optuna-оптимізацією гіперпараметрів та TensorBoard логуванням
- **NEAT-тренування** – еволюційний пошук нейромережі з нуля через neat-python, паралельна оцінка геномів
- **Редактор мап** – інтерактивний Pygame-редактор для створення трас з чекпоінтами та стартовою позицією
- **Гонка PPO vs NEAT** – одночасний запуск двох агентів в одному вікні на одній трасі 
- **Збір даних та аналіз** – автоматичний збір бенчмарків, траєкторій, crash-мап та порівняння швидкості інференсу
- **Jupyter-аналітика** – ноутбук з візуалізаціями, порівняльними графіками та статистикою

---

## Структура проєкту

```
autonomous-vehicles/
├── custom_env.py              # Gymnasium середовище (CarRacingCustom)
├── train_ppo.py               # Тренування PPO + Optuna оптимізація
├── train_neat.py              # Тренування NEAT
├── test.py                    # Візуальне тестування моделей
├── race.py                    # Гонка PPO vs NEAT
├── benchmark.py               # Швидкий бенчмарк моделей
├── collect_data.py            # Повний збір даних для аналізу
├── map_editor.py              # Редактор трас
├── neat_config.txt            # Конфігурація NEAT
├── requirements.txt           # Python залежності
├── track_01.png / .json       # Файли трас (зображення + чекпоінти)
├── track_02.png / .json
├── track_03.png / .json
├── track_04.png / .json
├── models/
│   ├── ppo_best/              # Найкраща PPO модель
│   └── neat_best/             # Найкращий NEAT геном
├── tb_logs/                   # TensorBoard логи
├── analysis_data/
│   ├── analysis.ipynb         # Jupyter-ноутбук з аналізом
│   ├── benchmark_clean.json   # Результати бенчмарків
│   ├── benchmark_noisy.json
│   ├── trajectories_ppo.json  # Траєкторії агентів
│   ├── trajectories_neat.json
│   ├── crash_map.json         # Карти аварій
│   ├── inference_speed.json   # Швидкість прийняття рішень
│   ├── training_metadata.json # Метадані навчання
│   └── plots/                 # Згенеровані графіки
└── screenshots/               # Скриншоти та демо GIF
```

---

## Як запустити проєкт з нуля

### 1. Клонування репозиторію

```bash
git clone https://github.com/romanpelekh136/autonomous-vehicles.git
cd autonomous-vehicles
```

### 2. Створення віртуального середовища (рекомендовано)

```bash
# Створення venv
python -m venv venv

# Активація (Windows – PowerShell)
.\venv\Scripts\Activate.ps1

# Активація (Windows – CMD)
.\venv\Scripts\activate.bat

# Активація (Linux / macOS)
source venv/bin/activate
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

**Залежності (`requirements.txt`):**

| Бібліотека | Версія | Призначення |
|---|---|---|
| `gymnasium` | ≥1.0.0 | RL-середовища |
| `numpy` | ≥2.0.0, <2.2.0 | Числові обчислення |
| `pygame` | ≥2.5.0 | Рендеринг та візуалізація |
| `Pillow` | ≥10.0.0 | Завантаження зображень трас |
| `optuna` | ≥3.6.0 | Оптимізація гіперпараметрів |
| `neat-python` | ≥0.92 | NEAT алгоритм |
| `stable-baselines3` | ≥2.3.0 | PPO та інші RL-алгоритми |
| `tensorboard` | ≥2.16.0 | Візуалізація навчання |
| `torch` | ≥2.0.0 | Backend для SB3 та TensorBoard у NEAT |
| `tqdm` | ≥4.66.0 | Прогрес-бари у collect_data.py |
| `jupyter` | ≥1.0.0 | Запуск аналітичного ноутбука |

---

## Запуск та використання

### Тестування моделей (візуально)

Запуск навченої моделі з рендерингом у вікні Pygame:

```bash
python test.py --method ppo --track track_02
python test.py --method neat --track track_03
```

| Параметр | Значення | За замовчуванням | Опис |
|---|---|---|---|
| `--method` | `ppo`, `neat` | `ppo` | Вибір моделі |
| `--track` | `track_01`–`track_04` | `track_02` | Вибір траси |

> Закриття вікна – `Ctrl+C` у терміналі або закрити вікно Pygame.

---

### Тренування PPO

```bash
python train_ppo.py
```

Параметри настроюються у коді (`train_ppo.py`):

| Параметр | Значення | Опис |
|---|---|---|
| `OPTIMIZE` | `True` / `False` | `True` – запуск Optuna-пошуку, `False` – тренування з кращими параметрами |
| `total_timesteps` | `5_000_000` | Загальна кількість кроків навчання |
| Кількість середовищ | `8` | Паралельні середовища (SubprocVecEnv) |
| Архітектура | `pi=[256,256] vf=[512,512]` | MLP-мережа: policy та value |

**Під час навчання:**
- Автоматично запускається **TensorBoard** на `http://localhost:6006`
- Найкраща модель зберігається у `models/ppo_best/`
- Фінальна модель зберігається у `models/ppo_final.zip`

---

### Тренування NEAT

```bash
python train_neat.py
```

Параметри настроюються у `neat_config.txt`:

| Параметр | Значення | Опис |
|---|---|---|
| `pop_size` | `150` | Розмір популяції |
| `num_inputs` | `26` | 21 LiDAR + steering + speed + 3 prev_action |
| `num_outputs` | `3` | steering, gas, brake |
| `num_hidden` | `0` | Початкова кількість прихованих нейронів |
| `activation_default` | `tanh` | Функція активації |
| `node_add_prob` | `0.1` | Ймовірність додавання нейрону |
| `conn_add_prob` | `0.3` | Ймовірність додавання зв'язку |

**Під час навчання:**
- Запускається TensorBoard на `http://localhost:6006`
- Найкращий геном зберігається у `models/neat_best/best_model.pkl`
- Фінальний геном зберігається у `models/neat_final.pkl`
- Використовуються всі ядра CPU для паралельної оцінки

---

### Гонка PPO vs NEAT

Обидва агенти запускаються в **одному процесі** та відображаються в **одному вікні** (split-screen):

```bash
python race.py --track track_02 --laps 2
```

| Параметр | Значення | За замовчуванням | Опис |
|---|---|---|---|
| `--track` | `track_01`–`track_04` | `track_02` | Вибір траси |
| `--laps` | ціле число | `2` | Кількість кіл для перемоги |

**Особливості:**
- Ліва панель — PPO, права — NEAT, розділені вертикальним роздільником
- Хедер з лейблами агентів та live-таймером
- Після фінішу — overlay з результатами та визначенням переможця
- Вихід: `ESC` або закриття вікна

---

### Бенчмарк

Швидке порівняння моделей на всіх трасах:

```bash
python benchmark.py
python benchmark.py --noise
python benchmark.py --episodes 50
python benchmark.py --noise --episodes 30
```

| Параметр | За замовчуванням | Опис |
|---|---|---|
| `--noise` | вимкнено | Додає випадковий шум до steering (тест стійкості) |
| `--episodes` | `20` | Кількість епізодів на кожну трасу |

---

### Повний збір даних для аналізу

Збирає всі дані для Jupyter-ноутбука:

```bash
python collect_data.py
python collect_data.py --skip-benchmark
python collect_data.py --skip-trajectories
python collect_data.py --skip-crashes
python collect_data.py --skip-benchmark --skip-crashes
```

| Параметр | Опис |
|---|---|
| `--skip-benchmark` | Пропустити бенчмарк-епізоди |
| `--skip-trajectories` | Пропустити збір траєкторій |
| `--skip-crashes` | Пропустити збір карти аварій |

**Зібрані дані (папка `analysis_data/`):**

| Файл | Опис |
|---|---|
| `benchmark_clean.json` | Бенчмарк без шуму |
| `benchmark_noisy.json` | Бенчмарк з шумом steering |
| `trajectories_ppo.json` | Траєкторії PPO (x, y, speed, steer, gas, brake) |
| `trajectories_neat.json` | Траєкторії NEAT |
| `crash_map.json` | Координати аварій при шумі |
| `inference_speed.json` | Швидкість інференсу PPO vs NEAT |
| `training_metadata.json` | Метадані навчання (архітектура, параметри) |

---

### Редактор мап

Інтерактивний редактор для створення нових трас:

```bash
python map_editor.py my_new_track
```

> [!NOTE]
> Потрібно попередньо створити PNG-зображення траси (важливо використовувати інструменти для створення без згладжування пікселів).

**Управління у редакторі:**

| Кнопка | Дія |
|---|---|
| **ЛКМ** (2 рази) | Намалювати чекпоінт (лінія з точки A в точку B) |
| **ПКМ** (2 рази) | Встановити стартову позицію (1-й клік) та напрямок (2-й клік) |
| **Середня кнопка** | Перетягування карти |
| **Scroll** | Зум (наближення/віддалення) |
| **Z** | Скасувати останній чекпоінт |
| **C** | Очистити все |
| **S** | Зберегти у JSON-файл |

### TensorBoard

Перегляд логів навчання:

```bash
tensorboard --logdir=./tb_logs/ --port=6006
```

Відкрийте у браузері: [http://localhost:6006](http://localhost:6006)

---

## Опис основних файлів

| Файл | Призначення |
|---|---|
| `custom_env.py` | Gymnasium-середовище `CarRacingCustom` з LiDAR, bicycle-моделлю фізики, чекпоінтами та HUD |
| `train_ppo.py` | Навчання PPO: Optuna-оптимізація гіперпараметрів, паралельні env'и, TensorBoard, EvalCallback |
| `train_neat.py` | Навчання NEAT: еволюція нейромереж, TensorBoardReporter, авто-збереження кращого генома |
| `test.py` | Візуальне тестування моделі з логуванням дій у JSON |
| `race.py` | Гонка: PPO та NEAT одночасно в одному вікні 
| `benchmark.py` | Швидкий бенчмарк: progress, crash rate, інференс на 4 трасах |
| `collect_data.py` | Масовий збір даних: бенчмарки, траєкторії, crash-map, інференс, метадані |
| `map_editor.py` | Pygame-редактор трас із зумом, паном, створенням чекпоінтів та стартової позиції |
| `neat_config.txt` | Конфігурація NEAT: розмір популяції, ймовірності мутацій, топологія |

---

## Середовище: CarRacingCustom

### Простір спостережень (26 значень)

| Індекс | Значення | Діапазон |
|---|---|---|
| 0–20 | LiDAR-промені (21 шт, 180° дуга) | [0, 1] |
| 21 | Поточне значення steering | [-1, 1] |
| 22 | Нормалізована швидкість | [0, 1] |
| 23–25 | Попередня дія (steer, gas, brake) | [-1/0, 1] |

### Простір дій (3 значення, continuous)

| Індекс | Дія | Діапазон |
|---|---|---|
| 0 | Steering (поворот) | [-1, 1] |
| 1 | Gas (газ) | [0, 1] |
| 2 | Brake (гальмо) | [0, 1] |

### Система нагород

| Подія | Нагорода |
|---|---|
| Кожен крок | -0.1 |
| Швидкість у напрямку чекпоінту | до +2.5 |
| Проїзд чекпоінту | +20.0 |
| Завершення кола | +50.0 |
| Виліт з траси / застрягання | -100.0 |

---

## Проблеми та рішення

| Проблема | Рішення |
|---|---|
| `ModuleNotFoundError: No module named 'stable_baselines3'` | `pip install stable-baselines3` |
| `FileNotFoundError: track_XX.png or track_XX.json missing` | Переконайтесь, що запускаєте з кореневої папки проєкту |
| `pygame.error: video system not initialized` | На сервері без дисплея – не використовуйте `render_mode="human"` |
| TensorBoard не відкривається | Перевірте порт: `tensorboard --logdir=./tb_logs/ --port=6007` |
| NEAT тренування повільне | Зменшіть `pop_size` у `neat_config.txt` або кількість поколінь |
| PPO не навчається | Перевірте гіперпараметри або запустіть Optuna (`OPTIMIZE = True`) |
| `numpy` конфлікт версій | Потрібна версія ≥2.0.0 та <2.2.0 |

## Використані технології

- [Gymnasium](https://gymnasium.farama.org/) – стандарт RL-середовищ
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) – PPO та інші алгоритми
- [NEAT-Python](https://neat-python.readthedocs.io/) – нейроеволюція
- [Pygame](https://www.pygame.org/) – рендеринг та інтерактивність
- [Optuna](https://optuna.org/) – оптимізація гіперпараметрів
- [TensorBoard](https://www.tensorflow.org/tensorboard) – візуалізація навчання
- [NumPy](https://numpy.org/) – числові обчислення
- [Pillow](https://pillow.readthedocs.io/) – обробка зображень

