# Лабораторна робота 3 — Нейронні мережі (Python + сайт)

Проєкт демонструє роботу з бібліотекою для нейронних мереж (**TensorFlow/Keras** на прикладі; FANN наведено лише як приклад альтернативи). Реалізовано навчання мережі Multi-Layer Perceptron (MLP) на датасеті MNIST, оцінку якості та інтерактивний веб-інтерфейс на **Streamlit**.

## Структура проєкту

| Файл | Призначення |
|------|-------------|
| `app.py` | Streamlit-сайт: навчання, оцінка, матриця помилок, інференс на зразках |
| `train_model.py` | Автоматичне навчання та збереження моделі у `models/mnist_mlp.keras` |
| `mlp_numpy.py` | MLP «з нуля» на NumPy ( forward / backprop на задачі XOR) — для розуміння всередини |
| `requirements.txt` | Python-залежності |
| `README.md` | Цей файл |

## Вимоги

- Python 3.10+
- Інтернет для першого запуску (завантаження датасету MNIST)

## Встановлення та запуск

### 1. Створити віртуальне середовище (рекомендовано)

```powershell
# Якщо uv не встановлено:
powershell -NoProfile -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Створити venv та встановити залежності
uv venv --python 3.12 .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Або звичайним способом:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Запустити веб-сайт (Streamlit)

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Відкриється сторінка `http://localhost:8501`, де можна:
- налаштувати параметри навчання (екози, розмір пакунку, розмір шарів, dropout, learning rate);
- навчити модель на MNIST;
- подивитися тестову точність, звіт класифікації та матрицю помилок;
- переглянути графіки навчання (accuracy / loss);
- перевірити модель на випадковому зразку з тесту з відображенням цифри та розподілом ймовірностей.

### 3. Навчити модель через скрипт (без сайту)

```powershell
.venv\Scripts\python.exe train_model.py
```

Навчена модель зберігатиметься у `models/mnist_mlp.keras`.

### 4. Запустити приклад MLP «з нуля» (NumPy)

```powershell
.venv\Scripts\python.exe mlp_numpy.py
```

## Що реалізовано

1. **Бібліотека нейронних мереж:** TensorFlow/Keras (Sequential API, Dense, Flatten, Dropout, Adam optimizer).
2. **Навчання:** forward propagation, backprop через Keras, EarlyStopping, ModelCheckpoint.
3. **Оцінка:** accuracy, classification report, матриця помилок (confusion matrix).
4. **Веб-інтерфейс:** Streamlit (сайт) з інтерактивними параметрами та візуалізацією.
5. **Теоретичний приклад:** MLP «з нуля» на NumPy (файл `mlp_numpy.py`), щоб зрозуміти, як працює навчання всередині бібліотеки.

## Примітка щодо FANN

FANN (Fast Artificial Neural Network) — це бібліотека на C з Python-обгортками, придатна для простих задач. У цьому проєкті обрано TensorFlow/Keras, оскільки це сучасний стандарт для навчання глибоких мереж, має гарну документацію і легко інтегрується з Streamlit-сайтом. За бажанням TensorFlow можна замінити на PyTorch або FANN — логіка навчання залишиться аналогічною.

## Ліцензія

Для навчальних цілей.
