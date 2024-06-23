# Разработчик #1 @aleksrf1 aleksrf@gmail.com - Загрузка и настройка модели

from fastapi import FastAPI  # импортируем класс `FastAPI` из модуля `fastapi`
# импортируем классы `AutoTokenizer` и `AutoModelForSequenceClassification` из модуля `transformers`
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch  # импортируем модуль `torch`

# Разработчик 2: xbtart, реализация маршрута `/predict` в приложении FastAPI

app = FastAPI()  # создаем экземпляр класса `FastAPI` и присваивает его переменной `app`

# загружаем предобученную модель для классификации последовательностей
# из пакета "cointegrated/rubert-tiny2-cedr-emotion-detection"
# с помощью метода `from_pretrained` класса `AutoModelForSequenceClassification` и присваивает ее переменной `model`
model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rubert-tiny2-cedr-emotion-detection")

# загружаем токенизатор для модели из того же пакета с помощью метода `from_pretrained`
# класса `AutoTokenizer` и присваивает его переменной `tokenizer`
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2-cedr-emotion-detection")


# определяем декоратор `@app.post`, который указывает,
# что следующая функция будет обрабатывать POST-запросы на маршрут "/predict"
@app.post("/predict")
def predict(text: str):  # объявляем функцию `predict`, которая принимает один аргумент `text` типа `str`
    # используем токенизатор `tokenizer` для токенизации текста `text` с помощью метода `tokenizer`
    # и преобразуем его в тензор PyTorch
    inputs = tokenizer(text, return_tensors="pt")
    # строка передает входной тензор `inputs` в модель `model` и получает выходной тензор `outputs`
    outputs = model(**inputs)
    # строка применяет функцию softmax к выходным данным модели,
    # чтобы получить вероятности эмоций. Результат присваивается переменной `probabilities`
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # преобразуем список вероятностей в простой список Python и возвращаем его в качестве ответа на запрос
    return probabilities.tolist()
