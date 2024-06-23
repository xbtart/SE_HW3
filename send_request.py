# Разработчик 3: Maksimus1987, отправить POST-запрос на веб-сервер, используя URL и текст,
# чтобы получить предсказание для данного текста. Затем распечатать ответ в формате JSON

import requests  # импортируем модуль `requests`, который предоставляет функции для отправки HTTP-запросов

# создаем переменную `text` и присваивает ей значение `"Я очень удивлен происходящим!"`
text = "Я очень удивлен происходящим!"

# создаем переменную `url` и присваивам ей значение `"http://localhost:8080/predict"`
url = "http://127.0.0.1:8000/predict?text=" + text

# создаем словарь `payload` с ключом `"text"` и значением `text`.
# Словарь `payload` будет преобразован в JSON-формат и передан в качестве тела запроса
payload = {"text": text}

# строка отправляет POST-запрос на указанный URL `url` с использованием модуля `requests`
response = requests.post(url, json=payload)

print(response.json())  # строка распечатывает содержимое ответа в виде JSON
