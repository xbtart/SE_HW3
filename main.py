# Разработчик #1 @aleksrf1 aleksrf@gmail.com - Загрузка и настройка модели

from fastapi import FastAPI # импортируем класс `FastAPI` из модуля `fastapi`
from transformers import AutoTokenizer, AutoModelForSequenceClassification # импортируем классы `AutoTokenizer` и `AutoModelForSequenceClassification` из модуля `transformers`
import torch # импортируем модуль `torch`