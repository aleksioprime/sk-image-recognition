# Справочное руководство для алгоритмов распознавания на PC

[Вернуться на главный README](../README.md)

## Установка pyenv

Для установки PyEnv воспользуйтесь [инструкцией](https://github.com/pyenv/pyenv)

## Создание и запуск среды окружения

Установите нужную версию Python
```
pyenv install 3.9.4
```

Создайте виртуальную среду
```
pyenv virtualenv 3.9.4 recognition
```

Запустите среду
```
pyenv activate recognition
```

Установите библиотеки:
```
python -m pip install --upgrade pip

pip install tensorflow
pip install scikit-learn
pip install imutils
pip install matplotlib
pip install opencv-python
```
## Создание датасета

Запустите программу с выбранными параметрами:
```
python pc/collection_simple/main.py --label "test" --instances 30 --prep_time 10 --interval 1 --crop_square
```

Параметры:

- label: Название класса объектов (по умолчанию: 'test')
- instances: Количество снимков (по умолчанию: 20)
- prep_time: Время подготовки в секундах (по умолчанию: 5)
- interval: Интервал между кадрами в секундах (по умолчанию: 2)
- crop_square: Обрезать кадры до квадратного формата

## Классификация изображений

Запустите программу:
```
python pc/recognition/classification/main.py
```