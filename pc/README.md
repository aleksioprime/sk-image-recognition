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
