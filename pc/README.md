# Справочное руководство для алгоритмов распознавания на PC

[Вернуться на главный README](../README.md)

## Установка pyenv

Для установки PyEnv воспользуйтесь [инструкцией](https://github.com/pyenv/pyenv)

```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
```

```
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

```
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win-venv/main/bin/install-pyenv-win-venv.ps1" -OutFile "$HOME\install-pyenv-win-venv.ps1";
&"$HOME\install-pyenv-win-venv.ps1"
```

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

## Обучение классификатора

### Сбор данных

Для сбора данных используется программа:
```
python pc/collection_simple/main.py --label "images" --instances 20 --prep_time 5 --interval 2 --crop_square --clear_folder
```

Параметры:

- **--instances**: Количество снимков (по умолчанию: 20)
- **--prep_time**: Время подготовки в секундах (по умолчанию: 5)
- **--interval**: Интервал между кадрами в секундах (по умолчанию: 2)
- **--crop_square**: Обрезать кадры до квадратного формата
- **--clear_folder**: Очищать папку перед началом записи

Старт автоматических стоп-кадров: клавиша S
Сделать ручные стоп-кадры: клавиша C
Закрытие программы: клавиша Пробел

### Обучение

```
python pc/training/classification/main.py
```

### Запуск программы

Запустите программу классификации изображений:
```
python pc/recognition/classification/main.py
```

## Обучение детектора

### Сбор данных

Для сбора данных используется программа:
```
python pc/collection_simple/main.py --label "images" --instances 20 --prep_time 5 --interval 2
```

Параметры:

- **--instances**: Количество снимков (по умолчанию: 20)
- **--prep_time**: Время подготовки в секундах (по умолчанию: 5)
- **--interval**: Интервал между кадрами в секундах (по умолчанию: 2)
- **--crop_square**: Обрезать кадры до квадратного формата
- **--clear_foldere**: Очищать папку перед началом записи

Старт автоматических стоп-кадров: клавиша S
Сделать ручные стоп-кадры: клавиша C
Закрытие программы: клавиша Пробел

### Установка LabelImg:

Выполните команду
```
pip install labelImg
```

Для запуска программы используйте:
```
labelImg
```

В окне программы нажмите на кнопку **Open Dir** и в диалоговом окне выберите директорию с изображениями (*pc/collection_simple/dataset/images*), чтобы добавить их в список, а затем на кнопку **Change Save Dir** и также выберите директорию с изображениями, чтобы указать куда будут сохраняться файлы разметки.

После открытия фотографии в программе **labelImg** нажмите на кнопку **Create RectBox** и выделите фрагмент изображения с интересующим вас объектом. После выделения появится диалоговое окно, в которое введите название метки на латинице. Затем нажмите на кнопку **ОК** и переходите к следующему изображению (кнопка **Next Image**). Обратите внимание, у вас должна быть активирована функция **Auto Save mode** из верхнего пункта меню **View**.

В результате в папке с изображениями около каждого файла с фотографией должен появиться парный файл разметки формата **xml**


### Копирование датасета:

```
python pc/training/object_detection/split.py --source pc/collection_simple/dataset/images --train_ratio 0.8
```

### Обучение модели в Jupyter Notebook на Google Colab:

https://colab.research.google.com/drive/1YFzorps2TT_zWAJxmXOTaYnpcELPFM0Y?usp=sharing

### Подготовка моделей

Скопируйте файл модели model.tflite в папку **pc/recognition/object_detection/date**

Создайте файл *labelmap.txt* в папке **pc/recognition/object_detection/date**


### Запуск программы

Запустите программу детектирования и распознавания объектов

```
python pc/recognition/object_detection/main.py
```