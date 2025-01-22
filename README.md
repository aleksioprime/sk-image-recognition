# Справочное руководство для алгоритмов распознавания на Raspberry

## Подготовка среды

### Установка Raspberry Pi OS

Скачайте и установите Raspberry Pi Imager ([ссылка](https://www.raspberrypi.com/software/))

Выберите операционную систему для вашей версии Raspberry Pi и задайте параметры:
- имя хоста;
- имя и пароль пользователя;
- SSID и пароль от WiFi
- Включите SSH

После установки узнайте IP-адрес платы и подключитесь к плате по SSH:
```
ssh <имя пользователя>@<IP-адрес>
```

Обновите список доступных пакетов Linux:
```
sudo apt update
```

Установите обновления для всех пакетов, которые можно обновить без удаления или замены других пакетов:
```
sudo apt upgrade -y
```

### Подключение по VNC

В терминале откройте настройки Raspberry Pi OS:
```
sudo raspi-config
```

Активируйте `Interface Options` => `I3 VNC` => `YES`

Подключитесь к VNC с помощью программы VNC Viewer ([ссылка](https://www.realvnc.com/en/connect/download/viewer/))

### Подготовка камеры

Установите драйвера камеры:
```
sudo apt install -y python3-picamera2
```

Установите зависимости для поддержки библиотеки камеры
```
sudo apt install -y python3-libcamera python3-kms++
sudo apt install -y python3-prctl libatlas-base-dev ffmpeg python3-pip
sudo apt install -y python3-pyqt5 python3-opengl # only if you want GUI features
```

Проверьте работу камеры:
```
sudo rpicam-hello
sudo libcamera-hello
```

Документация по библиотеке:
https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf

### Установка альтернативной версии Python (опционально)

Установите зависимости:
```
sudo apt update
sudo upgrade
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev liblzma-dev

sudo apt install -y libcap-dev
```

Проверьте текущую версию Python:
```
python --version
```

Скачайте исходный код Python 3.7.9 с официального сайта:
```
sudo wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
```

Распакуйте скачанный архив:
```
sudo tar xzf Python-3.7.9.tgz
```

Перейдите в директорию с распакованным исходным кодом, выполните сборку и установку:
```
cd Python-3.7.9
```

Подготовка к сборке пакетов с оптимизацией:
```
sudo ./configure --enable-optimizations
```

Запустите сборку Python (с параллельными процессами по количеству ядер процессора):
```
sudo make -j$(nproc)
```

Установите собранный версию Python (не заменяя системную)
```
sudo make altinstall
```

Проверьте каталог установленной версии Python:
```
whereis python3.7
```

Установите новую версию Python по-умолчанию для команды python:
```
sudo update-alternatives --install /usr/bin/python python /usr/local/bin/python3.7 1
sudo update-alternatives --config python
```

Проверьте новую версию Python:
```
python --version
```

*Дополнительные команды:*
Добавляет новую версию Python (3.7 в данном случае) в систему как альтернативу для команды python:
```
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
```
Удаляет указанную версию Python (/usr/local/bin/python3.7) из списка альтернатив для команды python:
```
sudo update-alternatives --remove python /usr/local/bin/python3.7
```

## Создание виртуального окружения virtualenv:

Установите среду виртуального окружения:
```
sudo apt install python3-venv
```

Создайте виртуальное окружение:
```
python -m venv --system-site-packages venv
```

Запустите виртуальное окружение
```
source venv/bin/activate
```

Для деактивации виртуального окружения:
```
deactivate
```

Установите необходимые библиотеки:
```
pip install -r requirements.txt
```

### Подготовка среды разработки

Установите VSCode ([ссылка](https://code.visualstudio.com/download))

Установите расширение Remote SSH в VSCode

Перейдите на вкладку "Удалённый обозреватель" и выберите "Новый удалённый репозиторий". В строке ввода введите подключение по SSH:
```
ssh <имя пользователя>@<адрес хоста>
```
Затем подтвердите подкючение и введите пароль от пользователя

Выберите папку проекта и дождитесь установки VSCode Server и расширения Python

Можете посмотреть конфигурацию подключения (значок шестерёнки):
```
Host 192.168.2.107
  HostName 192.168.2.107
  User your_username
```

## Запуск тестовых программ

```
wget https://github.com/aleksioprime/raspberry_examples/archive/refs/heads/main.zip
unzip main.zip
cd raspberry_examples-main
```

```
python examples/camera/hello.py
```

## Полезные команды

Проверить установленные пакеты:
```
pip list
```

Заморозить список зависимостей в файл:
```
pip freeze > requirements.txt
```

Установить зависимости из файла:
```
pip install -r requirements.txt
```

### Удаление старых ключей (в случае ошибки MitmPortForwardingDisabled)

Вариант 1. Откройте файл known_host. Найдите строку, связанную с IP-адресом Raspberry Pi, и удалите её:
```
nano ~/.ssh/known_hosts
```

Вариант 2. Используйте команду для удаления ключа по адресу хоста Raspberry:
```
ssh-keygen -R <IP-адрес_или_хост_Raspberry>
```