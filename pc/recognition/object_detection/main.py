# Импортируем необходимые библиотеки
import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util

# Парсим аргументы командной строки
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Папка, в которой находится .tflite файл модели', default='data')
parser.add_argument('--graph', help='Имя .tflite файла, если оно отличается от model.tflite', default='model.tflite')
parser.add_argument('--labels', help='Имя файла карты меток, если оно отличается от labelmap.txt', default='labelmap.txt')
parser.add_argument('--threshold', help='Минимальный порог доверия для отображения обнаруженных объектов', default=0.5)
parser.add_argument('--edgetpu', help='Использовать Coral Edge TPU для ускорения обнаружения', action='store_true')
args = parser.parse_args()

# Задаём пути и параметры
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.path.join(BASE_DIR, args.modeldir)
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
use_TPU = args.edgetpu

# Импортируем библиотеки TensorFlow Lite
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Если используется Edge TPU, задаём файл модели для TPU
if use_TPU:
    if (GRAPH_NAME == 'model.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'

# Путь к .tflite файлу модели
PATH_TO_CKPT = os.path.join(MODEL_NAME, GRAPH_NAME)

# Путь к файлу карты меток
PATH_TO_LABELS = os.path.join(MODEL_NAME, LABELMAP_NAME)

# Загружаем карту меток
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Удаляем первую метку, если она '???'
if labels[0] == '???':
    del(labels[0])

# Загружаем модель TensorFlow Lite
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Получаем информацию о модели
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Проверяем версию модели (TF1 или TF2) по имени выходного слоя
outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Открываем камеру ноутбука
video = cv2.VideoCapture(0)
imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(video.isOpened()):

    # Считываем кадр с камеры и изменяем размер до формы [1xHxWx3]
    ret, frame = video.read()
    if not ret:
        print('Не удалось захватить кадр с камеры. Завершаем работу.')
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Нормализуем значения пикселей, если используется плавающая модель
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Выполняем детектирование объектов
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Получаем результаты обнаружения
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]

    # Обрабатываем результаты и рисуем рамки на изображении
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Получаем координаты рамки и рисуем её
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 4)

            # Рисуем метку объекта
            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i] * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Отображаем кадр с обнаруженными объектами
    cv2.imshow('Object Detection', frame)

    # Для выхода нажмите 'Пробел'
    if cv2.waitKey(1) == ord(' '):
        break

# Завершаем работу и освобождаем ресурсы
video.release()
cv2.destroyAllWindows()
