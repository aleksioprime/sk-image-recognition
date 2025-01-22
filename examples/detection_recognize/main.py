#!/usr/bin/python3

import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2
import argparse

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Размеры изображений для работы с камерой
NORMAL_SIZE = (640, 480)
LOWRES_SIZE = (320, 240)
rectangles = []  # Хранилище прямоугольников для отображения

# Функция для чтения файла с метками объектов
def read_label_file(file_path):
    """Читает файл с метками и возвращает словарь {id: label}"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1].strip() for line in lines}

# Функция для выполнения инференса с использованием TensorFlow Lite
def inference_tensorflow(image, model_path, label_path=None):
    """
    Обрабатывает изображение и выполняет инференс
    """
    global rectangles

    # Загрузка меток, если они указаны
    labels = read_label_file(label_path) if label_path else None

    # Инициализация интерпретатора TensorFlow Lite
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()

    # Получение информации о входных/выходных тензорах
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Подготовка изображения
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Преобразуем в RGB
    resized_image = cv2.resize(rgb_image, (input_shape[2], input_shape[1]))

    # Преобразуем данные в ожидаемый формат
    if input_dtype == np.float32:
        input_data = (resized_image.astype(np.float32) - 127.5) / 127.5
    elif input_dtype == np.uint8:
        input_data = resized_image.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected input data type: {input_dtype}")

    input_data = np.expand_dims(input_data, axis=0)

    # Выполнение инференса
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Получение результатов
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = int(interpreter.get_tensor(output_details[3]['index'])[0])

    # Обработка результатов
    rectangles = []
    for i in range(num_boxes):
        if detected_scores[0][i] > 0.5:  # Фильтруем по порогу достоверности
            top, left, bottom, right = detected_boxes[0][i]
            box = [left * LOWRES_SIZE[0], bottom * LOWRES_SIZE[1],
                   right * LOWRES_SIZE[0], top * LOWRES_SIZE[1]]
            if labels:
                box.append(labels[int(detected_classes[0][i])])
            rectangles.append(box)
    return rectangles

# Основная функция
def main():
    """
    Инициализирует камеру и запускает процесс инференса
    """
    parser = argparse.ArgumentParser(description="TensorFlow Lite Object Detection")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (for SSH)")
    args = parser.parse_args()

    model_path = os.path.join(BASE_DIR, "data", "mobilenet_v2.tflite")  # Путь к модели
    label_path = os.path.join(BASE_DIR, "data", "coco_labels.txt")  # Путь к меткам

    picam2 = Picamera2()  # Инициализация камеры
    config = picam2.create_preview_configuration(main={"size": NORMAL_SIZE},
                                                 lores={"size": LOWRES_SIZE, "format": "YUV420"})
    picam2.configure(config)
    picam2.start()

    # Основной цикл обработки изображений
    while True:
        buffer = picam2.capture_buffer("lores")  # Получаем буфер низкого разрешения
        grey_image = buffer[:LOWRES_SIZE[1] * config["lores"]["stride"]].reshape((LOWRES_SIZE[1], -1))
        results = inference_tensorflow(grey_image, model_path, label_path)  # Выполняем инференс

        if args.headless:
            # Вывод результатов в консоль
            for result in results:
                print(f"Detected: {result}")
        else:
            # Здесь можно добавить код для отображения графического окна, если требуется
            pass

if __name__ == "__main__":
    main()