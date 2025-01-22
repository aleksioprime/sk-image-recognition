import os
import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_tensor(interpreter, image):
    """
    Загружает изображение во входной тензор интерпретатора
    """
    # Получаем индекс входного тензора
    tensor_index = interpreter.get_input_details()[0]['index']
    # Получаем ссылку на тензор
    input_tensor = interpreter.tensor(tensor_index)()[0]
    # Загружаем изображение в тензор
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """
    Классифицирует изображение с использованием модели TFLite
    """
    # Загружаем изображение в модель.
    set_input_tensor(interpreter, image)
    # Выполняем предсказание
    interpreter.invoke()
    # Получаем данные из выходного тензора
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    # Преобразуем выходные данные, если они квантованы
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    # Сортируем предсказания по вероятности
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    """
    Основная функция для загрузки модели, настройки камеры и выполнения предсказаний в реальном времени
    """
    # Инициализация интерпретатора TensorFlow Lite
    interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "data", "model.tflite"))

    # Загрузка меток из файла
    labels = open(os.path.join(BASE_DIR, "data", "label.txt")).read().strip().split("\n")
    # Убираем индексы из меток (если есть) и оставляем только названия классов
    labels = [l.split(",")[1] for l in labels]

    # Инициализация тензоров модели
    interpreter.allocate_tensors()
    # Получаем размер входного тензора (ожидаемый размер изображения)
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Форма входного слоя модели: {}x{}".format(width, height))

    # Настройка камеры Picamera2
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()  # Запускаем камеру

    try:
        while True:
            # Захват кадра с камеры в виде массива NumPy
            frame = picam2.capture_array()

            # Преобразование кадра в оттенки серого и изменение его размера до размера входного тензора модели
            image = Image.fromarray(frame).convert('L').resize((width, height), Image.LANCZOS)
            # Добавляем дополнительное измерение для совместимости с входным тензором модели
            image = np.expand_dims(np.array(image), axis=-1)

            # Измеряем время выполнения предсказания
            start_time = time.time()
            # Выполняем классификацию изображения
            results = classify_image(interpreter, image)
            elapsed_ms = (time.time() - start_time) * 1000  # Время в миллисекундах

            # Получаем предсказанный класс и вероятность
            label_id, prob = results[0]
            # Выводим результат классификации
            print(f"Label: {labels[label_id]}, Probability: {prob:.2f}, Inference Time: {elapsed_ms:.1f}ms")

    except KeyboardInterrupt:
        # Остановка программы при нажатии Ctrl+C
        print("Остановка...")

    finally:
        # Останавливаем камеру
        picam2.stop()

# Точка входа в программу
if __name__ == '__main__':
    main()