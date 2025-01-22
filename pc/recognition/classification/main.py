import tensorflow.lite as tflite
import cv2
import os
import numpy as np
import time
import csv

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_tensor(interpreter, image):
    """
    Задает входное изображение для интерпретатора
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """
    Выполняет классификацию изображения и возвращает топ-k результатов
    """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # Обработка квантованного вывода
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    # Сортировка по вероятности
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def load_labels(labels_path):
    """
    Загружает метки из файла CSV
    """
    labels = []
    with open(labels_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            labels.append(row[1])  # Вторая колонка содержит метки
    return labels

def main():
    """
    Главная функция для захвата видео, классификации и отображения результатов
    """
    # Загрузка модели и меток
    model_path = os.path.join(BASE_DIR, "data", "model.tflite")
    labels_path = os.path.join(BASE_DIR, "data", "labels.csv")

    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Получение размеров входного слоя модели
    _, model_height, model_width, _ = interpreter.get_input_details()[0]['shape']
    print(f"Форма входного слоя модели: {model_width}x{model_height}")

    # Загрузка меток
    labels = load_labels(labels_path)

    # Настройка камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return

    cap_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Нажмите пробел для выхода.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Ошибка: Не удалось захватить кадр.")
            break

        # Предобработка изображения
        frame = cv2.flip(frame, 1)  # Отражение по горизонтали
        frame_cut = frame[:, (cap_width - cap_height) // 2:(cap_width - cap_height) // 2 + cap_height]
        image = cv2.resize(frame_cut, (model_width, model_height))
        image = image.astype("float32") / 255.0  # Нормализация
        image = np.expand_dims(image, axis=0)  # Добавление размерности для модели

        # Классификация изображения
        start_time = time.time()
        results = classify_image(interpreter, image)
        elapsed_ms = (time.time() - start_time) * 1000

        # Отображение результатов
        label_id, prob = results[0]
        label_text = f"{labels[label_id]} ({prob * 100:.2f}%)"

        cv2.rectangle(frame, ((cap_width - cap_height) // 2, 0), ((cap_width - cap_height) // 2 + cap_height, cap_height), (255, 0, 0), 3)
        cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"{elapsed_ms:.1f} ms", (10, cap_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Result image", frame)

        # Завершение по нажатию пробела
        if cv2.waitKey(1) == ord(' '):
            print("Завершение работы.")
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
