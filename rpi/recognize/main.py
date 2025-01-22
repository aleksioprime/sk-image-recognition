import os
import time
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import argparse
import cv2  # Для графического режима

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def set_input_tensor(interpreter, image):
    """
    Загружает изображение во входной тензор интерпретатора
    """
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """
    Классифицирует изображение с использованием модели TFLite
    """
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def main():
    """
    Основная функция для загрузки модели, настройки камеры и выполнения предсказаний
    """
    parser = argparse.ArgumentParser(description="Real-time object classification")
    parser.add_argument("--headless", action="store_true", help="Run without GUI (console output)")
    args = parser.parse_args()

    # Инициализация интерпретатора TensorFlow Lite
    interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "data", "model.tflite"))

    # Загрузка меток
    labels = open(os.path.join(BASE_DIR, "data", "labels.txt")).read().strip().split("\n")
    labels = [l.split(",")[1] for l in labels]

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Форма входного слоя модели: {}x{}".format(width, height))

    # Настройка камеры
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()

            # Преобразование кадра для модели
            image = Image.fromarray(frame).convert('L').resize((width, height), Image.LANCZOS)
            image = np.expand_dims(np.array(image), axis=-1)

            start_time = time.time()
            results = classify_image(interpreter, image)
            elapsed_ms = (time.time() - start_time) * 1000

            label_id, prob = results[0]
            result_text = f"Label: {labels[label_id]}, Probability: {prob:.2f}, Time: {elapsed_ms:.1f}ms"

            if args.headless:
                # Вывод результата в консоль
                print(result_text)
            else:
                # Графический режим: отображение на кадре
                annotated_frame = frame.copy()
                cv2.putText(annotated_frame, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Real-time Classification", annotated_frame)

                # Выход по нажатию клавиши 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Остановка...")

    finally:
        picam2.stop()
        if not args.headless:
            cv2.destroyAllWindows()

# Точка входа в программу
if __name__ == '__main__':
    main()