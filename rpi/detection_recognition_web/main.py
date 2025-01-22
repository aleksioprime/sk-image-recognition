import os
import io
import cv2
import numpy as np
import logging
import socketserver
from threading import Condition
from http import server
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from tflite_runtime.interpreter import Interpreter

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Размеры изображений для работы с камерой
NORMAL_SIZE = (640, 480)  # Основное разрешение
LOWRES_SIZE = (320, 240)  # Разрешение для инференса
rectangles = []  # Хранилище прямоугольников для отображения

# Загрузка HTML-шаблона из файла
def load_html_template(filename):
    """
    Читает HTML-шаблон для веб-страницы
    """
    filepath = os.path.join(BASE_DIR, "template", filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"HTML template file '{filepath}' not found.")
        return "<html><body><h1>Error: Template not found.</h1></body></html>"

# Инициализация HTML-шаблона
HTML_TEMPLATE = load_html_template("index.html")

# Функция для чтения файла с метками объектов
def read_label_file(file_path):
    """
    Читает файл с метками и возвращает словарь {id: label}
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return {int(line.split(maxsplit=1)[0]): line.split(maxsplit=1)[1].strip() for line in lines}

# Функция для отрисовки прямоугольников вокруг обнаруженных объектов
def draw_rectangles(image):
    """
    Рисует прямоугольники и метки объектов на изображении
    """
    for rect in rectangles:
        rect_start = (int(rect[0] * 2) - 5, int(rect[1] * 2) - 5)
        rect_end = (int(rect[2] * 2) + 5, int(rect[3] * 2) + 5)
        cv2.rectangle(image, rect_start, rect_end, (0, 255, 0), 2)  # Зеленый прямоугольник
        if len(rect) == 5:
            text = rect[4]
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, text, (rect_start[0] + 10, rect_start[1] + 10),
                        font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Функция для выполнения инференса с использованием TensorFlow Lite
def inference_tensorflow(image, model_path, label_path=None):
    """
    Выполняет инференс для входного изображения
    """
    global rectangles

    labels = read_label_file(label_path) if label_path else None
    interpreter = Interpreter(model_path=model_path, num_threads=4)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    # Преобразование изображения
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(rgb_image, (input_shape[2], input_shape[1]))

    # Нормализация данных
    if input_dtype == np.float32:
        input_data = (resized_image.astype(np.float32) - 127.5) / 127.5
    elif input_dtype == np.uint8:
        input_data = resized_image.astype(np.uint8)
    else:
        raise ValueError(f"Unexpected input data type: {input_dtype}")

    input_data = np.expand_dims(input_data, axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Получение результатов инференса
    detected_boxes = interpreter.get_tensor(output_details[0]['index'])
    detected_classes = interpreter.get_tensor(output_details[1]['index'])
    detected_scores = interpreter.get_tensor(output_details[2]['index'])
    num_boxes = int(interpreter.get_tensor(output_details[3]['index'])[0])

    rectangles = []
    for i in range(num_boxes):
        if detected_scores[0][i] > 0.5:  # Порог достоверности
            top, left, bottom, right = detected_boxes[0][i]
            box = [left * LOWRES_SIZE[0], bottom * LOWRES_SIZE[1],
                   right * LOWRES_SIZE[0], top * LOWRES_SIZE[1]]
            if labels:
                box.append(labels[int(detected_classes[0][i])])
            rectangles.append(box)

class StreamingOutput(io.BufferedIOBase):
    """
    Хранилище для кадров MJPEG-потока
    """
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
    """
    Обработчик HTTP-запросов для стриминга MJPEG
    """
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = HTML_TEMPLATE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame

                    frame_array = np.frombuffer(frame, dtype=np.uint8)
                    frame_image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame_image is None:
                        logging.warning("Failed to decode frame")
                        continue

                    # Выполняем инференс и отрисовываем прямоугольники
                    gray_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY)
                    resized_image = cv2.resize(gray_image, (NORMAL_SIZE[0], NORMAL_SIZE[1]), interpolation=cv2.INTER_AREA)
                    inference_tensorflow(resized_image, model_path, label_path)
                    draw_rectangles(frame_image)

                    # Кодируем изображение в JPEG для стриминга
                    _, jpeg_frame = cv2.imencode('.jpg', frame_image)

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg_frame))
                    self.end_headers()
                    self.wfile.write(jpeg_frame.tobytes())
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                logging.warning(f"Removed streaming client {self.client_address}: Client disconnected.")
            except Exception as e:
                logging.error(f"Error streaming to client {self.client_address}: {e}")
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """
    Сервер с поддержкой многопоточной обработки запросов
    """
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    model_path = os.path.join(BASE_DIR, "data", "mobilenet_v2.tflite")  # Путь к модели
    label_path = os.path.join(BASE_DIR, "data", "coco_labels.txt")  # Путь к файлу меток

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": NORMAL_SIZE}))
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        logging.info("Server started on http://<your-ip>:8000")
        server.serve_forever()
    finally:
        picam2.stop_recording()
