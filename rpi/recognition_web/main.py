import io
import os
import cv2
import time
import logging
import argparse
import socket
import socketserver
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from threading import Condition
from http import server
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
from libcamera import Transform
from tflite_runtime.interpreter import Interpreter
# import tensorflow.lite as tflite

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Загрузка HTML-шаблона из файла
def load_html_template(filename):
    filepath = os.path.join(BASE_DIR, "template", filename)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"HTML template file '{filepath}' not found.")
        return "<html><body><h1>Error: Template not found.</h1></body></html>"

# Инициализация HTML-шаблона
HTML_TEMPLATE = load_html_template("index.html")

# Функции для классификации изображений
def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def draw_label_on_frame(frame, label, probability, inference_time):
    """Добавляет текстовые метки на кадр."""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font_path = os.path.join(BASE_DIR, 'arial.ttf')  # Укажите путь к шрифту
    try:
        font = ImageFont.truetype(font_path, 20)
    except IOError:
        font = ImageFont.load_default()

    text = f"Label: {label}, Probability: {probability:.2f}, Time: {inference_time:.1f}ms"
    text_position = (10, 10)
    text_color = (255, 255, 255)  # Белый текст
    shadow_color = (0, 0, 0)      # Чёрная тень

    # Рисуем тень
    draw.text((text_position[0] + 1, text_position[1] + 1), text, font=font, fill=shadow_color)
    # Рисуем текст
    draw.text(text_position, text, font=font, fill=text_color)

    return np.array(image)

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class StreamingHandler(server.BaseHTTPRequestHandler):
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
                        frame = output.frame  # Raw bytes from the MJPEG stream

                    # Декодируем байты в NumPy-массив с помощью OpenCV
                    frame_array = np.frombuffer(frame, dtype=np.uint8)
                    frame_image = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

                    if frame_image is None:
                        logging.warning("Failed to decode frame")
                        continue

                    # Обработка изображения
                    rgb_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)
                    processed_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
                    processed_image = (processed_image.astype(np.float32) / 127.5) - 1

                    # Классификация изображения и замер времени операции
                    start_time = time.time()
                    results = classify_image(interpreter, processed_image)
                    elapsed_ms = (time.time() - start_time) * 1000

                    label_id, prob = results[0]
                    label = labels[label_id]

                    # Добавляем подписи на кадр
                    annotated_frame = draw_label_on_frame(frame_image, label, prob, elapsed_ms)

                    # Кодируем обратно в JPEG для стрима
                    _, jpeg_frame = cv2.imencode('.jpg', annotated_frame)

                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(jpeg_frame))
                    self.end_headers()
                    self.wfile.write(jpeg_frame.tobytes())
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                logging.warning(f"Removed streaming client {self.client_address}: Client disconnected.")
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        elif self.path == '/favicon.ico':
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)
            self.end_headers()


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def get_local_ip():
    """Определение локального IP-адреса"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="MJPEG Streaming with PiCamera2")
    parser.add_argument("--flip", choices=["none", "h", "v", "hv"], default="none",
                        help="Set flip mode: 'none' (default), 'h' (horizontal), 'v' (vertical), 'hv' (both)")
    args = parser.parse_args()

    # Инициализация модели
    interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "data", "model.tflite"))
    labels = open(os.path.join(BASE_DIR, "data", "labels.csv")).read().strip().split("\n")
    labels = [l.split(",")[1] for l in labels]

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    logging.info(f"Форма входного слоя модели: {width}x{height}")

    transform = Transform(hflip="h" in args.flip, vflip="v" in args.flip)

    # Настройка камеры
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}), transform=transform)
    output = StreamingOutput()
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    try:
        local_ip = get_local_ip()
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        logging.info(f"Server started on http://{local_ip}:8000")
        server.serve_forever()
    finally:
        picam2.stop_recording()
