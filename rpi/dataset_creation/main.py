import os
import io
from PIL import Image
import json
import logging
import shutil
import socketserver
import threading
from http import server
from threading import Condition
from datetime import datetime, timedelta
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

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

class StreamingOutput(io.BufferedIOBase):
    """Класс для передачи кадров MJPEG."""
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        """Записывает новый кадр и уведомляет ожидающие потоки."""
        with self.condition:
            self.frame = buf
            self.condition.notify_all()

class DatasetRecorder:
    def __init__(self, output):
        self.output = output
        self.is_recording = False
        self.total_frames = 0  # Сколько кадров запланировано
        self.captured_frames = 0  # Сколько кадров уже записано
        self.thread = None

    def crop_to_square(self, frame):
        """
        Обрезает изображение до квадратного формата
        """
        image = Image.open(io.BytesIO(frame))
        width, height = image.size
        min_side = min(width, height)
        cropped_image = image.crop((
            (width - min_side) // 2,  # Левая граница
            (height - min_side) // 2,  # Верхняя граница
            (width + min_side) // 2,  # Правая граница
            (height + min_side) // 2   # Нижняя граница
        ))
        return cropped_image

    def start_recording(self, class_name, interval, frame_count):
        if self.is_recording:
            logging.warning("Recording already in progress.")
            return
        self.is_recording = True
        self.total_frames = frame_count
        self.captured_frames = 0
        self.thread = threading.Thread(target=self._record, args=(class_name, interval, frame_count))
        self.thread.start()

    def _record(self, class_name, interval, frame_count):
        folder_path = os.path.join(DATASET_DIR, class_name)

        # Удаление папки, если она уже существует
        if os.path.exists(folder_path):
            logging.info(f"Folder {folder_path} exists. Removing it before starting recording.")
            shutil.rmtree(folder_path)

        # Создание новой папки
        os.makedirs(folder_path, exist_ok=True)

        while self.is_recording and self.captured_frames < frame_count:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{self.captured_frames}_{timestamp}.jpg"
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'wb') as f:
                if self.output.frame:
                    cropped_image = self.crop_to_square(self.output.frame)
                    cropped_image.save(filepath, format='JPEG')
            logging.info(f"Saved frame to {filepath}")
            self.captured_frames += 1
            threading.Event().wait(interval)
        self.is_recording = False

    def stop_recording(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()

    def get_status(self):
        return {
            "is_recording": self.is_recording,
            "captured_frames": self.captured_frames,
            "total_frames": self.total_frames
        }



class StreamingHandler(server.BaseHTTPRequestHandler):
    """Обработчик HTTP-запросов."""
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
                        cropped_frame = recorder.crop_to_square(frame)  # Обрезаем кадр
                        byte_frame = io.BytesIO()
                        cropped_frame.save(byte_frame, format='JPEG')  # Конвертируем в JPEG
                        byte_frame.seek(0)
                        frame_bytes = byte_frame.read()
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                logging.warning(f"Removed streaming client {self.client_address}: Client disconnected.")
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        elif self.path == '/check_status':
            status = recorder.get_status()  # Возвращает словарь с состоянием
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(status).encode('utf-8'))
        else:
            self.send_error(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/start_recording':
            # Чтение тела запроса
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length).decode('utf-8')

            # Логирование полученного тела запроса для отладки
            logging.info(f"Received POST body: {body}")

            # Разбор параметров запроса
            params = dict(param.split('=') for param in body.split('&'))
            logging.info(f"Parsed parameters: {params}")

            # Извлечение параметров
            class_name = params.get('class_name', 'default_class')
            interval = float(params.get('interval', 2))
            frame_count = int(params.get('frame_count', 10))  # Используем frame_count вместо duration

            # Логирование параметров для отладки
            logging.info(f"Starting recording: class_name={class_name}, interval={interval}, frame_count={frame_count}")

            # Запуск записи
            recorder.start_recording(class_name, interval, frame_count)

            # Отправка ответа клиенту
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Recording started.")
        elif self.path == '/stop_recording':
            # Остановка записи
            recorder.stop_recording()

            # Логирование события
            logging.info("Recording stopped.")

            # Отправка ответа клиенту
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Recording stopped.")


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """HTTP-сервер с многопоточной обработкой."""
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Настройка и запуск камеры
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 640)}))  # Устанавливаем квадратное разрешение
    output = StreamingOutput()
    recorder = DatasetRecorder(output)
    picam2.start_recording(JpegEncoder(), FileOutput(output))

    try:
        # Запуск HTTP-сервера
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        logging.info("Server started on http://<your-ip>:8000")
        server.serve_forever()
    finally:
        # Остановка записи
        picam2.stop_recording()
