import os
import io
import logging
import socketserver
from http import server
from threading import Condition
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput

# Определение пути к текущей папке
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Загрузка HTML-шаблона из файла
def load_html_template(filename):
    filepath = os.path.join(BASE_DIR, filename)
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

class StreamingHandler(server.BaseHTTPRequestHandler):
    """Обработчик HTTP-запросов."""
    def do_GET(self):
        if self.path == '/':
            # Перенаправление на index.html
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            # Отправка HTML-страницы
            content = HTML_TEMPLATE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            # Передача MJPEG-потока
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
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning('Removed streaming client %s: %s', self.client_address, str(e))
        else:
            # 404: Страница не найдена
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    """HTTP-сервер с многопоточной обработкой."""
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Настройка и запуск камеры
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
    output = StreamingOutput()
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