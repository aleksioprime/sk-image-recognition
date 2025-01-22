from picamera2 import Picamera2, Preview
import time
import os

def main():
    # Определяем путь к папке
    user_home = os.path.expanduser("~")
    snapshots_dir = os.path.join(user_home, "snapshots")

    # Создаём папку, если её нет
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)
        print(f"Создана папка для снимков: {snapshots_dir}")

    # Инициализация камеры
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration()
    picam2.configure(camera_config)

    # Запуск камеры
    picam2.start_preview(Preview.NULL)
    picam2.start()

    print("Нажмите Enter, чтобы сделать стоп-кадр. Для выхода нажмите Ctrl+C.")

    try:
        while True:
            # Ждем нажатия клавиши Enter
            input("\nНажмите Enter для захвата кадра...")
            timestamp = int(time.time())  # Временная метка для имени файла
            filename = f"snapshot_{timestamp}.jpg"
            filepath = os.path.join(snapshots_dir, filename)
            picam2.capture_file(filepath)
            print(f"Снимок сохранён как {filepath}")
    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
    finally:
        picam2.stop_preview()
        picam2.close()

if __name__ == '__main__':
    main()
