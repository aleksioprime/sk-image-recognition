import os       # импорт библиотеки для работы с командами ОС
import time     # импорт библиотеки для работы с таймером
import cv2      # импорт библиотеки для работы с компьютерным зрением
import argparse # импорт библиотеки для обработки аргументов командной строки

def collect_images(
        label="test",
        dataset_dir="dataset",
        instances=20,
        n=5,
        crop_square=False,
        interval=2,
        clear_folder=False,
        ):
    """
    Функция для захвата изображений с камеры.
    """
    # Определение пути для сохранения текущего класса
    save_path = os.path.join(dataset_dir, label)

    # Очистка папки, если это указано
    if clear_folder and os.path.exists(save_path):
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    os.makedirs(save_path, exist_ok=True)

    # Определяем начальный номер кадра
    existing_files = [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))]
    if existing_files:
        count = max([int(f.split('-')[-1].split('.')[0]) for f in existing_files if f.startswith(label) and '-' in f and f.split('-')[-1].split('.')[0].isdigit()]) + 1
    else:
        count = 1

    final_count = count + instances - 1  # Определяем конечный номер кадра

    cap = cv2.VideoCapture(0)   # создание объекта камеры для захвата кадров
    text = f"Press S to start collecting data or C to capture manually"  # Сообщение для начала записи
    recording = False           # Флаг начала записи
    check_time = time.time()    # фиксирование текущего состояния таймера

    while True:
        success, img = cap.read()  # Захват кадра
        if not success:
            print("Image error!")
            break

        img = cv2.flip(img, 1)  # Отражение кадра по горизонтали

        if crop_square:
            # Преобразование кадра в квадрат
            height, width = img.shape[:2]
            min_dim = min(height, width)
            top = (height - min_dim) // 2
            left = (width - min_dim) // 2
            img = img[top:top + min_dim, left:left + min_dim]

        if recording:
            # Если состояние таймера подготовки положительное, идёт отсчёт n секунд
            if n > 0:
                if time.time() - check_time > 1:
                    check_time = time.time()
                    n -= 1
                text = f"Preparation... {n}"
            else:
                # Запись кадров с указанным интервалом
                if time.time() - check_time > interval and count <= final_count:
                    text = f"Recording: {count}/{final_count}"
                    cv2.imwrite(os.path.join(save_path, f"{label}-{str(count)}.jpg"), img)
                    count += 1
                    check_time = time.time()
                    # Отображение захваченного кадра
                    cv2.putText(img, "Captured!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Image collection", img)
                    cv2.waitKey(500)  # Задержка на 500 мс для отображения стоп-кадра

                # Завершение записи после нужного количества кадров
                if count > final_count:
                    text = "Dataset collection is complete!"

        # Отображение текста на кадре
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow("Image collection", img)

        # Обработка нажатия клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            if not recording:
                recording = True
                text = "Preparation..."
                check_time = time.time()
        elif key == ord("c"):
            # Ручное сохранение кадра
            cv2.imwrite(os.path.join(save_path, f"{label}-{str(count)}.jpg"), img)
            text = f"Captured manually: {count}"
            count += 1
            cv2.putText(img, "Captured!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow("Image collection", img)
            cv2.waitKey(500)  # Задержка на 500 мс для отображения стоп-кадра
        elif key == ord(' '):  # Выход из программы по пробелу
            break

    # Закрытие всех окон и освобождение камеры
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    # Создаем парсер аргументов
    parser = argparse.ArgumentParser(description="Программа для захвата изображений с камеры")
    parser.add_argument("--label", type=str, default="test", help="Название класса объектов (по умолчанию: 'test')")
    parser.add_argument("--instances", type=int, default=20, help="Количество снимков (по умолчанию: 20)")
    parser.add_argument("--prep_time", type=int, default=5, help="Время подготовки в секундах (по умолчанию: 5)")
    parser.add_argument("--interval", type=int, default=2, help="Интервал между кадрами в секундах (по умолчанию: 2)")
    parser.add_argument("--crop_square", action="store_true", help="Обрезать кадры до квадратного формата")
    parser.add_argument("--clear_folder", action="store_true", help="Очищать папку перед началом записи")

    # Разбираем аргументы командной строки
    args = parser.parse_args()

    # Определение базовой директории для датасета
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Запуск функции захвата изображений
    collect_images(
        label=args.label,
        dataset_dir=DATASET_DIR,
        instances=args.instances,
        n=args.prep_time,
        crop_square=args.crop_square,
        interval=args.interval,
        clear_folder=args.clear_folder,
    )