import os
import shutil
import random
import argparse

def split_dataset(source_dir, target_dir, train_ratio=0.8):
    """
    Функция для разделения датасета на обучающий (train) и тестовый (test) наборы.
    """
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Получаем список всех изображений и xml файлов
    files = [f for f in os.listdir(source_dir) if f.endswith(".jpg")]
    files.sort()  # Сортируем, чтобы пары jpg/xml были упорядочены

    # Перемешиваем файлы случайным образом
    random.shuffle(files)

    # Определяем границу для разделения
    split_index = int(len(files) * train_ratio)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # Функция для копирования файла с соответствующим XML
    def move_files(file_list, destination):
        for file in file_list:
            base_name = os.path.splitext(file)[0]
            jpg_path = os.path.join(source_dir, f"{base_name}.jpg")
            xml_path = os.path.join(source_dir, f"{base_name}.xml")

            shutil.copy(jpg_path, os.path.join(destination, f"{base_name}.jpg"))
            if os.path.exists(xml_path):
                shutil.copy(xml_path, os.path.join(destination, f"{base_name}.xml"))

    # Копируем файлы в соответствующие папки
    move_files(train_files, train_dir)
    move_files(test_files, test_dir)

    print(f"Разделение завершено: {len(train_files)} в train, {len(test_files)} в test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Разделение датасета на train и test")
    parser.add_argument("--source", type=str, required=True, help="Путь к исходной папке с изображениями и аннотациями")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Доля обучающего набора (по умолчанию 0.8)")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", "images")
    os.makedirs(DATASET_DIR, exist_ok=True)

    split_dataset(args.source, DATASET_DIR, train_ratio=args.train_ratio)