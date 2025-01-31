import os
import time
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, ResNet50, VGG16
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow import lite
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths

# Константы
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CLASSES_CSV_PATH = os.path.join(OUTPUT_DIR, "labels.csv")
PLOT_PATH = os.path.join(OUTPUT_DIR, "plot.png")
DEFAULT_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "model.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "model.tflite")

WIDTH, HEIGHT = 128, 128
EPOCHS = 50
BATCH_SIZE = 32

# Убедимся, что папка для вывода существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Функция загрузки предобученной модели
def load_pretrained_model(base_model_name_or_path, input_shape, num_classes):
    if os.path.isfile(base_model_name_or_path):
        print(f"[INFO] Loading model from file: {base_model_name_or_path}")
        return load_model(base_model_name_or_path)

    print(f"[INFO] Loading pretrained model: {base_model_name_or_path}")

    if base_model_name_or_path == "MobileNetV2":
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name_or_path == "ResNet50":
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif base_model_name_or_path == "VGG16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model name or path: {base_model_name_or_path}")

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model

# Загрузка и обработка датасета
def load_dataset(dataset_path, width, height, verbose=500):
    data, labels = [], []
    image_paths = list(paths.list_images(dataset_path))

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        label = image_path.split(os.path.sep)[-2]
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (width, height), interpolation=cv2.INTER_AREA)
        normalized_image = resized_image.astype(np.float32) / 255.0

        data.append(normalized_image)
        labels.append(label)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(f"[INFO] Processed {i + 1}/{len(image_paths)} images")

    return np.array(data), np.array(labels)

# Сохранение графиков обучения
def save_plot(history, output_path):
    epochs_range = range(len(history.history["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(epochs_range, history.history["loss"], label="train_loss")
    plt.plot(epochs_range, history.history["val_loss"], label="val_loss")
    plt.plot(epochs_range, history.history["accuracy"], label="train_acc")
    plt.plot(epochs_range, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(output_path)

# Сохранение классов в CSV
def save_classes_to_csv(classes, output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for idx, class_name in enumerate(classes):
            writer.writerow([idx, class_name])
    print(f"[INFO] Classes saved to {output_path}")

# Конвертация модели в TFLite
def convert_to_tflite(model, tflite_path):
    print("[INFO] Converting model to TFLite...")
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[INFO] TFLite model saved to {tflite_path}")

# Основная функция
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Pretrained model name or path to model file")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to save/load model weights")
    args = parser.parse_args()

    print("[INFO] Loading dataset...")
    data, labels = load_dataset(DATASET_PATH, WIDTH, HEIGHT)

    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    num_classes = len(lb.classes_)
    print(f"[INFO] Number of classes: {num_classes}")

    save_classes_to_csv(lb.classes_, CLASSES_CSV_PATH)

    input_shape = (HEIGHT, WIDTH, 3)

    if args.model:
        model = load_pretrained_model(args.model, input_shape, num_classes)
    else:
        print("[INFO] No pretrained model specified. Using custom architecture.")
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                             height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

    checkpoint = ModelCheckpoint(args.weights, monitor="val_loss", save_best_only=True, verbose=1)

    print("[INFO] Training model...")
    start_time = time.time()
    history = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        epochs=EPOCHS,
                        callbacks=[checkpoint],
                        verbose=1)
    print(f"[INFO] Training completed in {time.time() - start_time:.2f} seconds")

    print("[INFO] Evaluating model...")
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                target_names=lb.classes_, zero_division=0))

    save_plot(history, PLOT_PATH)

    # Конвертация и сохранение модели в TFLite
    convert_to_tflite(model, TFLITE_PATH)

if __name__ == "__main__":
    main()