import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
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
PLOT_PATH = os.path.join(OUTPUT_DIR, "plot.png")
WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "weights.h5")
TFLITE_PATH = os.path.join(OUTPUT_DIR, "weights.tflite")

WIDTH, HEIGHT = 64, 64
EPOCHS = 30
BATCH_SIZE = 32

# Убедимся, что папка для вывода существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Построение модели
def build_model(width, height, depth, classes):
    model = models.Sequential([
        layers.Conv2D(8, (5, 5), padding="same", input_shape=(height, width, depth)),
        layers.Activation("relu"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(16, (3, 3), padding="same"),
        layers.Activation("relu"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(32, (3, 3), padding="same"),
        layers.Activation("relu"),
        layers.BatchNormalization(axis=-1),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128),
        layers.Activation("relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(classes),
        layers.Activation("softmax")
    ])
    return model

# Загрузка и обработка датасета
def load_dataset(dataset_path, width, height, verbose=500):
    data, labels = [], []
    image_paths = list(paths.list_images(dataset_path))

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        label = image_path.split(os.path.sep)[-2]
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image_array = img_to_array(image)
        data.append(image_array)
        labels.append(label)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print(f"[INFO] Processed {i + 1}/{len(image_paths)} images")

    return np.array(data), np.array(labels)

# Сохранение графиков обучения
def save_plot(history, epochs, output_path):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(output_path)

# Конвертация модели в TFLite
def convert_to_tflite(model, tflite_path):
    print("[INFO] Converting model to TFLite...")
    converter = lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"[INFO] TFLite model saved to {tflite_path}")

# Основная функция
def main():
    print("[INFO] Loading dataset...")
    data, labels = load_dataset(DATASET_PATH, WIDTH, HEIGHT)
    data = data.astype("float") / 255.0

    # Разделение данных на тренировочную и тестовую выборки
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.25, random_state=42)

    print(f"[INFO] Dataset shape: {data.shape}")
    print(f"[INFO] Dataset size: {data.nbytes / (1024 * 1024):.2f} MB")

    # Преобразование меток в one-hot векторы
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.transform(testY)

    num_classes = len(lb.classes_)
    print(f"[INFO] Number of classes: {num_classes}")

    # Сборка и компиляция модели
    model = build_model(WIDTH, HEIGHT, 3, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.01, decay=0.01 / EPOCHS),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    # Настройка генератора данных и обратного вызова
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
                             width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
    checkpoint = ModelCheckpoint(WEIGHTS_PATH, monitor="val_loss", save_best_only=True, verbose=1)

    # Обучение модели
    print("[INFO] Training model...")
    start_time = time.time()
    history = model.fit(aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
                        validation_data=(testX, testY),
                        epochs=EPOCHS, callbacks=[checkpoint], verbose=1)
    print(f"[INFO] Training completed in {time.time() - start_time:.2f} seconds")

    # Оценка модели
    print("[INFO] Evaluating model...")
    predictions = model.predict(testX, batch_size=BATCH_SIZE)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

    # Сохранение графика обучения
    save_plot(history, EPOCHS, PLOT_PATH)

    # Конвертация модели в TFLite
    convert_to_tflite(model, TFLITE_PATH)

if __name__ == "__main__":
    main()
