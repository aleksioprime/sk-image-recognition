import io
import time
import numpy as np
import picamera
from PIL import Image
from tflite_runtime.interpreter import Interpreter


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


def main():
    interpreter = Interpreter(model_path='data/model.tflite')
    labels = open("data/labels.txt").read().strip().split("\n")
    labels = [l.split(",")[1] for l in labels]

    interpreter.allocate_tensors()
    _, height, width, _ = interpreter.get_input_details()[0]['shape']
    print("Форма входного слоя модели: {}x{}".format(width, height))

    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((width, height), Image.ANTIALIAS)
                start_time = time.time()
                results = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]
                stream.seek(0)
                stream.truncate()
                camera.annotate_text = '%s %.2f\n%.1fms' % (labels[label_id], prob, elapsed_ms)
        finally:
            camera.stop_preview()


if __name__ == '__main__':
    main()