import os
import cv2
import numpy as np
from time import time
import tflite_runtime.interpreter as tf

USE_HW_ACCELERATED_INFERENCE = os.environ.get("USE_HW_ACCELERATED_INFERENCE")

MINIMUM_SCORE = os.environ.get("MINIMUM_SCORE")
if not MINIMUM_SCORE:
    MINIMUM_SCORE = 0.55
MINIMUM_SCORE = float(MINIMUM_SCORE)

CAPTURE_DEVICE = os.environ.get("CAPTURE_DEVICE")
if not CAPTURE_DEVICE:
    CAPTURE_DEVICE = "/dev/video0"

CAPTURE_RESOLUTION_X = os.environ.get("CAPTURE_RESOLUTION_X")
if not CAPTURE_RESOLUTION_X:
    CAPTURE_RESOLUTION_X = 1280
CAPTURE_RESOLUTION_X = int(CAPTURE_RESOLUTION_X)

CAPTURE_RESOLUTION_Y = os.environ.get("CAPTURE_RESOLUTION_Y")
if not CAPTURE_RESOLUTION_Y:
    CAPTURE_RESOLUTION_Y = 720
CAPTURE_RESOLUTION_Y = int(CAPTURE_RESOLUTION_Y)

CAPTURE_FRAMERATE = os.environ.get("CAPTURE_FRAMERATE")
if not CAPTURE_FRAMERATE:
    CAPTURE_FRAMERATE = 30
CAPTURE_FRAMERATE = int(CAPTURE_FRAMERATE)

TFLITE_VX_DELEGATE = os.environ.get("TFLITE_VX_DELEGATE")
if not TFLITE_VX_DELEGATE:
    TFLITE_VX_DELEGATE = "/usr/lib/libvx_delegate.so"

DISPLAY_GST = os.environ.get("DISPLAY_GST")

MODEL_PATH = os.environ.get("MODEL")
if not MODEL_PATH:
    MODEL_PATH = "/app/src/models/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite"

LABELMAP_PATH = os.environ.get("LABELMAP")
if not LABELMAP_PATH:
    LABELMAP_PATH = "/app/src/labelmap.txt"


def draw_bounding_boxes(img, labels, x1, x2, y1, y2, object_class):
    box_colors = [(254, 153, 143), (253, 156, 104), (253, 157, 13), (252, 204, 26),
                  (254, 254, 51), (178, 215, 50), (118, 200, 60), (30, 71, 87),
                  (1, 48, 178), (59, 31, 183), (109, 1, 142), (129, 14, 64)]

    text_colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
                   (0, 0, 0), (0, 0, 0), (0, 0, 0), (255, 255, 255),
                   (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]

    cv2.rectangle(img, (x2, y2), (x1, y1),
                  box_colors[object_class % len(box_colors)], 2)
    cv2.rectangle(img, (x1 + len(labels[object_class]) * 10, y1 + 15), (x1, y1),
                  box_colors[object_class % len(box_colors)], -1)
    cv2.putText(img, labels[object_class], (x1, y1 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                text_colors[object_class % len(text_colors)], 1, cv2.LINE_AA)


def main():
    if USE_HW_ACCELERATED_INFERENCE:
        delegates = [tf.load_delegate(TFLITE_VX_DELEGATE)]
    else:
        delegates = []

    with open(LABELMAP_PATH, "r") as f:
        labels = f.read().splitlines()

    interpreter = tf.Interpreter(model_path=MODEL_PATH,
                                 experimental_delegates=delegates)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]["shape"][1]

    cap = cv2.VideoCapture(
        f'v4l2src device={CAPTURE_DEVICE} extra-controls="controls,horizontal_flip=1,vertical_flip=1" '
        f'! image/jpeg,width={CAPTURE_RESOLUTION_X},height={CAPTURE_RESOLUTION_Y},framerate={CAPTURE_FRAMERATE}/1 '
        f'! jpegdec ! videoconvert primaries-mode=fast n-threads=4 '
        f'! video/x-raw,format=BGR '
        f'! appsink drop=1 max-buffers=1',
        cv2.CAP_GSTREAMER
    )

    display_pipeline = DISPLAY_GST or (
        "appsrc is-live=true do-timestamp=true format=time "
        "! queue leaky=downstream max-size-buffers=1 "
        "! videoconvert "
        "! fbdevsink sync=false"
    )

    writer = cv2.VideoWriter(
        display_pipeline,
        cv2.CAP_GSTREAMER,
        0,
        float(CAPTURE_FRAMERATE),
        (CAPTURE_RESOLUTION_X, CAPTURE_RESOLUTION_Y),
        True
    )

    while cap.isOpened():
        ret, image_original = cap.read()
        if not ret:
            continue

        height1 = image_original.shape[0]
        width1 = image_original.shape[1]

        image = cv2.resize(image_original,
                           (input_size, int(input_size * height1 / width1)),
                           interpolation=cv2.INTER_NEAREST)
        height2 = image.shape[0]
        scale = height1 / height2
        border_top = int((input_size - height2) / 2)
        image = cv2.copyMakeBorder(image,
                                   border_top,
                                   input_size - height2 - border_top,
                                   0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        inp = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)], dtype=np.uint8)
        interpreter.set_tensor(input_details[0]["index"], inp)

        t1 = time()
        interpreter.invoke()
        t2 = time()

        locations = (interpreter.get_tensor(output_details[0]["index"])[0] * width1).astype(int)
        locations[locations < 0] = 0
        locations[locations > width1] = width1
        classes = interpreter.get_tensor(output_details[1]["index"])[0].astype(int)
        scores = interpreter.get_tensor(output_details[2]["index"])[0]
        n_detections = interpreter.get_tensor(output_details[3]["index"])[0].astype(int)

        img = image_original
        for i in range(n_detections):
            if scores[i] > MINIMUM_SCORE:
                y1 = locations[i, 0] - int(border_top * scale)
                x1 = locations[i, 1]
                y2 = locations[i, 2] - int(border_top * scale)
                x2 = locations[i, 3]
                draw_bounding_boxes(img, labels, x1, x2, y1, y2, classes[i])

        cv2.rectangle(img, (0, 0), (130, 20), (255, 0, 0), -1)
        cv2.putText(img, "inf time: %.3fs" % (t2 - t1), (0, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)

    cap.release()
    writer.release()


if __name__ == "__main__":
    main()
