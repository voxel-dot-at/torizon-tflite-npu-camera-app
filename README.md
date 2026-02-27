# tflite-npu-camera-app

Camera-based SSD object detection using TFLite on Verdin i.MX8MP (Torizon OS, Debian Bookworm).

The detection logic is identical to the tflite-rtsp sample. The only difference is the output: frames are written to a GStreamer display sink instead of an RTSP stream.

## Architecture

Base image `tflite-npu-base` provides all runtime dependencies: `tflite_runtime`, `libvx_delegate.so`, and OpenCV with GStreamer support. The app image is a thin layer that adds only `app.py`, `labelmap.txt`, and the model file.

## Detection pipeline

Mirrors `tflite-rtsp/demos/object-detection/object-detection.py` exactly:

1. Capture via v4l2src MJPEG GStreamer pipeline, BGR output
2. Resize to model input width preserving aspect ratio, letterbox pad with black borders
3. BGR to RGB conversion, cast to uint8, set as input tensor
4. Invoke interpreter with optional VX delegate
5. Decode: `output_details[0]` = locations, `[1]` = classes, `[2]` = scores, `[3]` = n_detections
6. Draw bounding boxes for detections above `MINIMUM_SCORE`, draw inference time overlay
7. Write annotated frame to GStreamer VideoWriter (fbdevsink by default)

## Model

`lite-model_ssd_mobilenet_v1_1_metadata_2.tflite`

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `CAPTURE_DEVICE` | `/dev/video0` | V4L2 device node |
| `CAPTURE_RESOLUTION_X` | `1280` | Capture width in pixels |
| `CAPTURE_RESOLUTION_Y` | `720` | Capture height in pixels |
| `CAPTURE_FRAMERATE` | `30` | Capture framerate |
| `MINIMUM_SCORE` | `0.55` | Detection confidence threshold |
| `USE_HW_ACCELERATED_INFERENCE` | unset | Set to any non-empty value to load the VX delegate |
| `TFLITE_VX_DELEGATE` | `/usr/lib/libvx_delegate.so` | Path to VX delegate shared library |
| `MODEL` | `/app/src/models/lite-model_ssd_mobilenet_v1_1_metadata_2.tflite` | TFLite model path |
| `LABELMAP` | `/app/src/labelmap.txt` | Label map path |
| `DISPLAY_GST` | unset | Custom GStreamer display pipeline; defaults to fbdevsink |

## Build

Building on an amd64 host for the arm64 target device:

```bash
docker build --network=host -t <DOCKERHUB_USERNAME>/tflite-npu-camera-app:<TAG> . 
docker push <DOCKERHUB_USERNAME>/tflite-npu-camera-app:<TAG>

```

## Run on device

```bash
docker run --rm -it --privileged \
  -v /dev:/dev -v /dev/dri:/dev/dri -v /tmp:/tmp -v /run/udev:/run/udev \
  -e CAPTURE_DEVICE=/dev/video2 \
  -e USE_HW_ACCELERATED_INFERENCE=1 \
  -e MINIMUM_SCORE=0.7 \
  <DOCKERHUB_USERNAME>/tflite-npu-camera-app:latest
```

## Run on device with prebuilt image

```bash
docker run --rm -it --privileged \
  -v /dev:/dev -v /dev/dri:/dev/dri -v /tmp:/tmp -v /run/udev:/run/udev \
  -e CAPTURE_DEVICE=/dev/video2 \
  -e USE_HW_ACCELERATED_INFERENCE=1 \
  -e MINIMUM_SCORE=0.7 \
  voxeldotat/tflite-npu-camera-app:latest
```

Adjust `CAPTURE_DEVICE` to the correct V4L2 node on the target device.
