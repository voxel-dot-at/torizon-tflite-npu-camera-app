# tflite-npu-camera-app/Dockerfile
# Thin app image: includes only your app code/models; all heavy deps come from tflite-npu-base.

ARG TFLITE_NPU_BASE_IMAGE=docker.io/voxeldotat/tflite-npu-base:latest
FROM ${TFLITE_NPU_BASE_IMAGE}
ARG IMAGE_ARCH=linux/arm64/v8

WORKDIR /app
COPY src /app/src

ENV PYTHONUNBUFFERED=1

# Fail fast by default: this app should not silently run on CPU when HW accel is expected.
ENV USE_HW_ACCELERATED_INFERENCE=1 \
	REQUIRE_HW_ACCELERATED_INFERENCE=1

ENTRYPOINT ["python3", "-u", "/app/src/app.py"]

# NOTE (build/push): this image must be built for the same platform as the target device.
# If you're building on an amd64 host for an arm64 Torizon device, use buildx:
#   docker buildx build --platform linux/arm64 -t <repo>:<tag> --push .
