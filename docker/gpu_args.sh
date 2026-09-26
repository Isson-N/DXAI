#!/usr/bin/env bash
# Source from a launcher after setting DEVICE and, optionally, GPU_MODE.
GPU=()
if [ "$DEVICE" = cuda ]; then
  if [ "${GPU_MODE:-toolkit}" = manual ]; then
    : "${CUDA_DRIVER_LIB:=/lib/x86_64-linux-gnu/libcuda.so.1}"
    for device in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do
      if [ ! -e "$device" ]; then
        echo "CUDA device unavailable: $device" >&2
        exit 1
      fi
      GPU+=(--device "$device")
    done
    if [ ! -f "$CUDA_DRIVER_LIB" ]; then
      echo "CUDA driver library unavailable: $CUDA_DRIVER_LIB" >&2
      exit 1
    fi
    GPU+=(-v "$CUDA_DRIVER_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1:ro")
  elif [ "${GPU_MODE:-toolkit}" = toolkit ]; then
    GPU=(--gpus all)
  else
    echo "Unknown GPU_MODE: $GPU_MODE (expected toolkit or manual)" >&2
    exit 1
  fi
fi
