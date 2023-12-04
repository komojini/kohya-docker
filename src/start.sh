#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

nvidia-smi
nvcc -V
python3 -c "import onnxruntime as rt; print(rt.get_device())"

echo "runpod-kohya-worker: Start Training"
python3 -u /rp_handler.py