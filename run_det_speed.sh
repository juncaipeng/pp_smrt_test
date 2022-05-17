#!/bin/bash
set +x
set -e

# 1. prepare
WITH_GPU=ON
USE_TENSORRT=ON
WITH_MKL=ON

work_path=$(dirname $(readlink -f $0))
paddle_root="${work_path}/paddle_inference"       # the root path of Paddle Inference lib
cuda_root="/usr/local/cuda/lib64"
tensorrt_root="${work_path}/TensorRT-7.1.3.4/" # the root path of TensorRT lib
opencv_root="/usr/local/opencv3"

img_name="det_1500.jpg"           # det_1500.jpg or "det_1024.jpg"
model_dir='infer_models_det'      # the dir of det inference models
device=GPU                        # run on GPU or CPU
use_trt=True                      # when device=GPU, whether to use trt
trt_precision=fp32                # when device=GPU and use_trt=True, set trt precision as fp32 or fp16
use_trt_dynamic_shape=False       # when device=GPU and use_trt=True, whether to use dynamic shape mode. If use_trt_dynamic_shape is True or 
                                  # use_dynamic_shape in infer_cfg.yml is True, this demo will use dynamic shape mode to run inference on TRT.
use_trt_auto_tune=True            # when device=GPU, use_trt=True and use_trt_dynamic_shape=True, whether to enable auto tune
warmup_iters=30
run_iters=50
save_path="res_det.txt"

gpu_id=0
gpu_name=""
#export CUDA_VISIBLE_DEVICES=${gpu_id}

if [ ! -f "${img_name}" ]; then
 wget https://paddle-smrt.bj.bcebos.com/data/demo_imgs/${img_name}
fi

echo "\n---Config Info---" >> ${save_path}
echo "device: ${device}" >> ${save_path}
echo "gpu_id: ${gpu_id}" >> ${save_path}
echo "gpu_name: ${gpu_name}" >> ${save_path}
echo "use_trt: ${use_trt}" >> ${save_path}
echo "trt_precision: ${trt_precision}" >> ${save_path}
echo "use_trt_dynamic_shape: ${use_trt_dynamic_shape}" >> ${save_path}
echo "use_trt_auto_tune: ${use_trt_auto_tune}" >> ${save_path}
echo "warmup_iters: ${warmup_iters}" >> ${save_path}
echo "run_iters: ${run_iters}" >> ${save_path}

echo "| model | resized img w&h | preprocess time (ms) | run time (ms) | total time (ms) |"  >> ${save_path}

# 2. compile
mkdir -p build
cd build
#rm -rf *

cmake .. \
  -DWITH_MKL=${WITH_MKL} \
  -DWITH_GPU=${WITH_GPU} \
  -DUSE_TENSORRT=${USE_TENSORRT} \
  -DWITH_STATIC_LIB=OFF \
  -DPADDLE_LIB=${paddle_root} \
  -DCUDA_LIB=${cuda_root} \
  -DTENSORRT_ROOT=${tensorrt_root} \
  -DOPENCV_ROOT=${opencv_root}

make -j

# 3. run
cd ..

for model in ${model_dir}/*
do
  echo "\n-----------------Test ${model}-----------------"
  ./build/test_det \
      --model_dir=${model} \
      --img_path=${img_name} \
      --device=${device} \
      --use_trt=${use_trt} \
      --trt_precision=${trt_precision} \
      --use_trt_dynamic_shape=${use_trt_dynamic_shape} \
      --use_trt_auto_tune=${use_trt_auto_tune} \
      --warmup_iters=${warmup_iters} \
      --run_iters=${run_iters} \
      --save_path=${save_path}
done
