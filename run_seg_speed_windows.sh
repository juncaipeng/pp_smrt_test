#!/bin/bash
set +x
set -e

gpu_id=0
gpu_name=""
#set CUDA_VISIBLE_DEVICES=${gpu_id}

img_path="../imgs/cityscapes_demo.png"
model_dir='../infer_models_seg'    # the dir of seg inference models

target_width=512                # the width of resized image, which is the input of inference model
target_height=512               # the height of resized image
device=GPU                      # run on GPU or CPU
use_trt=True                    # when device=GPU, whether to use trt
trt_precision=fp32              # when device=GPU and use_trt=True, set trt precision as fp32 or fp16
use_trt_dynamic_shape=True      # when device=GPU and use_trt=True, whether to use dynamic shape mode
use_trt_auto_tune=True          # when device=GPU, use_trt=True and use_trt_dynamic_shape=True, whether to enable auto tune
warmup_iters=30
run_iters=50
save_path="./res_seg.txt"

echo "\n---Config Info---" >> ${save_path}
echo "gpu_id: ${gpu_id}" >> ${save_path}
echo "gpu_name: ${gpu_name}" >> ${save_path}
echo "target_width: ${target_width}" >> ${save_path}
echo "target_height: ${target_height}" >> ${save_path}
echo "device: ${device}" >> ${save_path}
echo "use_trt: ${use_trt}" >> ${save_path}
echo "trt_precision: ${trt_precision}" >> ${save_path}
echo "use_trt_dynamic_shape: ${use_trt_dynamic_shape}" >> ${save_path}
echo "use_trt_auto_tune: ${use_trt_auto_tune}" >> ${save_path}
echo "warmup_iters: ${warmup_iters}" >> ${save_path}
echo "run_iters: ${run_iters}" >> ${save_path}

echo "| model | preprocess time (ms) | run time (ms) | total time (ms) |"  >> ${save_path}

for model in ${model_dir}/*
do
  echo "\n-----------------Test ${model}-----------------"
  ./out/Release/test_seg.exe \
      --model_dir=${model} \
      --img_path=${img_path} \
      --target_width=${target_width} \
      --target_height=${target_height} \
      --device=${device} \
      --use_trt=${use_trt} \
      --trt_precision=${trt_precision} \
      --use_trt_dynamic_shape=${use_trt_dynamic_shape} \
      --use_trt_auto_tune=${use_trt_auto_tune} \
      --warmup_iters=${warmup_iters} \
      --run_iters=${run_iters} \
      --save_path=${save_path}
done

