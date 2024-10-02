#!/bin/sh
PARTITION=Segmentation

#GPU_ID=0,1,2,3
GPU_ID=0
dataset=DLRSD # iSAID
#exp_name=split0

arch=tbrnet
net=resnet50 # vgg resnet50


for net in resnet50
do
  for shot in 5
  do
    for split in 2
    do
      exp_dir=exp/${dataset}/${arch}/split${split}/${net}
      snapshot_dir=${exp_dir}/snapshot
      result_dir=${exp_dir}/result
      config=config/${dataset}/${dataset}_split${split}_${net}.yaml
      mkdir -p ${snapshot_dir} ${result_dir}
      now=$(date +"%Y%m%d_%H%M%S")
      cp train.sh train.py ${config} ${exp_dir}

      echo ${arch}
      echo ${config}
      CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train.py \
                  --config=${config} \
                  --arch=${arch} \
                  2>&1 | tee ${result_dir}/train-$now.log
    done
  done
done