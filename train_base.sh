#!/bin/sh
PARTITION=Segmentation

GPU_ID=0
dataset=DLRSD
exp_name=split2
arch=PSPNet
net=resnet50 # vgg resnet50

exp_dir=exp/${dataset}/${arch}/${exp_name}/${net}
snapshot_dir=${exp_dir}/snapshot
result_dir=${exp_dir}/result
config=config/${dataset}/${dataset}_${exp_name}_${net}_base.yaml
mkdir -p ${snapshot_dir} ${result_dir}
now=$(date +"%Y%m%d_%H%M%S")
cp train_base.sh train_base.py ${config} ${exp_dir}


echo ${arch}
echo ${config}

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_base.py \
        --config=${config} \
        --arch=${arch} \
        2>&1 | tee ${result_dir}/train-$now.log