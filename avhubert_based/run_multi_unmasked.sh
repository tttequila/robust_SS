#!/bin/sh

# gpu_id=4
mask_percentage=0.0
# mask_percentage=0.4
# mask_percentage=0.6
# mask_percentage=0.8
continue_from=
for 
if [ -z ${continue_from} ]; then
    log_name='mask_repeat'
    mkdir logs/$log_name/$mask_percentage/'graded_multi'
else
    log_name=${continue_from}/$mask_percentage/'graded_multi'
fi

# CUDA_VISIBLE_DEVICES="$gpu_id" \
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:1421 --nnodes=1 --nproc_per_node=1 main.py \
--mix_lst_path '/mntnfs/lee_data1/liuqinghua/dataset/lrs3/lrs3_mixture_20k.csv' \
--mixture_direc '/mntnfs/lee_data1/liuqinghua/dataset/lrs3/wav/mixture/' \
--audio_direc '/mntnfs/lee_data1/liuqinghua/dataset/lrs3/wav/' \
--video_direc '/mntnfs/lee_data1/liuqinghua/dataset/lrs3/' \
--mask_type 'unmasked' \
--log_name $log_name \
--batch_size 4 \
--epochs 30 \
--lr 1e-3 \
--use_tensorboard 1 \
--feature_layers 1 6 12 \
--pretrain_grad True \
--mask_percentage $mask_percentage \
>logs/$log_name/console.txt 2>&1
# --continue_from logs/${continue_from} \