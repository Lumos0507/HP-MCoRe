#!/usr/bin/env sh
mkdir -p logs
now=$(date +"%m%d_%H%M")
DN="w_DN" # w_DN, wo_DN
config_name="$DN/$2_$3_$now"
log_name="$DN_$2_$3_$now"


CUDA_VISIBLE_DEVICES=$4 nohup python3 -u main.py --benchmark $1 --prefix $config_name --feature_arch $2 --temporal_arch $3 $5 2>&1|tee logs/$log_name.log  2>&1 &


