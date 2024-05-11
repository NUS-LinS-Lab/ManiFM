export CUDA_VISIBLE_DEVICES=2,3

accelerate launch --config_file configs/acc_config.yaml scripts/train.py 