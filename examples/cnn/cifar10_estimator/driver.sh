CUDA_VISIBLE_DEVICES=5 python -u cifar10_main.py --data-dir=${PWD}/cifar-10-data \
                       --job-dir=${PWD}/tmp \
                       --num-gpus=1 \
