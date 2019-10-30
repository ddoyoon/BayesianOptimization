CUDA_VISIBLE_DEVICES=0 python ptb_word_lm.py \
    --model "test" \
    --data_path "/home/ddoyoon/dataset/ptb" \
    --num_gpus 1 \
    --strategy "proposed" \
    --verbose
