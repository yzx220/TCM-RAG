#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 基本参数
MODEL_NAME="BAAI/bge-m3"
OUTPUT_DIR="./output/tcm_bge_m3"
TRAIN_DATA="./data/tcm_data"
TCM_VOCAB_PATH="./data/tcm_vocab.txt"
TCM_TERM_LIST="./data/tcm_terms.txt"

# 模型参数
BATCH_SIZE=32
LEARNING_RATE=2e-5
NUM_EPOCHS=3
MAX_LENGTH=512

# TCM特定参数
ALPHA=0.7      # TCM术语权重系数
LAMBDA1=0.5    # 困惑度损失权重
LAMBDA2=0.3    # 药物属性预测损失权重
HERB_DIM=4     # 药物属性向量维度

# 创建输出目录
mkdir -p $OUTPUT_DIR

python -m torch.distributed.launch --nproc_per_node=4 run.py \
    --model_name_or_path $MODEL_NAME \
    --output_dir $OUTPUT_DIR \
    --train_data $TRAIN_DATA \
    --tcm_vocab_path $TCM_VOCAB_PATH \
    --tcm_term_list $TCM_TERM_LIST \
    --alpha $ALPHA \
    --lambda1 $LAMBDA1 \
    --lambda2 $LAMBDA2 \
    --herb_dim $HERB_DIM \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --max_seq_length $MAX_LENGTH \
    --save_strategy "steps" \
    --save_steps 1000 \
    --logging_steps 100 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --warmup_ratio 0.1 \
    --seed 42 \
    --fp16 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --overwrite_output_dir