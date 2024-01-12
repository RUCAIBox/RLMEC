export WANDB_MODE=offline
export OMP_NUM_THREADS=24

torchrun --nproc_per_node=4 \
    --master_port=11452 \
    train_grm.py \
    --model_name_or_path YOUR_MODEL_PATH \
    --data_path ../process_data/Gen_Training_Data/result/grm_qa.jsonl \
    --prompt_type simple_inference \
    --bf16 True \
    --output_dir YOUR_SAVE_PATH \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --deepspeed ds_z3_bf16.json \
    --gradient_checkpointing True \
    --tf32 True \
&> logs/GRM_qa.log
