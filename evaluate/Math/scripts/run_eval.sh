model_path="YOUR_MODEL_PATH"


python run_open.py \
  --model $model_path \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 400 \
  --dataset 'gsm8k' \
  --model_max_length 1500 \
  --cot_backup \
  --print \
  --use_vllm \
  --gpus 1 \
&> log/gsm8k.log

python run_open.py \
  --model $model_path \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 400 \
  --dataset 'math' \
  --model_max_length 1500 \
  --cot_backup \
  --print \
  --use_vllm \
  --gpus 1 \
&> log/math.log

python run_open.py \
  --model $model_path \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 400 \
  --dataset 'svamp' \
  --model_max_length 1500 \
  --cot_backup \
  --print \
  --use_vllm \
  --gpus 1 \
&> log/svamp.log

python run_choice.py \
  --model $model_path \
  --shots 0 \
  --stem_flan_type "" \
  --batch_size 400 \
  --dataset 'mmlu_mathematics' \
  --model_max_length 1500 \
  --cot_backup \
  --print \
  --use_vllm \
&> log/mmlu_mathematics.log
