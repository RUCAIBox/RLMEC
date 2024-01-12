model_path="YOUR_MODEL_PATH"

export CUDA_VISIBLE_DEVICES=0
cd ./ECQA/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file ecqa-0_shot.jsonl \
&> ../log/ecqa-0_shot.log&
cd ..


export CUDA_VISIBLE_DEVICES=1
cd ./QASC/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file qasc-0_shot.jsonl \
&> ../log/qasc-0_shot.log&
cd ..


export CUDA_VISIBLE_DEVICES=2
cd ./OpenbookQA/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file openbookqa-0_shot.jsonl \
&> ../log/openbookqa-0_shot.log&
cd ..


export CUDA_VISIBLE_DEVICES=3
cd ./ARC/
python eval.py \
    --model $model_path \
    --n_shot 0 \
    --result_file ../result/arc_easy-0_shot.jsonl \
&> ../log/arc_easy-0_shot.log&
cd ..


wait
