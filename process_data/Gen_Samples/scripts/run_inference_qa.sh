NUM_GPU=4
NUM_PROBLEM=15360
NUMBER_PER_PROCESS=$((NUM_PROBLEM/NUM_GPU))

NewStartIDX=(0 0 0 0)

CUDAid=(0 1 2 3)

date

for ((i=0;i<$NUM_GPU;i++)) do
{
    startidx=$(((i*NUMBER_PER_PROCESS)))
	endidx=$((((i+1)*NUMBER_PER_PROCESS)))
    cuda0idx=$i
    rseed=$((i+2023))
    echo $startidx $endidx
    python inference.py \
        --start_idx $startidx \
        --end_idx $endidx \
        --batch_size 16 \
        --model_path YOUR_MODEL_PATH \
        --data_name qa \
        --data_path data/qa.jsonl  \
        --target_path result/qa/result-$startidx-$endidx.jsonl \
        --cuda_device ${CUDAid[$cuda0idx]} \
        --write_mode w \
        --seed $rseed \
    &> log/qa-part$i.log
}&
done

wait

date
