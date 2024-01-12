NUM_GPU=4
NUM_PROBLEM=590440
NUMBER_PER_PROCESS=$((NUM_PROBLEM/NUM_GPU))

CUDAid=(0 1 2 3)

for ((i=0;i<$NUM_GPU;i++)) do
{
    startidx=$(((i*NUMBER_PER_PROCESS)))
    endidx=$((((i+1)*NUMBER_PER_PROCESS)))
    cuda0idx=$i
    echo $startidx $endidx
    python miniedit_grm_math.py \
        --gen_data_folder ../Gen_Samples/result/math \
        --src_data_path ../Gen_Samples/data/math.jsonl  \
        --result_path result/math_grm/result-$startidx-$endidx.jsonl \
        --write_mode w \
        --start_idx $startidx \
        --end_idx $endidx \
        --cuda_device ${CUDAid[$cuda0idx]} \
    &> log/math_grm-part$i.log
}&
done
wait

echo "Finish!!!"

