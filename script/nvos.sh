
i=0
wandb_group=$1
data_path=/root/data1/jinhyeok/ludvig/dataset/llff_data

gpus=(3 4 5 6 7)
for data in flower  fortress  horns_center horns_left  orchids  trex leaves fern; do
# gpus=(2)
# for data in fern; do
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    echo "gpu:$gpu_id data:$data"
    
    CUDA_VISIBLE_DEVICES=$gpu_id bash script/seg.sh $data $wandb_group dif_NVOS 2>&1 | tee terminal/$data.log &
    # CUDA_VISIBLE_DEVICES=$gpu_id bash script/seg.sh $data $wandb_group dif_NVOS

    i=$((i + 1))
    if [ $gpu_id == ${gpus[-1]} ]; then
        wait
    fi
done

