
i=0
wandb_group=tmp
data_path=/root/data1/jinhyeok/ludvig/dataset/llff_data

gpus=(3)
for data in fern; do
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    echo "gpu:$gpu_id data:$data"
    
    CUDA_VISIBLE_DEVICES=$gpu_id bash script/seg_vggt.sh $data $wandb_group dif_NVOS 2>&1 | tee terminal/$data.log &
    # CUDA_VISIBLE_DEVICES=$gpu_id bash script/seg.sh $data $wandb_group dif_NVOS

    i=$((i + 1))
    if [ $gpu_id == ${gpus[-1]} ]; then
        wait
    fi
done

