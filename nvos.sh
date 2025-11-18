gpus=(1 2 3 4 5 6 7)
i=0


data_path=/root/data1/jinhyeok/ludvig/dataset/llff_data
for data in fern  flower  fortress  horns_center horns_left  orchids  trex leaves; do
    gpu_id=${gpus[$((i % ${#gpus[@]}))]}
    echo "gpu: $gpu_id data: $data"
    
    if [ $data == "leaves" ]; then
        sed -i 's/num_neighbors: 200/num_neighbors: 160/' configs/dif_NVOS.yaml
    fi

    CUDA_VISIBLE_DEVICES=$gpu_id bash script/seg.sh $data dif_NVOS &

    if [ $data == "leaves" ]; then
        sleep 10
        sed -i 's/num_neighbors: 160/num_neighbors: 200/' configs/dif_NVOS.yaml
    fi
    

    i=$((i + 1))
    if [ $gpu_id == ${gpus[-1]} ]; then
        wait
    fi
done

