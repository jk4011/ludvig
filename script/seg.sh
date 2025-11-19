#!/bin/bash
scene=$1
wandb_group=$2

it="30000"
src_path="SPIn-NeRF_data"
[[ "$scene" == @(fern|orchids|trex|horns|flower|fortress|leaves|room|horns_center|horns_left) ]] && src_path="llff_data"
height=1199
width=1600
scene_tag=${scene}
if [ "$scene" = "fork" ]; then
    height=1202
elif [ "$scene" = "room" ]; then
    height=1200
elif [ "$scene" = "truck" ]; then
    width=979; height=546
elif [ "$scene" = "lego" ]; then
    width=1015; height=764
elif [ "$scene" = "horns_center" ]; then
    scene="horns"
elif [ "$scene" = "horns_left" ]; then
    scene="horns"
elif [ "$scene" = "trex" ]; then
    width=1599
fi

python ludvig_uplift.py \
    --colmap_dir ./dataset/${src_path}/$scene/ \
    --gs_source ./dataset/${src_path}/$scene/gs/point_cloud/iteration_$it/point_cloud.ply \
    --config configs/$3.yaml \
    --height $height \
    --width $width \
    --tag ${scene_tag} \
    --wandb \
    --wandb_group ${wandb_group} \


    