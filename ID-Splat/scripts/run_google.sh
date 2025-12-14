3DGS_PATH='xxx'
scene_name=("Country" "City" "Port")
for scene in ${scene_name[@]}; do
    scene_idx=(scene0 scene1 scene2)
    echo "source_path"
    for idx in ${scene_idx[@]}; do
        CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=1 python gaussian_feature_extractor.py -m $3DGS_PATH/$scene/$idx/ --iteration 30000 --eval --feature_level 1 -r 2 --experiment_name "IDSPLAT"
        CUDA_LAUNCH_BLOCKING=0 CUDA_VISIBLE_DEVICES=1 python feature_map_renderer.py -m $3DGS_PATH/$scene/$idx/ --iteration 30000 --eval --feature_level 1 --resolution 2 --experiment_name "IDSPLAT"
    done
done
