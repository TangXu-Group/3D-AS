
for scene in "Country" "City" "Port" ; do
    query_id=(0 7 14 21 28 35 42 49 56 63)
    for scene_name in "scene0" ; do
        for query_frame in "${query_id[@]}"; do
        CUDA_VISIBLE_DEVICES=1 python demo.py --mp4_path /YOU_PATH/${scene}/${scene_name}/video.mp4  --query_frame $query_frame --conf_thr 0.01 --bkg_opacity 0.7 --rate 1  --save_path /YOU_PATH/${scene}/${scene_name}
        done
    done
done

