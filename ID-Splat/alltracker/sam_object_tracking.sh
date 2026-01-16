echo "start"


scenes=("Country" "City" "Port")
scene_names=("scene0" "scene1" "scene2")
query_frames=(0 7 14 21 28 35 42 49 56 63)
scale_list=(0)
for scene in "${scenes[@]}"; do
  for scene_name in "${scene_names[@]}"; do
    for scale in "${scale_list[@]}"; do
    for query_frame in "${query_frames[@]}"; do
      CUDA_VISIBLE_DEVICES=1

      python objectsam_tracking_all.py \
        --scene "$scene" \
        --scene_name "$scene_name" \
        --query_frame "$query_frame" \
        --scale $scale \
        --pseudo_label

      done
    done
  done
done

echo "All tasks completed."

