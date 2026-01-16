dataset_path=YOUR_PATH_TO_DATASET

echo $dataset_path
for scene_name  in Country Port City
do
	for scene_idx in  scene0 scene1 scene2
	do
		CUDA_VISIBLE_DEVICES=0 python preprocess.py --dataset_path $dataset_path/$scene_name/$scene_idx/ --resolution 2 --save_folder /YOUR_PATH_TO_SAM/
	done
done

