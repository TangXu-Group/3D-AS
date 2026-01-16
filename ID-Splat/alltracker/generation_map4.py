import numpy as np
import os
import cv2
from natsort import natsorted

# Video parameters
fps = 30


# Get all images from all scenes
output_path = "YOU_PATH_TO_DATASET/"
os.makedirs(output_path, exist_ok=True)
for scene in ["City", "Country", "Port"]:
    scene_names = ["scene0", "scene1", "scene2"]
    for scene_name in scene_names:
        all_images = []
        os.makedirs(f"{output_path}/{scene}/{scene_name}", exist_ok=True)
        output_video = f"{output_path}/{scene}/{scene_name}/video.mp4"
        print(output_video)
        img_path = f"/YOU_PATH_TO_DATASET/{scene}/{scene_name}/images/"
        if os.path.exists(img_path):
            img_list = os.listdir(img_path)
            img_list = natsorted(img_list)
            print("this is img_list:{}".format(img_list))
            # Add full paths to the images
            for img_name in img_list:
                full_path = os.path.join(img_path, img_name)
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(full_path)
        print(all_images)
        print(f"Total images found: {len(all_images)}")

        if len(all_images) > 0:
            # Read the first image to get dimensions
            first_img = cv2.imread(all_images[0])
            height, width, layers = first_img.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Write each image to video
            for i, img_path in enumerate(all_images):
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize image if needed to match the first image dimensions
                    if img.shape[:2] != (height, width):
                        img = cv2.resize(img, (width, height))
                    video_writer.write(img)
                
                # Print progress
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(all_images)} images")
            
            # Release the video writer
            video_writer.release()
            print(f"Video saved as: {output_video}")
        else:
            print("No images found!")

        
