import numpy as np
from PIL import Image
import os

ttv_dataset_dit = "./ttv_dataset/"


def calculate_mean_std():
    mean = [0.0, 0.0, 0.0]
    std = [0.0, 0.0, 0.0]
    num_images = 0

    for ttv_dir in os.listdir(ttv_dataset_dit):
        print(ttv_dir)
        for class_folder in os.listdir(os.path.join(ttv_dataset_dit, ttv_dir)):
            print(class_folder)
            class_path = os.path.join(os.path.join(ttv_dataset_dit, ttv_dir), class_folder)
            if os.path.isdir(class_path):
                images = os.listdir(class_path)
                num_images += len(images)
                for img_path in images:
                    img = np.array(Image.open(os.path.join(class_path,
                                                           img_path))) / 255.0
                    mean += np.mean(img, axis=(0, 1))
                    std += np.std(img, axis=(0, 1))
    mean /= num_images
    std /= num_images

    return mean, std


mean, std = calculate_mean_std()

print("Mean values:", mean)
print("Std values:", std)
