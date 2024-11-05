import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "./origin_dataset"

ttv_dataset_dit = "./ttv_dataset/"
train_dir = ttv_dataset_dit + "train"
test_dir = ttv_dataset_dit + "test"
val_dir = ttv_dataset_dit + "val"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_folder in os.listdir(data_dir):
    print(class_folder)
    class_path = os.path.join(data_dir, class_folder)
    if os.path.isdir(class_path):
        images = os.listdir(class_path)
        train_images, test_val_images = train_test_split(images, test_size=0.3, random_state=42)
        test_images, val_images = train_test_split(test_val_images, test_size=0.5, random_state=42)

        os.makedirs(train_dir + f"/{class_folder}/", exist_ok=True)
        os.makedirs(test_dir + f"/{class_folder}/", exist_ok=True)
        os.makedirs(val_dir + f"/{class_folder}/", exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_folder, image))
        for image in test_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(test_dir, class_folder, image))
        for image in val_images:
            shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_folder, image))