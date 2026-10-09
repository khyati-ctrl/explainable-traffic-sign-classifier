from pathlib import Path #Path helps Python work with file and folder paths
from collections import defaultdict #defaultdict is a subclass of the built-in dict class. It overrides one method and adds one writable instance variable. The remaining functionality is the same as for the dict class and is not documented here.
import random #used to randomly shuffle images before splitting them into training and validation sets

import torch #PyTorch is an open source machine learning library based on the Torch library, used for applications such as computer vision and natural language processing
from torch.utils.data import Dataset, DataLoader #DataLoader takes the dataset and gives it to the model in batches, which is more efficient than giving it all at once. Dataset is an abstract class representing a dataset.
from torchvision import transforms #transforms contains ready-made image preprocessing operations.
from PIL import Image #PIL/Pillow allows us to open image files.
import matplotlib.pyplot as plt #for visualization

#PATHS 
DATA_DIR = Path("data") #finding the dataset folder
TRAIN_DIR = DATA_DIR / "Train" #finding the training dataset folder inside data folder

#COLLECT IMAGE PATHS
all_images = [] #empty list to collect all images. each element will be a tuple of (image_path, class label)
for class_dir in sorted(TRAIN_DIR.iterdir()): #iterating through each class folder inside the training dataset folder
    if class_dir.is_dir(): #checking if the current path is a directory
        class_id = int(class_dir.name) #getting the class label from the folder name

        for image_path in class_dir.glob("*.ppm"): #iterating through each image file inside the current class folder
            all_images.append((image_path, class_id)) #adding the image path and class label to the list    

print("Total training images:", len(all_images))

#SPLIT INTO TRAIN/VALIDATION
#the available dataset was already split into training and testing sets. so we are dividing the existing training set into a new training and validation set. we will use 80% of the images for training and 20% for validation.

class_images = defaultdict(list) 

for image_path, class_id in all_images: #each item looks like (image_path, class_id) so Python differentiates between the two
    class_images[class_id].append(image_path) # Group images by class

train_data = []
val_data = []

random.seed(42) # random seed is set to ensure that every time the code is run, the same random order will be generated, which is important for consistency in experiments.

for class_id, images in class_images.items():

    random.shuffle(images)
    split_index = int(0.8 * len(images)) #splitting the images in each class into 80% training and 20% validation

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_data.extend((path, class_id) for path in train_images)
    val_data.extend((path, class_id) for path in val_images)


print("Training images:", len(train_data))
print("Validation images:", len(val_data))


# CHECK CLASS BALANCE

print("\nImages per class:")

for class_id in sorted(class_images):
    print(
        f"Class {class_id}: "
        f"{len(class_images[class_id])} images" #print the number of images in each class
    )

