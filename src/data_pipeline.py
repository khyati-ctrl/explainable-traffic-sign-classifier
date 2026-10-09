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

        for image_path in class_dir.glob("*.png"): #iterating through each image file inside the current class folder
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

#CHECK CLASS BALANCE
print("\nImages per class:")
for class_id in sorted(class_images):
    print(
        f"Class {class_id}: "
        f"{len(class_images[class_id])} images" #print the number of images in each class
    )


# PREPROCESSING
# Training images:
# resize + augmentation + normalization

train_transform = transforms.Compose([ #compose means aplly these transformations one after another in the order they are listed. the output of one transformation is the input to the next transformation.
    transforms.Resize((64, 64)), #resize all images to 64x64 pixels. this is important because the model expects a fixed input size.

    transforms.RandomRotation(10), #data augmentation: randomly rotate the image by a maximum of 10 degrees (cw or acw). this helps the model generalize better by seeing different orientations of the same image.

    transforms.ColorJitter(  #Randomly changes brightness and contrast.
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(), #convert the image to a PyTorch tensor. this also scales the pixel values from [0, 255] to [0, 1].

    transforms.Normalize( #normalize the image tensor using the mean and standard deviation of the ImageNet dataset. this is important because the model was pre-trained on ImageNet, so we want to use the same normalization.
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Validation images:
# resize + normalization
# NO random augmentation

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# CUSTOM DATASET- creating our own dataset class by inheriting from the PyTorch Dataset class. this allows us to define how to load and preprocess our data.
class TrafficSignDataset(Dataset):

    def __init__(self, data, transform=None):
        self.data = data #list of tuples where each tuple is (image_path, class_id)
        self.transform = transform #transformations to apply to each image

    def __len__(self):
        return len(self.data) #returns the total number of images in the dataset

    def __getitem__(self, index):
        image_path, label = self.data[index] #get the image path and class label for the given index
        image = Image.open(image_path).convert("RGB") #open the image file and convert it to RGB format (in case it's grayscale or has an alpha channel)
        if self.transform:
            image = self.transform(image) #apply the transformations to the image if any are specified
        return image, label


# CREATE DATASETS
train_dataset = TrafficSignDataset(
    train_data,
    train_transform
)

val_dataset = TrafficSignDataset(
    val_data,
    val_transform
)

# CREATE DATALOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=32, #loading the dataset in batches of 32 images. this is important for training efficiency and memory usage.
    shuffle=True #shuffle the data because we want the model to see the data in a different order each time, which helps prevent overfitting and improves generalization.
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

# CHECK A BATCH
images, labels = next(iter(train_loader))

print("\nBatch information:")
print("Image shape:", images.shape)
print("Labels shape:", labels.shape)

# VISUALIZE AUGMENTED IMAGES

# Normalized images need to be converted back before displaying them.

images = images[:8]
mean = torch.tensor(
    [0.485, 0.456, 0.406]
).view(3, 1, 1)

std = torch.tensor(
    [0.229, 0.224, 0.225]
).view(3, 1, 1)

images = images * std + mean # This reverses the normalization approximately
images = images.permute(0, 2, 3, 1)

plt.figure(figsize=(12, 6)) # Creates the figure where our images will be displayed.

for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].clamp(0, 1))
    plt.title(f"Class: {labels[i].item()}")
    plt.axis("off")

plt.tight_layout()
plt.show()