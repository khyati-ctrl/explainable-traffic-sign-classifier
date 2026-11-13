import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms

from PIL import Image
from pathlib import Path
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------
# 2. DEFINE PROJECT PATHS AND SETTINGS
# ---------------------------------------------------------

# Find the main project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Location of the official test images and CSV file.
TEST_DIR = PROJECT_DIR / "data" / "Test"
TEST_CSV = PROJECT_DIR / "data" / "Test.csv"

# Location of our trained model weights.
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.pth"

# Use GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
NUM_CLASSES = 43

print("Using device:", device)


# ---------------------------------------------------------
# 3. DEFINE TEST IMAGE TRANSFORMATIONS
# ---------------------------------------------------------

# These transformations must match the validation
# transformations used during training.
#
# We do not apply random augmentation during evaluation.
# Every test image must be processed consistently.

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ---------------------------------------------------------
# 4. CREATE THE TEST DATASET
# ---------------------------------------------------------

class TrafficSignTestDataset(Dataset):

    def __init__(self, csv_file, test_dir, transform=None):

        # Read image information and actual class labels.
        self.data = pd.read_csv(csv_file)

        self.test_dir = test_dir
        self.transform = transform

    def __len__(self):

        # Return the total number of test images.
        return len(self.data)

    def __getitem__(self, index):

        # Get the information for one image.
        row = self.data.iloc[index]

        # Get the image filename from the CSV.
        image_path = self.test_dir / Path(row["Path"]).name

        # Open the image and convert it to RGB.
        image = Image.open(image_path).convert("RGB")

        # Get the actual traffic sign class.
        label = int(row["ClassId"])

        # Apply resizing and normalization.
        if self.transform:
            image = self.transform(image)

        return image, label


# Create the test dataset.
test_dataset = TrafficSignTestDataset(
    csv_file=TEST_CSV,
    test_dir=TEST_DIR,
    transform=test_transform
)

# Load images in batches.
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Total test images:", len(test_dataset))


# ---------------------------------------------------------
# 5. LOAD THE TRAINED MOBILENETV2 MODEL
# ---------------------------------------------------------

# Recreate the same MobileNetV2 architecture.
# We do not need pretrained weights because we will
# load our already-trained checkpoint.

model = models.mobilenet_v2(weights=None)

# Replace the final layer with 43 output classes.
input_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    input_features,
    NUM_CLASSES
)

# Load the weights learned during training.
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)

# Set the model to evaluation mode.
model.eval()

print("Trained model loaded successfully!")


# ---------------------------------------------------------
# 6. GENERATE PREDICTIONS
# ---------------------------------------------------------

all_labels = []
all_predictions = []

# Disable gradient calculation because we are only
# making predictions, not training the model.

with torch.no_grad():

    for images, labels in test_loader:

        # Move images to the selected device.
        images = images.to(device)

        # Get model predictions (43 class scores).
        outputs = model(images)

        # Select the class with the highest score.
        _, predictions = torch.max(outputs, 1)

        # Store actual and predicted labels.
        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


# ---------------------------------------------------------
# 7. CALCULATE EVALUATION METRICS
# ---------------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ---------------------------------------------------------
# 8. DISPLAY THE RESULTS
# ---------------------------------------------------------

print("\n========== FINAL TEST RESULTS ==========")

print(f"Test Accuracy:       {accuracy * 100:.2f}%")
print(f"Macro Precision:     {precision:.4f}")
print(f"Macro Recall:        {recall:.4f}")
print(f"Macro F1-Score:      {f1:.4f}")
print(f"Weighted F1-Score:   {weighted_f1:.4f}")

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
)


# ---------------------------------------------------------
# 9. GENERATE AND SAVE THE CONFUSION MATRIX
# ---------------------------------------------------------

# A confusion matrix shows which classes the model
# predicts correctly and which classes it confuses.

fig, ax = plt.subplots(figsize=(18, 16))

ConfusionMatrixDisplay.from_predictions(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    ax=ax,
    cmap="Blues",
    colorbar=True
)

plt.title("MobileNetV2 - GTSRB Test Confusion Matrix")
plt.tight_layout()

# Save the confusion matrix in the project directory.
output_path = PROJECT_DIR / "confusion_matrix.png"

plt.savefig(output_path, dpi=300)

print("\nConfusion matrix saved at:", output_path)

plt.show()

print("\nTest evaluation completed!")
