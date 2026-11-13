
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models
from torchvision.models import MobileNet_V2_Weights

from pathlib import Path

# Import the DataLoaders from our existing data pipeline.
# We do NOT need to repeat preprocessing or train-validation splitting.
from data_pipeline import train_loader, val_loader


# -------------------- CONFIGURATION --------------------

# Check whether a GPU is available.
# Training is generally faster on a GPU than on a CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# GTSRB contains 43 different traffic sign classes (0 to 42).
NUM_CLASSES = 43

# Number of times the model will go through the complete training dataset.
EPOCHS = 5

# Learning rate controls how much the model updates its weights.
LEARNING_RATE = 0.001

# Location where the trained model will be saved.
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mobilenetv2_best.pth"


# -------------------- LOAD PRETRAINED MODEL --------------------

# Load MobileNetV2 with weights already learned from ImageNet.
# This allows us to reuse features learned from millions of images.
weights = MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(weights=weights)


# -------------------- FREEZE FEATURE EXTRACTION LAYERS --------------------

# Freeze the feature extraction layers.
# requires_grad = False means PyTorch will NOT update these weights
# during training.
#
# These layers have already learned useful features such as edges,
# shapes, textures and patterns from ImageNet.

for parameter in model.features.parameters():
    parameter.requires_grad = False


# -------------------- MODIFY CLASSIFICATION LAYER --------------------

# MobileNetV2 originally predicts ImageNet categories.
# We need it to predict our 43 traffic sign classes instead.

# Get the number of input features expected by the original final layer.
input_features = model.classifier[1].in_features

# Replace the original final layer with a new layer having 43 outputs.
# Each output represents one traffic sign class.
model.classifier[1] = nn.Linear(input_features, NUM_CLASSES)

# Move the model to the selected device (GPU or CPU).
model = model.to(device)


# -------------------- LOSS FUNCTION AND OPTIMIZER --------------------

# CrossEntropyLoss measures how far the model's predictions are
# from the actual class labels.
criterion = nn.CrossEntropyLoss()

# Adam optimizer updates the trainable classification layer's weights.
# We only pass classifier parameters because the feature layers are frozen.
optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# -------------------- TRAINING AND VALIDATION --------------------

# Store the best validation accuracy achieved so far.
# We will save the model whenever this value improves.
best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING PHASE
    # ========================================================

    # Set the model to training mode.
    model.train()

    # Keep the frozen feature extractor in evaluation mode.
    # This prevents its BatchNorm running statistics from changing.
    model.features.eval()

    running_train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Go through every batch of training images.
    for images, labels in train_loader:

        # Move images and labels to the GPU or CPU.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients left over from the previous batch.
        optimizer.zero_grad()

        # Forward pass: give images to the model and get predictions.
        outputs = model(images)

        # Calculate the difference between predictions and actual labels.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients for trainable parameters.
        loss.backward()

        # Update the classification layer's weights.
        optimizer.step()

        # Add this batch's loss to the total training loss.
        running_train_loss += loss.item() * images.size(0)

        # Find the class with the highest prediction score.
        _, predicted = torch.max(outputs, 1)

        # Count how many predictions match the actual labels.
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Calculate average training loss and training accuracy.
    train_loss = running_train_loss / train_total
    train_accuracy = 100 * train_correct / train_total


    # ========================================================
    # VALIDATION PHASE
    # ========================================================

    # Set the model to evaluation mode.
    # This disables training-specific behaviour such as dropout.
    model.eval()

    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    # We do not calculate gradients during validation because
    # we are only checking the model's performance.
    with torch.no_grad():

        for images, labels in val_loader:

            # Move the validation batch to the selected device.
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions for the validation images.
            outputs = model(images)

            # Calculate validation loss.
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)

            # Get the predicted class for each image.
            _, predicted = torch.max(outputs, 1)

            # Count correct predictions.
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy.
    val_loss = running_val_loss / val_total
    val_accuracy = 100 * val_correct / val_total


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")


    # ========================================================
    # SAVE THE BEST MODEL
    # ========================================================

    # Save the model only if its validation accuracy is better
    # than the best accuracy recorded so far.
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        # Save the model's learned weights.
        torch.save(model.state_dict(), MODEL_PATH)

        print("Best model saved!")


# -------------------- TRAINING COMPLETE --------------------

print("\nTraining completed!")

print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
print("Model saved at:", MODEL_PATH)
