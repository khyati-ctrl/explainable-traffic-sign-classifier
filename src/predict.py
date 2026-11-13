
import torch
import torch.nn as nn

from torchvision import models, transforms
from PIL import Image

from pathlib import Path
import argparse


# -------------------- CONFIGURATION --------------------

# Select GPU if available, otherwise use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We have 43 traffic sign classes in the GTSRB dataset.
NUM_CLASSES = 43

# Get the main project directory.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Location of the model weights saved during training.
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.pth"


# -------------------- IMAGE PREPROCESSING --------------------

# The input image must undergo the same preprocessing
# that we applied to the validation images during training.

transform = transforms.Compose([

    # Resize the image to the same dimensions used during training.
    transforms.Resize((64, 64)),

    # Convert the image into a PyTorch tensor.
    # Pixel values are converted from 0-255 to 0-1.
    transforms.ToTensor(),

    # Normalize using ImageNet mean and standard deviation.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -------------------- LOAD THE TRAINED MODEL --------------------

def load_model():

    # Create the same MobileNetV2 architecture used during training.
    # weights=None prevents downloading pretrained weights again.
    model = models.mobilenet_v2(weights=None)

    # Replace the final classification layer with 43 outputs.
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, NUM_CLASSES)

    # Load the weights learned during our training process.
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )

    # Move the model to the selected device.
    model = model.to(device)

    # Set the model to evaluation mode.
    # This ensures that we use it for predictions, not training.
    model.eval()

    return model


# -------------------- PREDICT TRAFFIC SIGN --------------------

def predict_image(image_path, model):

    # Open the image and convert it to RGB.
    # This ensures that the image has exactly 3 colour channels.
    image = Image.open(image_path).convert("RGB")

    # Apply the preprocessing transformations.
    image = transform(image)

    # Add a batch dimension because the model expects batches.
    # Shape changes from [3, 64, 64] to [1, 3, 64, 64].
    image = image.unsqueeze(0)

    # Move the image to the GPU or CPU.
    image = image.to(device)

    # We only need predictions, not gradients.
    with torch.no_grad():

        # Forward pass: get the model's 43 output scores.
        outputs = model(image)

        # Convert raw scores (logits) into probabilities.
        probabilities = torch.softmax(outputs, dim=1)

        # Find the highest probability and its corresponding class.
        confidence, predicted_class = torch.max(probabilities, dim=1)

    # Convert the results from tensors into ordinary Python values.
    predicted_class = predicted_class.item()
    confidence = confidence.item() * 100

    return predicted_class, confidence


# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":

    # Accept an image path from the command line.
    parser = argparse.ArgumentParser(
        description="Predict a traffic sign using MobileNetV2"
    )

    parser.add_argument(
        "image_path",
        help="Path of the traffic sign image"
    )

    args = parser.parse_args()

    # Check whether the image file exists.
    image_path = Path(args.image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Check whether the trained model exists.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py first."
        )

    # Load the trained model.
    model = load_model()

    # Predict the traffic sign.
    predicted_class, confidence = predict_image(image_path, model)

    # Display the prediction results.
    print("\n---------- PREDICTION RESULT ----------")

    print("Predicted Traffic Sign Class:", predicted_class)
    print(f"Confidence: {confidence:.2f}%")

    print("---------------------------------------")
