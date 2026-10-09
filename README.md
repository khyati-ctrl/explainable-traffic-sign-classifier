# Explainable Traffic Sign Classification Using Transfer Learning

## Problem Statement

Automatically classify traffic sign images into predefined categories using computer vision, while providing prediction confidence and visual explanations.

## Objective

Build a small image-classification application using a pretrained convolutional neural network, evaluate its performance, and visualize relevant image regions using Grad-CAM.

## Proposed Features

* Traffic sign image classification
* Confidence-aware predictions
* Grad-CAM visual explanations
* Optional robustness testing under image distortions

## Proposed Architecture

Image Dataset → Preprocessing → Pretrained Model → Evaluation → Grad-CAM → Web Application

## Planned Technology Stack

* Python
* OpenCV
* PyTorch or TensorFlow
* scikit-learn
* Streamlit or Gradio
* Git and GitHub

## Current Progress

* Project scope proposed
* Initial architecture documented
* Week 1: 
Foundations and Image Inspection
Studied fundamental computer vision concepts and image representation.
Explored image preprocessing and the role of transfer learning.
Studied the importance of explainability in image classification.

* Week 2: Data Preprocessing Pipeline

Implemented the initial data pipeline in src/data_pipeline.py.

1. Image Collection and Labeling
Collected image paths from the numbered class directories.
Associated each image with its corresponding class ID.
Successfully identified all 39,209 training images.

2. Training and Validation Split
Divided the original training dataset into training and validation subsets using an 80:20 ratio.
Used a fixed random seed (42) for reproducibility.
Performed the split separately within each class.

3. Image Preprocessing
Implemented preprocessing using torchvision.transforms.
Resizing: Resized images to 64 × 64 pixels.
Data augmentation: Applied random rotation and brightness/contrast adjustments to training images.
Tensor conversion: Converted images into PyTorch tensors.
Normalization: Normalized RGB channels using ImageNet mean and standard deviation values.
Separate pipelines: Used different transformations for training and validation data to avoid random augmentation during validation.

4. Custom Dataset Implementation
Created a custom PyTorch Dataset class named TrafficSignDataset.
Loads an image from its file path.
Converts the image to RGB format.
Applies the specified transformations.
Returns the processed image and its corresponding class label.

5. DataLoader Implementation
Created training and validation DataLoader objects with a batch size of 32.
Shuffles training examples to vary their order during training.
Keeps validation examples unshuffled for predictable evaluation.
Organizes images and labels into batches for efficient processing.

6. Pipeline Verification
Verified the data pipeline by:
Printing the total number of images discovered.
Checking the training and validation image counts.
Inspecting the image distribution across all 43 classes.
Checking image and label tensor dimensions.
Visualizing eight preprocessed images to verify image loading, appearance, and label association.
Verified Output
Total training images: 39209
Training images: 31367
Validation images: 7842

Batch information:
Image shape: torch.Size([32, 3, 64, 64])
Labels shape: torch.Size([32])

The output confirms that the data pipeline successfully loads the images, creates the training and validation subsets, and produces batches with the expected tensor dimensions.

## Initial Observation: Class Imbalance

The number of images varies across the 43 classes.

For example:

Class 2 contains 2,250 images.
Class 1 contains 2,220 images.
Class 0 contains 210 images.

This indicates class imbalance in the training dataset. It will be considered during model evaluation, particularly when comparing performance across individual classes.

## Next Steps
Configure image preprocessing for the selected pretrained model.
Implement transfer learning using a pretrained CNN, such as MobileNetV2.
Train the classifier and monitor validation performance.
Evaluate the model using accuracy, per-class metrics, and a confusion matrix.
Implement Grad-CAM to visualize image regions influencing predictions.
Add confidence-aware predictions.
Develop a simple web application using Streamlit or Gradio.
Optionally evaluate robustness under image distortions.
Current Limitations
Model training and classification performance have not yet been evaluated.
Grad-CAM explanations have not yet been implemented.
The web application has not yet been developed.
The current training/validation split is random at the image level. Related images from the same recording sequence may appear in both subsets, potentially affecting the reliability of validation results.

## Project Structure
explainable-traffic-sign-classifier/
├── data/

│   ├── Train/

│   └── Test/

├── src/

│   └── data_pipeline.py

├── README.md

├── requirements.txt

└── .gitignore



## Future Scope
Improve classification performance through transfer learning and hyperparameter tuning.
Analyze class-wise performance and common misclassifications.
Generate Grad-CAM visualizations for model interpretability.
Test model behavior under changes in brightness, contrast, and other image conditions.
Deploy the classifier through a simple interactive web interface.
