# CIFAR-10 Image Generation Using Deep Convolutional GANs (DCGAN)

## Project Overview

This project explores the use of **Deep Convolutional Generative Adversarial Networks (DCGANs)** to generate synthetic images based on the CIFAR-10 dataset.

The project focuses on developing, training, and improving a generative deep learning model to produce more realistic images.

## Objectives

- Develop a DCGAN using TensorFlow and Keras.
- Train the model using the CIFAR-10 dataset.
- Improve the generator and discriminator architectures.
- Evaluate the quality of generated images using Fréchet Inception Distance (FID).
- Compare model performance before and after improvements.

## Technologies Used

**Programming Language:** Python

**Frameworks and Libraries:**
- TensorFlow
- Keras
- NumPy
- Matplotlib

**Development Tools:**
- Jupyter Notebook
- Visual Studio Code

## Dataset

The project uses the **CIFAR-10 dataset**, which contains 60,000 colour images across 10 categories.

Each image has a resolution of 32 × 32 pixels.

## Methodology

### 1. Data Preprocessing
- Load and prepare the CIFAR-10 dataset.
- Normalise image pixel values for model training.

### 2. Model Development
- Develop a generator to produce synthetic images from random noise.
- Develop a discriminator to distinguish real images from generated images.
- Train both networks using adversarial learning.

### 3. Model Improvement
- Refine the DCGAN architecture and training process.
- Evaluate the impact of improvements on generated image quality.

### 4. Model Evaluation
- Generate synthetic images using the trained generator.
- Evaluate model performance using FID.
- Compare the baseline and improved models.

## Results

| Model | FID Score |
|---|---:|
| Baseline DCGAN | 224.50 |
| Improved DCGAN | 204.87 |

The improved DCGAN achieved an approximately **8.7% reduction in FID score**, indicating better generated image quality based on this evaluation metric.

## Project Structure

```text
PartA/
├── PartA.ipynb
├── SourceCodeA.py
├── PartA.html.html
├── work_contri.png
├── .gitignore
└── README.md
```

## Future Improvements

- Experiment with alternative GAN architectures.
- Explore additional hyperparameter tuning techniques.
- Improve training stability and image quality.
- Evaluate generated images using additional metrics.

## Author

Developed as part of an academic Deep Learning project.
