#!/usr/bin/env python
# coding: utf-8

# In[27]:


from IPython.display import HTML, display

display(HTML("""
<style>

/* ================================
   NOTEBOOK BACKGROUND
   ================================ */

/* Main notebook background */
body,
.jp-NotebookPanel,
.jp-NotebookPanel-notebook,
.jp-Notebook,

/* JupyterLab newer windowed notebook containers */
.jp-WindowedPanel,
.jp-WindowedPanel-outer,
.jp-WindowedPanel-inner {
    background-color: #f3f0f8 !important;
}


/* Space surrounding notebook cells */
.jp-Notebook {
    background-color: #f3f0f8 !important;
}


/* ================================
   NOTEBOOK CELLS
   ================================ */

.jp-Cell {
    background-color: #ffffff !important;
    border-radius: 12px !important;
    margin: 14px 18px !important;
    padding: 10px !important;

    border: 1px solid #ded8e8 !important;

    box-shadow:
        0 3px 10px rgba(65, 50, 90, 0.08) !important;
}


/* ================================
   CODE CELLS
   ================================ */

.jp-CodeCell {
    border-left: 4px solid #8b74b8 !important;
    background-color: #ffffff !important;
}

.jp-CodeCell .cm-editor,
.jp-CodeCell .CodeMirror,
.jp-CodeCell .cm-scroller {
    background-color: #fcfaff !important;
    border-radius: 8px !important;
}


/* ================================
   MARKDOWN CELLS
   ================================ */

.jp-MarkdownCell {
    background-color: #ffffff !important;
}


/* ================================
   HEADINGS
   ================================ */

.jp-RenderedHTMLCommon h1 {
    color: #4c3b70 !important;
    font-weight: 750 !important;
    border-bottom: 3px solid #9076bc !important;
    padding-bottom: 10px;
}

.jp-RenderedHTMLCommon h2 {
    color: #604b86 !important;
    font-weight: 700 !important;
    border-left: 5px solid #9076bc !important;
    padding-left: 12px;
}

.jp-RenderedHTMLCommon h3 {
    color: #745c99 !important;
    font-weight: 650 !important;
}


/* ================================
   PARAGRAPHS
   ================================ */

.jp-RenderedHTMLCommon p {
    line-height: 1.7 !important;
    color: #393447 !important;
}


/* ================================
   TABLES
   ================================ */

.jp-RenderedHTMLCommon table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(65,50,90,0.08);
}

.jp-RenderedHTMLCommon th {
    background-color: #eee8f6 !important;
    color: #604b86 !important;
    padding: 10px !important;
}

.jp-RenderedHTMLCommon td {
    padding: 9px !important;
}


/* ================================
   OUTPUT AREA
   ================================ */

.jp-OutputArea {
    background-color: #fcfaff !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

</style>
"""))


# # <span style="color:#1F618D;">Part A – Image Generation Using GAN and VAE</span>
# 
# ### Background, Data Preparation, Model Development and Evaluation
# 
# This project investigates and compares **Deep Convolutional Generative Adversarial Networks (DCGANs)** and **Variational Autoencoders (VAEs)** for generating synthetic images based on the CIFAR-10 dataset.
# 
# ---
# 
# # <span style="color:#1F618D;">1. BACKGROUND OF THE PROBLEM</span>
# 
# The objective of this project is to develop deep generative models that can learn the visual characteristics of the **CIFAR-10 dataset** and generate new synthetic images resembling the original objects.
# 
# Unlike image classification, where a model predicts a predefined class label, image generation requires the model to learn features such as:
# 
# - Colours
# - Textures
# - Shapes
# - Backgrounds
# - Object structures
# 
# CIFAR-10 contains several visually diverse object categories. Images belonging to the same class may appear in different poses, viewpoints, lighting conditions, sizes and backgrounds.
# 
# Some classes also share similar visual characteristics. For example, cats and dogs may have similar body shapes, while deer and horses may contain similar colours and structures.
# 
# Therefore, the models must generate images that are both:
# 
# 1. **Recognisable**, where meaningful object features can be observed.
# 2. **Diverse**, where the model does not repeatedly generate highly similar images.
# 
# ---
# 
# # <span style="color:#1F618D;">2. BACKGROUND RESEARCH ON CIFAR-10</span>
# 
# ## <span style="color:#7D3C98;">2.1 Dataset Description</span>
# 
# The **CIFAR-10 dataset** contains **60,000 RGB colour images**. Each image has a resolution of **32 × 32 pixels** with three colour channels.
# 
# The dataset contains ten classes:
# 
# | Class Number | Class Name |
# |:---:|---|
# | 0 | Airplane |
# | 1 | Automobile |
# | 2 | Bird |
# | 3 | Cat |
# | 4 | Deer |
# | 5 | Dog |
# | 6 | Frog |
# | 7 | Horse |
# | 8 | Ship |
# | 9 | Truck |
# 
# Each class contains 6,000 images.
# 
# | Dataset Split | Number of Images | Images per Class |
# |---|---:|---:|
# | Training set | 50,000 | 5,000 |
# | Testing set | 10,000 | 1,000 |
# | **Total** | **60,000** | **6,000** |
# 
# The dataset is balanced because every class contains the same number of images. Therefore, no class-balancing method is required.
# 
# Although CIFAR-10 images have a relatively low resolution, the dataset remains challenging for image generation. Small errors in colour, shape or object structure can make a generated image difficult to recognise.
# 
# Only the training images are used to train the generative models. The testing images remain unseen during training and may be used as real-image references during evaluation.
# 
# ---
# 
# # <span style="color:#1F618D;">3. EXPLORATORY DATA ANALYSIS</span>
# 
# Exploratory Data Analysis (EDA) is performed to understand the structure, distribution and visual characteristics of the CIFAR-10 dataset before model training.
# 
# ---
# 
# ## <span style="color:#7D3C98;">3.1 Dataset Structure and Data Types</span>
# 
# The shapes and data types of the training and testing datasets are checked to confirm the number of images, image dimensions, RGB colour channels and data format.
# 
# ---
# 
# ## <span style="color:#7D3C98;">3.2 Class Distribution</span>
# 
# A bar chart is used to show the number of training images in each CIFAR-10 class and to check whether the dataset is balanced.
# 
# ---
# 
# ## <span style="color:#7D3C98;">3.3 Sample Images from Each Class</span>
# 
# A grid of sample images is displayed to provide a visual overview of the ten CIFAR-10 classes and the different types of objects contained in the dataset.
# 
# ---
# 
# ## <span style="color:#7D3C98;">3.4 Variation Within the Same Class</span>
# 
# Multiple images from the same class are displayed to examine differences in colour, background, position, size and viewing angle.
# 
# ---
# 
# ## <span style="color:#7D3C98;">3.5 Pixel Value Analysis</span>
# 
# ### 3.5.1 Pixel Statistics Before Normalisation
# 
# The minimum and maximum pixel values are checked to understand the original numerical range of the image data before preprocessing.
# 
# ### 3.5.2 Pixel Intensity Distribution
# A histogram is used to show how pixel intensity values are distributed across the CIFAR-10 images before normalisation.
# 
# ---
# 
# # <span style="color:#1F618D;">4. FEATURE ENGINEERING AND DATA PREPARATION</span>
# 
# ## <span style="color:#7D3C98;">4.1 Data Cleaning</span>
# 
# CIFAR-10 is already provided as a structured image dataset. Therefore, major data-cleaning operations such as handling missing values, duplicated rows or inconsistent categorical values are not required.
# 
# The structure and pixel ranges are checked before preprocessing to ensure that the images are suitable for model training.
# 
# ---
# 
# ## <span style="color:#7D3C98;">4.2 Feature Engineering</span>
# 
# Traditional feature engineering, such as manually extracting edges, textures, shapes or colour features, is not performed.
# 
# The convolutional layers in the DCGAN and VAE automatically learn relevant image features directly from the CIFAR-10 images.
# 
# This allows the models to learn features at different levels of complexity without requiring manually designed image features.
# 
# ---
# 
# ## <span style="color:#7D3C98;">4.3 Data Normalisation</span>
# 
# Before training, the images are converted to the `float32` data type.
# 
# The pixel values are then normalised from the original range of **0 to 255** to the range **−1 to 1**.
# 
# Normalisation places the image values on a smaller and more consistent scale, which can improve training stability and numerical efficiency.
# 
# The range of −1 to 1 also matches the `tanh` output used by the image-generating networks.
# 
# 
# 
# ---
# 
# # <span style="color:#1F618D;">5. GENERATIVE MODELLING APPROACHES</span>
# 
# Two generative modelling frameworks are investigated:
# 
# 1. **Deep Convolutional Generative Adversarial Network**
# 2. **Variational Autoencoder**
# 
# Both approaches can learn patterns from image data and generate new synthetic images, but they use different learning strategies.
# 
# ---
# 
# ## <span style="color:#7D3C98;">5.1 Deep Convolutional Generative Adversarial Network</span>
# 
# A **Deep Convolutional Generative Adversarial Network**, or **DCGAN**, contains two main neural networks.
# 
# | Component | Function |
# |---|---|
# | **Generator** | Creates synthetic images from random latent vectors. |
# | **Discriminator** | Determines whether an image is real or generated. |
# 
# The Generator attempts to produce increasingly realistic images, while the Discriminator attempts to correctly distinguish generated images from real CIFAR-10 images.
# 
# During training, both networks improve through adversarial competition.
# 
# The Generator receives feedback based on how successfully it can fool the Discriminator. Over time, it learns to produce images that more closely resemble the distribution of the real training images.
# 
# DCGANs use convolutional layers to automatically learn visual characteristics such as:
# 
# - Colours
# - Edges
# - Textures
# - Shapes
# - Object structures
# 
# ---
# 
# ### 5.1.1 Why DCGAN Is Suitable for This Problem
# 
# DCGAN is suitable for CIFAR-10 because the task involves generating new image data rather than predicting predefined labels.
# 
# The Generator can learn how to transform random latent vectors into **32 × 32 RGB images**, while the Discriminator provides feedback about whether those generated images resemble real CIFAR-10 images.
# 
# Convolutional layers are especially suitable for image data because they preserve spatial relationships and learn local visual patterns.
# 
# Through adversarial training, increasingly realistic synthetic images can be produced.
# 
# ---
# 
# ## <span style="color:#7D3C98;">5.2 Variational Autoencoder</span>
# 
# A **Variational Autoencoder**, or **VAE**, contains an Encoder, a sampling process and a Decoder.
# 
# | Component | Function |
# |---|---|
# | **Encoder** | Compresses an input image into a latent representation. |
# | **Sampling process** | Samples a value from the learned latent distribution. |
# | **Decoder** | Reconstructs or generates an image from the latent representation. |
# 
# The Encoder processes an input image and learns the parameters of a probability distribution in the latent space.
# 
# A latent vector is sampled from this distribution before being passed into the Decoder.
# 
# The Decoder then attempts to reconstruct the original image from the sampled latent representation.
# 
# The VAE is trained using two main losses:
# 
# - **Reconstruction loss**
# - **KL-divergence loss**
# 
# The reconstruction loss measures the difference between the original image and its reconstruction.
# 
# The KL-divergence loss helps organise the latent space so that new images can be generated by sampling new latent values.
# 
# ---
# 
# ### 5.2.1 Why VAE Is Suitable for This Problem
# 
# VAE is suitable for CIFAR-10 because it can learn a compressed latent representation of the visual features present in the images.
# 
# Instead of directly memorising training images, the Encoder learns a probability distribution representing important image characteristics.
# 
# New latent vectors can then be sampled and passed into the Decoder to generate new images.
# 
# This organised latent space also provides a different approach to image generation compared with the adversarial training used by DCGAN.
# 
# ---
# 
# ## <span style="color:#7D3C98;">5.3 Comparison Between DCGAN and VAE</span>
# 
# | Comparison Area | DCGAN | VAE |
# |---|---|---|
# | **Main components** | Generator and Discriminator | Encoder and Decoder |
# | **Learning method** | Adversarial competition | Image reconstruction and latent-space regularisation |
# | **Generated images** | Often sharper and more detailed | Often smoother but less detailed |
# | **Training behaviour** | May be unstable | Generally more stable |
# | **Latent space** | Learned through adversarial training | Explicitly organised during training |
# | **Main limitation** | Mode collapse or unstable training | Blurry generated images |
# | **Purpose in this project** | Evaluate adversarial image generation | Provide a probabilistic alternative |
# 
# DCGANs commonly generate sharper images, while VAEs commonly provide more stable training and a more organised latent space.
# 
# However, the final comparison will be based on the actual experimental results rather than assuming that one framework is better.
# 
# ---
# 
# # <span style="color:#1F618D;">6. PROPOSED EXPERIMENTAL APPROACH</span>
# 
# Four models are developed to compare baseline and improved versions of each generative framework.
# 
# | Model | Purpose |
# |---|---|
# | **Baseline DCGAN** | Provides a simple reference DCGAN model. |
# | **Improved DCGAN** | Tests whether architectural and training improvements increase image quality. |
# | **Baseline VAE** | Provides a simple reference VAE model. |
# | **Improved VAE** | Tests whether architectural and training improvements increase image quality. |
# 
# ---
# 
# ## <span style="color:#7D3C98;">6.1 Baseline DCGAN</span>
# 
# The baseline DCGAN consists of a Generator and Discriminator.
# 
# The Generator converts random latent vectors into 32 × 32 RGB images.
# 
# The Discriminator processes both real and generated images and predicts whether each image is real or generated.
# 
# The baseline DCGAN provides a reference point for evaluating the Improved DCGAN.
# 
# ---
# 
# ## <span style="color:#7D3C98;">6.2 Improved DCGAN</span>
# 
# The Improved DCGAN modifies the baseline architecture and training configuration to investigate whether image quality can be improved.
# 
# The main improvements are evaluated against the baseline model to determine whether the additional model capacity and training changes result in more realistic and diverse generated images.
# 
# ---
# 
# ## <span style="color:#7D3C98;">6.3 Baseline VAE</span>
# 
# The Baseline VAE consists of:
# 
# - A convolutional Encoder
# - A latent sampling process
# - A convolutional Decoder
# 
# The Encoder compresses the input images into a latent representation.
# 
# The Decoder then uses the sampled latent representation to reconstruct or generate an image.
# 
# The Baseline VAE provides a reference point for evaluating the Improved VAE.
# 
# ---
# 
# ## <span style="color:#7D3C98;">6.4 Improved VAE</span>
# 
# The Improved VAE uses a deeper Encoder and Decoder architecture than the baseline model.
# 
# The improved model is designed to provide greater network capacity for learning the visual characteristics of CIFAR-10 images.
# 
# The baseline and improved VAE use the same latent dimension so that the comparison mainly reflects differences in architecture and training configuration.
# 
# ---
# 
# # <span style="color:#1F618D;">7. OVERALL COMPARISON PROCESS</span>
# 
# The models are compared in three main stages:
# 
# 1. **Baseline DCGAN vs Improved DCGAN**
# 2. **Baseline VAE vs Improved VAE**
# 3. **Best-performing DCGAN vs Best-performing VAE**
# 
# Where possible, the models use the same:
# 
# - Training dataset
# - Batch size
# - Evaluation sample size
# - Number of generated images
# - Evaluation procedures
# 
# This allows the performance differences between the models to be compared more consistently.
# 
# The comparison first determines whether the proposed improvements increase performance within each framework.
# 
# The strongest DCGAN and VAE models are then compared to determine which generative framework performs better for the CIFAR-10 image-generation task.
# 
# ---
# 
# # <span style="color:#1F618D;">8. EVALUATION APPROACH</span>
# 
# The generated images are evaluated using both **qualitative** and **quantitative** methods.
# 
# | Evaluation Type | Method |
# |---|---|
# | Qualitative | Generated image grids |
# | Qualitative | Manual image assessment |
# | Quantitative | Fréchet Inception Distance |
# | Quantitative | Inception Score |
# | Training analysis | Loss curves |
# 
# Using multiple evaluation methods provides a more complete assessment of model performance.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.1 Visual Evaluation</span>
# 
# Generated image grids are displayed after model training.
# 
# The images are examined based on:
# 
# - Object recognisability
# - Colour quality
# - Object structure
# - Image diversity
# - Visual noise
# - Repeated images
# 
# Generated images from the baseline and improved models are displayed using the same grid size to support a consistent visual comparison.
# 
# Visual evaluation is important because numerical metrics may not fully represent whether the generated images appear meaningful to a human observer.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.2 Manual Image Assessment</span>
# 
# A sample of generated images from each model is manually categorised.
# 
# | Category | Definition |
# |---|---|
# | **Clear** | The object is recognisable and contains plausible colours, shapes and structures. |
# | **Marginal** | Some object-like features are visible, but the image is unclear or distorted. |
# | **Nonsense** | No recognisable object or meaningful structure can be identified. |
# 
# The same categories and number of images are used for all models.
# 
# The results can then be presented as the number or percentage of generated images belonging to each category.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.3 Fréchet Inception Distance</span>
# 
# **Fréchet Inception Distance (FID)** compares the feature distributions of real and generated images.
# 
# Features are extracted using a pretrained Inception network.
# 
# The feature distribution of the generated images is then compared with the feature distribution of real CIFAR-10 images.
# 
# A **lower FID score** indicates that the generated images are more similar to the real images.
# 
# The same number of real and generated images is used when calculating FID for each model so that the comparison remains consistent.
# 
# When a smaller evaluation sample is used because of computational limitations, this should be stated as a limitation because FID scores may vary with sample size.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.4 Inception Score</span>
# 
# The **Inception Score (IS)** evaluates the quality and diversity of generated images.
# 
# A higher Inception Score is generally preferred.
# 
# A strong score suggests that individual generated images can be recognised confidently while the complete set of generated images remains diverse.
# 
# However, Inception Score does not directly compare generated images with real CIFAR-10 images.
# 
# Therefore, it is used together with FID and visual evaluation rather than being treated as the only measure of model performance.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.5 DCGAN Training Behaviour</span>
# 
# For the DCGAN models, the Generator and Discriminator losses are plotted across the training epochs.
# 
# The plots are examined for:
# 
# - Large loss fluctuations
# - One network overpowering the other
# - Sudden deterioration in image quality
# - Repeated images
# - Possible mode collapse
# 
# GAN losses may naturally fluctuate because the Generator and Discriminator continuously compete with each other.
# 
# Therefore, the losses are evaluated together with the generated images rather than being interpreted independently.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.6 VAE Training Behaviour</span>
# 
# For the VAE models, the following losses are recorded:
# 
# - Total loss
# - Reconstruction loss
# - KL-divergence loss
# 
# The reconstruction loss indicates how well the Decoder is able to reproduce images from the latent representation.
# 
# The KL-divergence loss measures how closely the learned latent distribution follows the desired probability distribution.
# 
# The loss results are used to observe whether the model improves image reconstruction while maintaining an organised latent representation.
# 
# ---
# 
# ## <span style="color:#7D3C98;">8.7 Reusable Evaluation Utilities</span>
# 
# To keep the evaluation process consistent across the four models, reusable evaluation functions and classes are created.
# 
# These include:
# 
# - A function for displaying generated images
# - A reusable FID evaluator
# - A reusable Inception Score evaluator
# 
# The same evaluators can then be applied to the Baseline DCGAN, Improved DCGAN, Baseline VAE and Improved VAE.
# 
# This helps ensure that the models are evaluated using the same procedures and sample sizes.
# 
# ---
# 
# # <span style="color:#1F618D;">9. EXPECTED OUTCOME</span>
# 
# The DCGAN is expected to generate sharper images but may experience less stable training.
# 
# The VAE is expected to train more consistently, although its generated images may appear smoother or less detailed.
# 
# | Model Framework | Expected Strength | Possible Limitation |
# |---|---|---|
# | **DCGAN** | Sharper and more realistic images | Unstable training or mode collapse |
# | **VAE** | Stable training and organised latent space | Smoother or blurrier images |
# 
# The final comparison will consider:
# 
# - Generated-image quality
# - Manual image assessment
# - FID score
# - Inception Score
# - Image diversity
# - Training-loss behaviour
# - Training stability
# 
# The final model selection will be based on the combined evaluation results rather than a single metric.
# 
# ---
# 
# # <span style="color:#1F618D;">10. REFERENCES</span>
# 
# 1. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B. and Hochreiter, S. (2017). *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*.
# 
# 2. Kingma, D. P. and Welling, M. (2014). *Auto-Encoding Variational Bayes*.
# 
# 3. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.
# 
# 4. Radford, A., Metz, L. and Chintala, S. (2016). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
# 
# 5. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A. and Chen, X. (2016). *Improved Techniques for Training GANs*.

# ## 1. Libraries

# In[1]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,ops

import numpy as np
import matplotlib.pyplot as plt

# InceptionV3 is a pre-trained model used to extract image features
# for evaluation metrics such as FID and Inception Score
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)

# sqrtm calculates the matrix square root
# It is used when calculating the FID score
from scipy.linalg import sqrtm


# ## 2. Importing the dataset and Splitting the dataset

# In[2]:


# Set random seeds so results are more consistent when the code is run again (Reproducibility)
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# # 3. BACKGROUND RESEARCH
# ## Exploratory Data Analysis (EDA)
# - Exploratory data analysis was carried out before model development to understand the structure, distribution and visual characteristics of the CIFAR-10 dataset.
# ## 3.1 Dataset Structure and Data Types

# In[3]:


print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)

print("Image data type:", x_train.dtype)
print("Label data type:", y_train.dtype)


# ### Evaluation
# - The **training set** contains **50,000 images**, which will be used to train the DCGAN model.
# - The **testing set** contains **10,000 images**, which can be used for model evaluation and comparison if required.
# - Each image has a dimension of **32 × 32 pixels** with **3 colour channels (RGB)**, represented by the shape `(32, 32, 3)`.
# - The `uint8` image format means that each pixel value ranges from **0 to 255**. Before training the VAE, the images are normalised to a smaller range that is more suitable for neural network training.
# - The labels have a shape of `(50000, 1)` and `(10000, 1)`, indicating that each image is assigned a single class label corresponding to one of the **10 CIFAR-10 object classes**.

# ## 3.2 Class Distribution

# In[4]:


class_names = [
    "Airplane","Automobile","Bird","Cat","Deer",
    "Dog","Frog","Horse","Ship","Truck"
]

counts = np.bincount(y_train.flatten())

plt.figure(figsize=(8,5))
plt.bar(class_names, counts)
plt.xticks(rotation=45)
plt.ylabel("Number of Images")
plt.title("Distribution of CIFAR-10 Classes")
plt.show()


# ### Observation
# - CIFAR-10 Dataset is balanced

# ## 3.3 Sample Images from Each Class

# In[10]:


plt.figure(figsize=(12,5))

for i in range(10):
    idx = np.where(y_train==i)[0][0]

    plt.subplot(2,5,i+1)
    plt.imshow(x_train[idx])
    plt.title(class_names[i])
    plt.axis("off")

plt.tight_layout()
plt.show()


# ## Sample Images from Each Class
# 
# The figure below displays one sample image from each of the **10 classes** in the CIFAR-10 dataset.
# 
# The dataset consists of the following object categories:
# 
# - Airplane
# - Automobile
# - Bird
# - Cat
# - Deer
# - Dog
# - Frog
# - Horse
# - Ship
# - Truck
# 
# Each class contains **6,000 images**, with **5,000 images allocated for training** and **1,000 images allocated for testing**. Although the images have a relatively low resolution of **32 × 32 pixels**.

# ## 3.4 Variation Within the Same Class

# In[11]:


# We can change the number from 1-10 to see how each class looks like
selected_class = 4  # Deer
selected_indices = np.where(y_train.flatten() == selected_class)[0][:10]

plt.figure(figsize=(15, 4))

for position, image_index in enumerate(selected_indices):
    plt.subplot(2, 5, position + 1)
    plt.imshow(x_train[image_index])
    plt.axis("off")

plt.suptitle(f"Examples from the '{class_names[selected_class]}' Class")
plt.tight_layout()
plt.show()


# ## 3.5 Pixel Value Analysis

# In[12]:


print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())
print("Mean pixel value:", x_train.mean())
print("Standard deviation:", x_train.std())


# 
# ### The CIFAR-10 images are stored using pixel values ranging from **0 to 255**.
# 
# | Statistic | Value |
# |---|---:|
# | Minimum pixel value | 0 |
# | Maximum pixel value | 255 |
# | Mean pixel value | 120.71 |
# | Standard deviation | 64.15 |
# 
# The results show that the images were originally stored using the standard RGB pixel range of **0 to 255**. The mean pixel value of approximately **120.71** indicates the average brightness of the training images, while the standard deviation of approximately **64.15** shows that there is considerable variation in colour and brightness across the dataset.
# 
# The images will be converted to `float32` and normalised before being used for model training.

# ### 3.5.1 Pixel Statistics Before Normalisation

# In[5]:


plt.figure(figsize=(8, 5))
plt.hist(x_train.flatten(), bins=50)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.title("Distribution of CIFAR-10 Pixel Intensities")
plt.show()


# ### Evaluation of Pixel Intensity Distribution
# 
# The histogram shows that the CIFAR-10 images contain a **wide range of pixel intensities from 0 to 255**. Most pixel values are concentrated around the middle of the range, particularly between approximately **50 and 180**, which is consistent with the mean pixel value of **120.71**.
# 
# The distribution also shows noticeable peaks at certain intensity values, including near **0 and 255**. These represent very dark and very bright pixels that may come from shadows, backgrounds, or bright areas within the images.
# 
# Overall, the wide spread of pixel intensities shows that CIFAR-10 contains considerable variation in colour and brightness. Since the values are originally between 0 and 255, **normalization is required before training the DCGAN and VAE** to provide a more suitable input range and improve training stability.

# # 4. DATA PREPARATION
# 
# ## 4.1 Data Cleaning
# 
# No major data-cleaning operations were required because CIFAR-10 is provided as a structured image dataset by tensorflow

# ## 4.2 Data Normalisation
# 
# The images were converted to the float32 data type and the pixel values were normalised before training.
# 
# The pixel values were transformed from the original range of 0 to 255 to the range -1 to 1.
# 
# This range matches the tanh output activation used by the image-generating networks.

# In[6]:


# Convert images to float32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalise pixel values from [0, 255] to [-1, 1]
x_train = (x_train - 127.5) / 127.5
x_test = (x_test - 127.5) / 127.5

print("Training shape:", x_train.shape)
print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())


# ### Evaluation of Data Normalisation
# 
# After preprocessing, the CIFAR-10 training dataset contains **50,000 images**, with each image having a size of **32 × 32 pixels and 3 RGB colour channels**.
# 
# The pixel values have been successfully normalised from the original range of **0 to 255** to a new range of **-1 to 1**. This range is suitable for the DCGAN because the generator uses a **tanh activation function**, which also produces output values between -1 and 1.
# 
# Normalising the images helps keep the input values on a smaller and consistent scale, which can improve the stability of the DCGAN training process.

# ## 4.3 Feature Engineering
# 
# Traditional feature engineering was not required.
# 
# Instead of manually extracting features such as edges, textures or shapes, convolutional layers were used to automatically learn important visual features directly from the CIFAR-10 images.

# # 5. EVALUATION FUNCTIONS
# 
# The following reusable functions, evaluator classes and supporting methods are defined before model evaluation. They are used consistently across the DCGAN and VAE experiments to support a fair comparison between models.
# 
# The evaluation setup includes:
# 
# - **Generated Image Visualisation** – displays generated images in a grid for visual inspection.
# - **CIFAR-10 Class Prediction** – predicts the class that each generated image most closely resembles and displays its confidence score.
# - **Fréchet Inception Distance (FID)** – measures how similar the generated-image distribution is to real CIFAR-10 images.
# - **Inception Score (IS)** – evaluates the quality and diversity of generated images.
# - **Real vs Model Image Comparison** – compares real images with GAN-generated images or VAE reconstructions.
# 

# ### Why DCGAN Was Used Instead of CGAN
# 
# A **DCGAN was selected instead of a Conditional GAN (CGAN)** because the main objective of this project is to evaluate how well a generative model can learn the overall visual distribution of CIFAR-10 and generate realistic and diverse images. The project does not require the user to specify which CIFAR-10 class should be generated.
# 
# Using a DCGAN also keeps the generation process simpler because only a random latent vector is provided to the Generator. A CGAN would require additional class-label inputs to be incorporated into both the Generator and Discriminator, increasing the complexity of the model and training process.
# 
# The limitation of using an unconditional DCGAN is that the class of each generated image is unknown. Therefore, a separate CIFAR-10 classifier is used **after generation** to estimate which class an image most closely resembles. This classifier is used only to support visual evaluation and does not influence the Generator.
# 
# A CGAN would be more suitable if the objective were to generate a specific class on demand, such as requesting only airplanes or ships. Since controlled class generation is not required in this project, the DCGAN provides a simpler and more suitable approach for evaluating general CIFAR-10 image generation.

# ## 5.1 Generated Image Visualisation and Class Prediction
# 
# To provide additional information about what each generated image resembles, a separate **CIFAR-10 classifier** is used after image generation. The classifier predicts one of the ten CIFAR-10 classes and displays the predicted class together with its confidence score above each image.
# 
# For example:
# 
# `Ship — 82.4%`
# 
# indicates that the classifier estimates that the generated image most closely resembles a ship with 82.4% confidence.
# 
# The ten possible predicted classes are:
# 
# - Airplane
# - Automobile
# - Bird
# - Cat
# - Deer
# - Dog
# - Frog
# - Horse
# - Ship
# - Truck
# 
# These predicted labels are used only for **post-generation evaluation and visualisation**. They do not make the model a Conditional GAN (CGAN), because class labels are not provided to the Generator during the image-generation process.
# 
# The generated images are also visually assessed using the **Clear / Marginal / Nonsense eye-power method** to evaluate how recognisable and coherent the generated outputs are.

# ## 5.1 CIFAR-10 Classifier for Generated Images
# 
# A separate CNN classifier was created to estimate which CIFAR-10 class each generated image most closely resembles.
# 
# This classifier is **not part of the DCGAN** and does not affect image generation. It is only used after an image has been generated to provide a predicted class and confidence score.
# 
# ### Classifier Architecture
# 
# | Layer | Output Shape | Purpose |
# |---|---|---|
# | **Input** | `32 × 32 × 3` | Receives a CIFAR-10 RGB image. |
# | **Conv2D (32 filters)** | `32 × 32 × 32` | Learns basic features such as edges and colours. |
# | **MaxPooling2D** | `16 × 16 × 32` | Reduces spatial dimensions while retaining important features. |
# | **Conv2D (64 filters)** | `16 × 16 × 64` | Learns more complex textures and patterns. |
# | **MaxPooling2D** | `8 × 8 × 64` | Further reduces the feature-map size. |
# | **Conv2D (128 filters)** | `8 × 8 × 128` | Learns deeper object and structural features. |
# | **MaxPooling2D** | `4 × 4 × 128` | Compresses the extracted features. |
# | **Flatten** | `2048` | Converts the feature maps into a 1D vector. |
# | **Dense** | `128` | Combines the extracted features for classification. |
# | **Output Dense** | `10` | Uses Softmax to predict one of the 10 CIFAR-10 classes. |
# 
# **Total parameters:** `356,810`
# 
# 

# In[7]:


# CIFAR-10 classifier used to label generated images

classifier = keras.Sequential([
    keras.Input(shape=(32, 32, 3)),

    layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128,
        kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        10,
        activation="softmax"
    )
])

classifier.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

classifier.summary()


# In[8]:


classifier_history = classifier.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)


# In[9]:


classifier_loss, classifier_accuracy = classifier.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(
    f"Classifier Test Accuracy: "
    f"{classifier_accuracy * 100:.2f}%"
)


# In[20]:


def display_generated_images(
    generator,
    latent_dim,
    classifier=None,
    num_images=25,
    title="Generated Images"
):
    # Generate random latent vectors
    random_noise = tf.random.normal(
        shape=(num_images, latent_dim)
    )

    # Generate images
    generated_images = generator(
        random_noise,
        training=False
    ).numpy()

    # -------------------------------------------------
    # Predict CIFAR-10 classes
    # -------------------------------------------------

    if classifier is not None:

        predictions = classifier.predict(
            generated_images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        confidence_scores = np.max(
            predictions,
            axis=1
        )

    # -------------------------------------------------
    # Convert images from [-1, 1] to [0, 1]
    # -------------------------------------------------

    display_images = (
        generated_images + 1.0
    ) / 2.0

    display_images = np.clip(
        display_images,
        0.0,
        1.0
    )

    # -------------------------------------------------
    # Display images
    # -------------------------------------------------

    plt.figure(figsize=(10, 11))

    for i in range(num_images):

        plt.subplot(5, 5, i + 1)

        plt.imshow(
            display_images[i]
        )

        # Add predicted class and confidence
        if classifier is not None:

            predicted_label = class_names[
                predicted_classes[i]
            ]

            confidence = confidence_scores[i]

            plt.title(
                f"{predicted_label}\n"
                f"{confidence * 100:.1f}%",
                fontsize=9
            )

        plt.axis("off")

    plt.suptitle(
        title,
        fontsize=16
    )

    plt.tight_layout()
    plt.show()


# ## <span style="color:#1F618D;">5.2 Fréchet Inception Distance Evaluation</span>
# 
# The **Fréchet Inception Distance (FID)** was used to evaluate how closely the generated images resembled real CIFAR-10 images.
# 
# FID compares the feature distributions of real and generated images. A pretrained InceptionV3 model was used to extract higher-level image features such as shapes, colours, textures and object structures.
# 
# A **lower FID score** indicates that the generated images are more similar to the real CIFAR-10 image distribution.
# 
# ---
# 
# ### <span style="color:#7D3C98;">Evaluation Setup</span>
# 
# A reusable FID evaluation process was created so that all models could be assessed consistently.
# 
# For each model:
# 
# - The same number of real and generated images was used.
# - The same real-image samples were selected.
# - The same random latent vectors were used to generate images.
# - The same preprocessing and feature-extraction steps were applied.
# 
# Using the same evaluation conditions ensures that differences in FID scores are mainly caused by model performance rather than inconsistent test data.
# 
# ---
# 
# ### <span style="color:#7D3C98;">Image Preparation</span>
# 
# The real and generated images were first converted to the same pixel range.
# 
# Since CIFAR-10 images have a resolution of **32 × 32 pixels**, they were resized to **299 × 299 pixels** before being passed into InceptionV3.
# 
# The images were then processed using the required InceptionV3 preprocessing method.
# 
# ---
# 
# ### <span style="color:#7D3C98;">Feature Extraction</span>
# 
# InceptionV3 was used without its final classification layer.
# 
# Instead of predicting class labels, the model produced feature representations for each image.
# 
# These features describe important visual information, including:
# 
# - Object shapes
# - Colour patterns
# - Textures
# - Image structures
# - Visual diversity
# 
# The real-image features were extracted once and reused when evaluating different generators.
# 
# ---
# 
# ### <span style="color:#7D3C98;">FID Calculation</span>
# 
# The mean and covariance of the extracted features were calculated separately for the real and generated images.
# 
# FID then measured the distance between these two feature distributions.
# 
# If the generated images contain similar visual patterns and diversity to the real CIFAR-10 images, the distance between the distributions will be smaller.
# 
# ---
# 
# ### <span style="color:#7D3C98;">FID Interpretation</span>
# 
# | FID Result | Interpretation |
# |---|---|
# | **Lower FID** | Generated images are closer to the real CIFAR-10 distribution. |
# | **Higher FID** | Generated images differ more from the real CIFAR-10 distribution. |
# 
# The model with the lower FID score is generally considered to produce more realistic and diverse images.
# 
# However, FID was interpreted together with generated-image grids and manual image assessment because a single metric may not fully represent image quality.

# In[10]:


class FIDEvaluator:
    def __init__(
        self,
        real_images,
        num_images=1000,
        batch_size=32,
        real_image_range="-1_to_1",
        seed=42
    ):
        self.num_images = num_images
        self.batch_size = batch_size
        self.seed = seed

        if num_images > len(real_images):
            raise ValueError(
                "num_images cannot be larger than the number "
                "of available real images."
            )

        # Select the same real images for every model
        random_generator = np.random.default_rng(seed)

        self.real_indices = random_generator.choice(
            len(real_images),
            size=num_images,
            replace=False
        )

        selected_real_images = real_images[self.real_indices]

        self.real_images = self._convert_to_zero_one(
            selected_real_images,
            real_image_range
        )

        # Load InceptionV3 feature extractor
        self.feature_model = InceptionV3(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(299, 299, 3)
        )

        # Extract real-image features once
        self.real_features = self._extract_features(
            self.real_images
        )

    def _convert_to_zero_one(self, images, image_range):
        images = np.asarray(images, dtype=np.float32)

        if image_range == "-1_to_1":
            images = (images + 1.0) / 2.0

        elif image_range == "0_to_1":
            pass

        elif image_range == "0_to_255":
            images = images / 255.0

        else:
            raise ValueError(
                "image_range must be '-1_to_1', "
                "'0_to_1', or '0_to_255'."
            )

        return np.clip(images, 0.0, 1.0)

    def _extract_features(self, images):
        extracted_features = []

        # Manual batching avoids TensorFlow
        # 'End of sequence' messages
        for start_index in range(
            0,
            len(images),
            self.batch_size
        ):
            end_index = start_index + self.batch_size

            image_batch = images[
                start_index:end_index
            ]

            image_batch = tf.convert_to_tensor(
                image_batch,
                dtype=tf.float32
            )

            # Resize CIFAR-10 images for InceptionV3
            image_batch = tf.image.resize(
                image_batch,
                size=(299, 299)
            )

            # Convert [0, 1] to [0, 255]
            image_batch = image_batch * 255.0

            # Apply InceptionV3 preprocessing
            image_batch = preprocess_input(image_batch)

            batch_features = self.feature_model(
                image_batch,
                training=False
            )

            extracted_features.append(
                batch_features.numpy()
            )

        return np.concatenate(
            extracted_features,
            axis=0
        )

    def _calculate_score(
        self,
        real_features,
        generated_features
    ):
        real_mean = np.mean(
            real_features,
            axis=0
        )

        generated_mean = np.mean(
            generated_features,
            axis=0
        )

        real_covariance = np.cov(
            real_features,
            rowvar=False
        )

        generated_covariance = np.cov(
            generated_features,
            rowvar=False
        )

        mean_difference = (
            real_mean - generated_mean
        )

        covariance_mean = sqrtm(
            real_covariance @ generated_covariance
        )

        # Handle numerical instability
        if not np.isfinite(covariance_mean).all():
            epsilon = 1e-6

            covariance_offset = (
                np.eye(real_covariance.shape[0])
                * epsilon
            )

            covariance_mean = sqrtm(
                (real_covariance + covariance_offset)
                @
                (
                    generated_covariance
                    + covariance_offset
                )
            )

        if np.iscomplexobj(covariance_mean):
            covariance_mean = covariance_mean.real

        fid_score = (
            mean_difference.dot(mean_difference)
            +
            np.trace(
                real_covariance
                + generated_covariance
                - (2.0 * covariance_mean)
            )
        )

        return float(fid_score)

    def evaluate(
        self,
        generator,
        latent_dim,
        model_name="Model",
        generated_image_range="-1_to_1",
        noise_seed=(42, 2)
    ):
        # Use the same noise for every model
        evaluation_noise = tf.random.stateless_normal(
            shape=(self.num_images, latent_dim),
            seed=list(noise_seed)
        )

        generated_images = generator(
            evaluation_noise,
            training=False
        ).numpy()

        generated_images = self._convert_to_zero_one(
            generated_images,
            generated_image_range
        )

        generated_features = self._extract_features(
            generated_images
        )

        fid_score = self._calculate_score(
            real_features=self.real_features,
            generated_features=generated_features
        )

        print("Real images shape:", self.real_images.shape)
        print(
            "Generated images shape:",
            generated_images.shape
        )
        print(f"{model_name} FID: {fid_score:.2f}")

        return fid_score


# ### 5.2.1 Initialise FID Evaluator

# In[11]:


fid_evaluator = FIDEvaluator(
    real_images=x_test,
    num_images=1000,
    batch_size=32,
    real_image_range="-1_to_1",
    seed=42
)


# ## <span style="color:#1F618D;">5.3 Inception Score Evaluation</span>
# 
# The **Inception Score (IS)** was used to evaluate the quality and diversity of the generated images.
# 
# A pretrained InceptionV3 model was used to predict the possible object class for each generated image. The score considers two main aspects:
# 
# - **Image quality** — whether the model confidently recognises the generated image.
# - **Image diversity** — whether the generated images represent a variety of visual classes instead of producing similar outputs repeatedly.
# 
# A **higher Inception Score** generally indicates that the generated images are more recognisable and diverse.
# 
# ---
# 
# ### <span style="color:#7D3C98;">Evaluation Process</span>
# 
# For each generative model:
# 
# 1. A fixed set of random latent vectors was used to generate images.
# 2. The generated images were converted to the required pixel range.
# 3. The images were resized to match the InceptionV3 input size.
# 4. InceptionV3 produced class-probability predictions for each image.
# 5. The predictions were divided into several groups to calculate the average score and standard deviation.
# 
# Using the same number of images and the same latent vectors supports a consistent comparison between the baseline and improved models.
# 
# ---
# 
# ### <span style="color:#7D3C98;">Inception Score Interpretation</span>
# 
# | Inception Score | Interpretation |
# |---|---|
# | **Higher score** | Images are generally more recognisable and diverse. |
# | **Lower score** | Images may be unclear, repetitive or difficult to classify. |
# | **Lower standard deviation** | The score is more consistent across the evaluation groups. |
# | **Higher standard deviation** | Image quality or diversity may vary across the generated samples. |
# 
# The Inception Score is reported using the following format:
# 
# **Mean Inception Score ± Standard Deviation**
# 
# ---
# 
# ### <span style="color:#7D3C98;">Evaluation Limitation</span>
# 
# Inception Score does not directly compare generated images with real CIFAR-10 images. InceptionV3 was also trained using ImageNet classes, which differ from the CIFAR-10 classes.
# 
# Therefore, the score is used as a supporting measurement together with:
# 
# - Fréchet Inception Distance
# - Generated-image grids
# - Manual image assessment
# - Training-loss behaviour

# In[12]:


class InceptionScoreEvaluator:
    def __init__(
        self,
        num_images=1000,
        batch_size=32,
        num_splits=10
    ):
        self.num_images = num_images
        self.batch_size = batch_size
        self.num_splits = num_splits

        if num_images < num_splits:
            raise ValueError(
                "num_images must be greater than or equal "
                "to num_splits."
            )

        # Keep the ImageNet classification layer
        self.inception_model = InceptionV3(
            include_top=True,
            weights="imagenet"
        )

    def _convert_to_zero_one(
        self,
        images,
        image_range
    ):
        images = np.asarray(
            images,
            dtype=np.float32
        )

        if image_range == "-1_to_1":
            images = (images + 1.0) / 2.0

        elif image_range == "0_to_1":
            pass

        elif image_range == "0_to_255":
            images = images / 255.0

        else:
            raise ValueError(
                "image_range must be '-1_to_1', "
                "'0_to_1', or '0_to_255'."
            )

        return np.clip(
            images,
            0.0,
            1.0
        )

    def _get_predictions(self, images):
        all_predictions = []

        for start_index in range(
            0,
            len(images),
            self.batch_size
        ):
            end_index = (
                start_index + self.batch_size
            )

            image_batch = images[
                start_index:end_index
            ]

            image_batch = tf.convert_to_tensor(
                image_batch,
                dtype=tf.float32
            )

            # Resize CIFAR-10 images for InceptionV3
            image_batch = tf.image.resize(
                image_batch,
                size=(299, 299)
            )

            # Convert [0, 1] to [0, 255]
            image_batch = image_batch * 255.0

            # Apply InceptionV3 preprocessing
            image_batch = preprocess_input(
                image_batch
            )

            predictions = self.inception_model(
                image_batch,
                training=False
            )

            all_predictions.append(
                predictions.numpy()
            )

        return np.concatenate(
            all_predictions,
            axis=0
        )

    def _calculate_score(self, predictions):
        epsilon = 1e-10
        split_scores = []

        # Divide predictions into separate splits
        prediction_splits = np.array_split(
            predictions,
            self.num_splits
        )

        for split_predictions in prediction_splits:
            # Marginal class distribution
            marginal_distribution = np.mean(
                split_predictions,
                axis=0,
                keepdims=True
            )

            # KL divergence for every image
            kl_divergence = (
                split_predictions
                *
                (
                    np.log(
                        split_predictions + epsilon
                    )
                    -
                    np.log(
                        marginal_distribution + epsilon
                    )
                )
            )

            kl_divergence = np.sum(
                kl_divergence,
                axis=1
            )

            split_score = np.exp(
                np.mean(kl_divergence)
            )

            split_scores.append(split_score)

        mean_score = float(
            np.mean(split_scores)
        )

        standard_deviation = float(
            np.std(split_scores)
        )

        return mean_score, standard_deviation

    def evaluate(
        self,
        generator,
        latent_dim,
        model_name="Model",
        generated_image_range="-1_to_1",
        noise_seed=(42, 2)
    ):
        # Use the same noise for every model
        evaluation_noise = (
            tf.random.stateless_normal(
                shape=(
                    self.num_images,
                    latent_dim
                ),
                seed=list(noise_seed)
            )
        )

        generated_images = generator(
            evaluation_noise,
            training=False
        ).numpy()

        generated_images = (
            self._convert_to_zero_one(
                generated_images,
                generated_image_range
            )
        )

        predictions = self._get_predictions(
            generated_images
        )

        mean_score, standard_deviation = (
            self._calculate_score(
                predictions
            )
        )

        print(
            "Generated images shape:",
            generated_images.shape
        )

        print(
            f"{model_name} Inception Score: "
            f"{mean_score:.2f} ± "
            f"{standard_deviation:.2f}"
        )

        return mean_score, standard_deviation


# ### 5.1.1 Initialise Inception Score Evaluator

# In[13]:


is_evaluator = InceptionScoreEvaluator(
    num_images=1000,
    batch_size=32,
    num_splits=10
)


# # 6. First Model: Baseline DCGAN

# ## Baseline DCGAN Discriminator
# 
# The Discriminator is responsible for determining whether an input image is **real**, meaning it comes from the CIFAR-10 dataset, or **generated**, meaning it was produced by the Generator.
# 
# | Component / Parameter | Setting Used | Purpose / Justification |
# |---|---|---|
# | **Input Shape** | `32 × 32 × 3` | Matches the dimensions of CIFAR-10 RGB images. |
# | **First Conv2D Layer** | 64 filters, `4 × 4` kernel, stride `2`, `same` padding | Learns basic visual features such as edges, colours and textures while reducing the image size from **32 × 32 to 16 × 16**. |
# | **LeakyReLU** | Negative slope = `0.2` | Allows a small gradient for negative values, helping maintain gradient flow during GAN training. |
# | **Dropout** | `0.3` | Reduces overfitting and helps prevent the Discriminator from becoming too strong compared with the Generator. |
# | **Second Conv2D Layer** | 128 filters, `4 × 4` kernel, stride `2`, `same` padding | Learns more complex visual patterns and object structures while reducing the feature maps from **16 × 16 to 8 × 8**. |
# | **Increase from 64 to 128 filters** | 64 → 128 | Deeper layers use more filters so the network can learn more detailed and complex image features. |
# | **Flatten Layer** | Converts feature maps into a 1D vector | Prepares the extracted image features for the final classification layer. |
# | **Output Layer** | 1 neuron with `sigmoid` activation | Produces a value between 0 and 1 representing the probability that the image is real. |
# 
# ### Discriminator Flow
# 
# | Stage | Output Shape |
# |---|---|
# | Input Image | `32 × 32 × 3` |
# | First Conv2D | `16 × 16 × 64` |
# | Second Conv2D | `8 × 8 × 128` |
# | Flatten | 1D feature vector |
# | Sigmoid Output | Real / Fake probability |
# 
# The Discriminator supports the image-generation process by providing feedback to the Generator. When generated images look unrealistic, the Discriminator is more likely to classify them as fake. This feedback encourages the Generator to gradually produce images that more closely resemble real CIFAR-10 images.

# In[15]:


baseline_discriminator = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),

        # 32 × 32 → 16 × 16
        layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        # 16 × 16 → 8 × 8
        layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ],
    name="baseline_discriminator"
)

baseline_discriminator.summary()


# ## Baseline DCGAN Generator
# 
# The Generator is responsible for converting a random latent vector into a synthetic **32 × 32 RGB image** that attempts to resemble images from the CIFAR-10 dataset.
# 
# | Component / Parameter | Setting Used | Purpose / Justification |
# |---|---|---|
# | **Latent Dimension** | `100` | Provides the Generator with a compact random representation from which different image features can be learned. A latent size of 100 provides enough capacity for variation while keeping the model manageable. |
# | **Input Layer** | `(100,)` | Accepts a random latent vector that acts as the starting point for generating a new image. |
# | **Dense Layer** | `8 × 8 × 128` units | Expands the 100-dimensional latent vector into enough values to form an initial set of spatial feature maps. |
# | **LeakyReLU** | Negative slope = `0.2` | Allows a small gradient for negative values, helping maintain gradient flow during GAN training. |
# | **Reshape Layer** | `8 × 8 × 128` | Converts the Dense output into spatial feature maps that can be progressively upsampled into an image. |
# | **First Conv2DTranspose** | 128 filters, `4 × 4` kernel, stride `2`, `same` padding | Upsamples the feature maps from **8 × 8 to 16 × 16** while learning visual patterns required to construct the image. |
# | **Second Conv2DTranspose** | 64 filters, `4 × 4` kernel, stride `2`, `same` padding | Further upsamples the feature maps from **16 × 16 to 32 × 32**, matching the required CIFAR-10 image resolution. |
# | **Decrease from 128 to 64 filters** | 128 → 64 | As the spatial dimensions increase, fewer feature maps are required while the Generator gradually converts learned features into the final image. |
# | **Output Conv2D Layer** | 3 filters, `3 × 3` kernel | Produces the final three channels required for an RGB image: red, green and blue. |
# | **Output Activation** | `tanh` | Produces pixel values between **−1 and 1**, matching the normalised range of the CIFAR-10 training images. |
# 
# ### Generator Flow
# 
# | Stage | Output Shape |
# |---|---|
# | Random Latent Vector | `100` |
# | Dense + Reshape | `8 × 8 × 128` |
# | First Conv2DTranspose | `16 × 16 × 128` |
# | Second Conv2DTranspose | `32 × 32 × 64` |
# | Final Conv2D | `32 × 32 × 3` |
# 
# The Generator gradually transforms random noise into an image by first creating low-resolution feature maps and then increasing their spatial dimensions. The transposed convolutional layers allow the network to learn how visual features should be arranged as the image becomes larger.
# 
# The final output has a shape of **32 × 32 × 3**, which matches the CIFAR-10 images. During adversarial training, feedback from the Discriminator encourages the Generator to improve these synthetic images so that they increasingly resemble real CIFAR-10 images.

# In[16]:


latent_dim = 100
baseline_generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),

        # Convert latent vector into 8 × 8 feature maps
        layers.Dense(8 * 8 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((8, 8, 128)),

        # 8 × 8 → 16 × 16
        layers.Conv2DTranspose(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),

        # 16 × 16 → 32 × 32
        layers.Conv2DTranspose(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),

        # Produce 32 × 32 RGB image
        layers.Conv2D(
            3,
            kernel_size=3,
            padding="same",
            activation="tanh"
        )
    ],
    name="baseline_generator"
)

baseline_generator.summary()


# ## Custom GAN Training Process
# 
# The custom `GAN` class combines the Generator and Discriminator and defines how both networks are trained during each training step.
# 
# | Component / Step | What the Code Does | Purpose / Justification |
# |---|---|---|
# | **Generator and Discriminator** | Stores both networks inside the GAN model | Allows both networks to be trained together using the adversarial learning process. |
# | **Latent Dimension** | Uses `latent_dim = 100` | Determines the size of the random noise vector passed to the Generator. |
# | **Loss Trackers** | Tracks `d_loss` and `g_loss` | Allows the performance of the Discriminator and Generator to be monitored during training. |
# | **Separate Optimizers** | Uses one optimizer for the Discriminator and another for the Generator | The two networks have different parameters and therefore need their own gradient updates. |
# | **Loss Function** | Uses the same binary classification loss for both networks | Suitable because the Discriminator performs a real-versus-generated classification task. |
# | **Batch Size** | Obtained from the current batch using `tf.shape(real_images)[0]` | Allows the training step to work with different batch sizes if required. |
# 
# ### Training the Discriminator
# 
# | Step | What Happens | Purpose |
# |---|---|---|
# | **1. Generate random latent vectors** | Random values are sampled using `tf.random.normal()` | Provides random inputs for the Generator. |
# | **2. Generate synthetic images** | The Generator converts the latent vectors into images | Creates fake examples for training the Discriminator. |
# | **3. Combine real and generated images** | Real CIFAR-10 images and generated images are joined together | Allows the Discriminator to learn from both real and fake examples in the same training step. |
# | **4. Create labels** | Real images are labelled `1` and generated images are labelled `0` | Provides the correct targets for binary classification. |
# | **5. Calculate Discriminator loss** | The Discriminator predicts whether each image is real or generated | Measures how well the Discriminator distinguishes real images from generated images. |
# | **6. Update Discriminator weights** | Gradients are calculated using `GradientTape` and applied using the Discriminator optimizer | Improves the Discriminator's ability to classify real and generated images correctly. |
# 
# ### Training the Generator
# 
# | Step | What Happens | Purpose |
# |---|---|---|
# | **1. Generate new latent vectors** | A new batch of random noise is created | Provides new inputs for the Generator. |
# | **2. Create misleading labels** | All generated images are given target label `1` | The Generator is trained to make the Discriminator believe its generated images are real. |
# | **3. Generate synthetic images** | The Generator produces another batch of images | These images are used to test whether the Generator can fool the Discriminator. |
# | **4. Pass images through Discriminator** | The generated images are classified by the Discriminator | Provides feedback about how realistic the generated images appear. |
# | **5. Calculate Generator loss** | Generated images are compared against the target label `1` | A lower Generator loss indicates that the Generator is becoming better at fooling the Discriminator. |
# | **6. Update Generator weights** | Gradients are calculated and applied only to the Generator | Improves the Generator so that future images become more realistic. |
# 
# ### How Adversarial Training Solves the Problem
# 
# The Generator and Discriminator improve through competition.
# 
# The **Discriminator** learns to correctly classify:
# 
# - Real CIFAR-10 images as `1`
# - Generated images as `0`
# 
# At the same time, the **Generator** attempts to produce images that the Discriminator will classify as `1`.
# 
# The training process can therefore be represented as:
# 
# **Random Noise → Generator → Generated Image → Discriminator → Real/Fake Prediction**
# 
# During Generator training, the target is deliberately set to **real (`1`)**. This means that the Generator receives a stronger penalty when its generated images are easily recognised as fake.
# 
# As training continues, this feedback encourages the Generator to learn visual patterns that more closely resemble the real CIFAR-10 image distribution.
# 
# Therefore, the adversarial training process allows the DCGAN to gradually generate more realistic synthetic images without requiring predefined class labels.

# In[17]:


class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [
            self.d_loss_tracker,
            self.g_loss_tracker
        ]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()

        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        # Handle data passed as a tuple
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        batch_size = tf.shape(real_images)[0]

        # -------------------------------------------------
        # Train Discriminator
        # -------------------------------------------------

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        generated_images = self.generator(
            random_latent_vectors,
            training=True
        )

        combined_images = tf.concat(
            [real_images, generated_images],
            axis=0
        )

        # Real images = 1, generated images = 0
        labels = tf.concat(
            [
                tf.ones((batch_size, 1)),
                tf.zeros((batch_size, 1))
            ],
            axis=0
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator(
                combined_images,
                training=True
            )

            d_loss = self.loss_fn(
                labels,
                predictions
            )

        d_gradients = tape.gradient(
            d_loss,
            self.discriminator.trainable_weights
        )

        self.d_optimizer.apply_gradients(
            zip(
                d_gradients,
                self.discriminator.trainable_weights
            )
        )

        # -------------------------------------------------
        # Train Generator
        # -------------------------------------------------

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        # Generator wants generated images classified as real
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = self.generator(
                random_latent_vectors,
                training=True
            )

            predictions = self.discriminator(
                generated_images,
                training=False
            )

            g_loss = self.loss_fn(
                misleading_labels,
                predictions
            )

        g_gradients = tape.gradient(
            g_loss,
            self.generator.trainable_weights
        )

        self.g_optimizer.apply_gradients(
            zip(
                g_gradients,
                self.generator.trainable_weights
            )
        )

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)

        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result()
        }


# ## Baseline DCGAN Compilation and Training
# 
# After defining the Generator and Discriminator, both networks are combined using the custom `GAN` class.
# 
# The model is then compiled using separate optimizers for the Generator and Discriminator before being trained using the augmented CIFAR-10 training dataset.
# 
# | Component / Parameter | Setting Used | Purpose / Justification |
# |---|---|---|
# | **Discriminator** | `baseline_discriminator` | Determines whether an input image is real or generated. |
# | **Generator** | `baseline_generator` | Generates synthetic CIFAR-10-like images from random latent vectors. |
# | **Latent Dimension** | `100` | Determines the size of the random noise vector used as input to the Generator. |
# | **Discriminator Optimizer** | Adam | Adam adapts the learning rate for individual parameters and is commonly suitable for GAN training. |
# | **Generator Optimizer** | Adam | A separate optimizer allows the Generator weights to be updated independently from the Discriminator. |
# | **Learning Rate** | `0.0002` | A relatively small learning rate is used to avoid very large parameter updates that could make adversarial training unstable. |
# | **Adam `beta_1`** | `0.5` | Reduces the influence of previous gradient momentum compared with the default Adam setting, allowing the optimizer to respond more quickly to changes during adversarial training. |
# | **Loss Function** | Binary Crossentropy | Suitable because the Discriminator performs binary classification between real (`1`) and generated (`0`) images. |
# | **Training Dataset** | `dataset` | Uses the prepared CIFAR-10 dataset containing shuffled, batched and augmented training images. |
# | **Batch Size** | `128` | The dataset was previously divided into batches of 128 images, providing a balance between training efficiency and memory usage. |
# | **Epochs** | `35` | Provides an initial baseline training duration against which the Improved DCGAN can later be compared. |
# 
# ### Why Separate Optimizers Are Used
# 
# The Generator and Discriminator perform different tasks and contain separate trainable parameters.
# 
# Therefore, two Adam optimizers are used:
# 
# - `d_optimizer` updates only the **Discriminator**
# - `g_optimizer` updates only the **Generator**
# 
# This allows each network to learn independently during its respective training step.
# 
# ### Why Binary Crossentropy Is Used
# 
# The Discriminator performs a binary classification task:
# 
# | Image Type | Target Label |
# |---|---:|
# | Real CIFAR-10 image | `1` |
# | Generated image | `0` |
# 
# Binary Crossentropy measures the difference between these target labels and the Discriminator's predicted probabilities.
# 
# During Generator training, generated images are deliberately assigned a target label of `1`. This encourages the Generator to produce images that the Discriminator is more likely to classify as real.
# 
# ### Training Process
# 
# The Baseline DCGAN is trained for **35 epochs** using the dataset.
# 
# During every training batch:
# 
# 1. The Discriminator learns to distinguish real CIFAR-10 images from generated images.
# 2. The Generator learns to create images that can fool the Discriminator.
# 3. The Generator and Discriminator losses are recorded for later evaluation.
# 
# Through this repeated adversarial process, the Generator gradually learns to produce synthetic images that more closely resemble the CIFAR-10 training distribution.

# In[18]:


baseline_gan = GAN(
    discriminator=baseline_discriminator,
    generator=baseline_generator,
    latent_dim=latent_dim
)

baseline_gan.compile(
    d_optimizer=keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5
    ),
    g_optimizer=keras.optimizers.Adam(
        learning_rate=0.0002,
        beta_1=0.5
    ),
    loss_fn=keras.losses.BinaryCrossentropy()
)

baseline_history = baseline_gan.fit(
    x_train,
    epochs=35,
    batch_size=128
)


# # Baseline DCGAN Training Process
# 
# The GAN training process follows these steps:
# 
# 1. Random noise is sampled from the latent space.
# 2. The generator converts the noise into synthetic images.
# 3. The discriminator receives both real and generated images.
# 4. The discriminator updates its weights using Binary Crossentropy loss.
# 5. The generator updates its weights based on feedback from the discriminator.
# 6. The process repeats for 30 training epochs.
# 
# Unlike supervised learning, GANs do not require labelled output images. Instead, the discriminator acts as a learned loss function that guides the generator toward producing increasingly realistic images.

# ## Images generated by the Baseline DCGAN

# In[21]:


display_generated_images(
    baseline_generator,
    latent_dim,
    classifier=classifier,
    num_images=25,
    title="Baseline DCGAN Generated Images"
)


# ## Generated Image Evaluation
# 
# The Baseline DCGAN was evaluated using **CIFAR-10 class predictions**, **manual visual assessment**, **FID**, and **Inception Score**.
# 
# ### Class Prediction Evaluation
# 
# A separate CIFAR-10 classifier was used to estimate which class each generated image most closely resembles. Since the DCGAN is unconditional, these predicted classes are used only for **post-generation evaluation** and are not ground-truth labels.
# 
# Across the 25 generated images, the classifier predicted **8 out of the 10 CIFAR-10 classes**:
# 
# - Cat: 6 images
# - Deer: 5 images
# - Automobile: 4 images
# - Horse: 3 images
# - Truck: 3 images
# - Dog: 2 images
# - Ship: 1 image
# - Bird: 1 image
# 
# No images in this sample were predicted as **Airplane** or **Frog**.
# 
# Several images received high-confidence predictions. Examples include **Horse (98.0%)**, **Cat (95.1%)**, **Ship (94.6%)**, **Automobile (93.6%)**, and **Dog (91.5%)**.
# 
# However, classifier confidence does not directly represent human-perceived image quality. Some images received high-confidence predictions even though the object remained blurry or difficult to recognise. This suggests that the Generator learned certain class-related features such as colours, textures and shapes, but had not yet consistently learned clear object structures.
# 
# Therefore, the predicted classes and confidence scores are treated only as supporting information rather than a direct measure of generated-image quality.
# 
# ### Eye-Power Evaluation
# 
# The same 25 generated images were manually assessed using the **Clear / Marginal / Nonsense** method.
# 
# | Category | Number of Images | Percentage | Description |
# |---|---:|---:|---|
# | **Clear** | 6 | 24% | The main object or overall structure was reasonably recognisable. |
# | **Marginal** | 15 | 60% | Some meaningful shapes, colours or object-like features were visible, but the image remained blurry or ambiguous. |
# | **Nonsense** | 4 | 16% | No clear or coherent object could be confidently identified. |
# | **Total** | **25** | **100%** | |
# 
# Most images were classified as **Marginal (60%)**, indicating that the Baseline DCGAN had started to learn meaningful CIFAR-10 visual features but was not yet able to generate consistently clear and well-defined objects.
# 
# ### Overall Evaluation
# 
# The visual results show that the Baseline DCGAN can generate images containing recognisable CIFAR-10 characteristics, particularly colours, textures and rough object structures. However, many outputs remain blurry and lack clear object boundaries.
# 
# The classifier predictions also show that the generated images contain features associated with several different CIFAR-10 classes, suggesting some level of output diversity. However, the difference between high classifier confidence and human visual quality demonstrates why classifier predictions should not be used as the only evaluation method.
# 
# Therefore, the visual assessment is considered together with the **FID score of 121.45** and **Inception Score of 3.61 ± 0.13** to provide a more complete evaluation of the Baseline DCGAN.

# ### Predicted Class Distribution of Generated Images
# 
# To further evaluate the diversity of the Baseline DCGAN, a larger set of generated images is passed through the CIFAR-10 classifier.
# 
# The predicted classes are counted and displayed using a bar chart. This provides an indication of whether the Generator produces visual features associated with a variety of CIFAR-10 classes or concentrates mainly on a small number of classes.
# 
# Since the DCGAN is unconditional, these predictions are used only for post-generation analysis and do not represent ground-truth labels.

# In[28]:


# Number of generated images to analyse
num_samples = 1000

# Generate random latent vectors
noise = tf.random.normal(
    shape=(num_samples, latent_dim)
)

# Generate images
generated_images = baseline_generator(
    noise,
    training=False
).numpy()

# Predict CIFAR-10 classes
predictions = classifier.predict(
    generated_images,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# Count predicted classes
class_counts = np.bincount(
    predicted_classes,
    minlength=10
)

# Plot class distribution
plt.figure(figsize=(10, 5))

plt.bar(
    class_names,
    class_counts
)

plt.xlabel("Predicted CIFAR-10 Class")
plt.ylabel("Number of Generated Images")
plt.title(
    "Baseline DCGAN: Predicted Class Distribution"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Predicted Class Distribution of Generated Images
# 
# A total of **1,000 images** were generated by the Baseline DCGAN and passed through the CIFAR-10 classifier to examine the diversity of the generated outputs.
# 
# The predictions are distributed across all **10 CIFAR-10 classes**, showing that the Generator is able to produce visual features associated with a variety of object categories.
# 
# However, the distribution is not balanced. **Cat** is predicted much more frequently than the other classes, while classes such as **Airplane, Frog and Ship** appear less often. This suggests that the Baseline DCGAN may have learned some visual patterns more strongly than others.
# 
# Since the DCGAN is unconditional, an equal number of images from each class is not expected. The graph is therefore used to examine the **diversity of generated outputs** rather than class-generation accuracy.
# 
# Overall, the presence of predictions across all ten classes indicates that the model has learned a reasonably diverse range of CIFAR-10 features, although the uneven distribution suggests that there is still some bias towards particular visual patterns.

# ## Baseline DCGAN Training Loss Graph
# 
# This graph shows how the **Generator loss** and **Discriminator loss** change across the 35 training epochs.
# 
# The Generator loss indicates how well the Generator is able to fool the Discriminator, while the Discriminator loss shows how well it can distinguish between real and generated images.
# 
# The graph is used to observe the **training stability and balance** between the two networks. Large or continuous changes may indicate unstable training, while relatively stable losses suggest that neither network is strongly overpowering the other.

# In[29]:


plt.figure(figsize=(10, 5))

plt.plot(
    baseline_history.history["d_loss"],
    label="Discriminator Loss"
)

plt.plot(
    baseline_history.history["g_loss"],
    label="Generator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline DCGAN Training Loss")
plt.legend()
plt.grid()
plt.show()


# ## Evaluation
# 
# ### Generator Loss
# 
# The generator loss decreased from approximately **1.9** at the beginning of training to around **0.72** by the final epoch. This shows that the generator gradually improved its ability to produce images that could fool the discriminator.
# 
# **Observation:**
# 
# - The high initial loss indicates that the early generated images were easy for the discriminator to identify as fake.
# - The generator loss decreased quickly during the first few epochs as the generator learned from the discriminator's feedback.
# - Small fluctuations occurred during training, which is normal for GANs because the generator and discriminator continuously compete with each other.
# - The final loss of approximately **0.72** suggests that the generator became more effective at producing realistic-looking outputs.
# 
# ### Discriminator Loss
# 
# The discriminator loss increased from approximately **0.41** and stabilised around **0.69**.
# 
# **Observation:**
# 
# - The low initial discriminator loss shows that the discriminator could easily distinguish real CIFAR-10 images from generated images at the start of training.
# - As the generator improved, the discriminator found it more difficult to separate real and generated images.
# - The discriminator loss remained relatively stable at around **0.66–0.69** for most of the training process.
# - The stable loss suggests that the discriminator did not completely overpower the generator.
# 
# ### Overall Evaluation
# 
# The loss graph shows a relatively **stable and balanced adversarial training process**. By the end of training, the generator and discriminator losses were close to each other, at approximately **0.72** and **0.69** respectively.
# 
# This suggests that neither network consistently dominated the other, allowing both models to continue learning during training.
# 
# The Baseline DCGAN therefore provides a suitable benchmark for comparison with the Improved DCGAN.

# ### Real vs Generated Pixel Value Distribution Graph
# 
# This graph compares the **pixel-value distribution of real CIFAR-10 images with images generated by the Baseline DCGAN**.
# 
# It is used to check whether the Generator has learned similar colour and brightness patterns to the real dataset. Greater overlap between the two distributions indicates that the generated images have pixel characteristics that are more similar to real images.

# In[30]:


# Generate 1000 images
noise = tf.random.normal(
    shape=(1000, latent_dim)
)

generated_images = baseline_generator(
    noise,
    training=False
).numpy()

# Flatten pixel values
real_pixels = x_test[:1000].flatten()
generated_pixels = generated_images.flatten()

plt.figure(figsize=(8, 5))

plt.hist(
    real_pixels,
    bins=50,
    alpha=0.5,
    density=True,
    label="Real CIFAR-10"
)

plt.hist(
    generated_pixels,
    bins=50,
    alpha=0.5,
    density=True,
    label="Baseline DCGAN"
)

plt.xlabel("Pixel Value")
plt.ylabel("Density")
plt.title("Real vs Generated Pixel Value Distribution")
plt.legend()

plt.show()


# ### Real vs Generated Pixel Distribution
# 
# The generated images have a pixel-value distribution that is similar to the real CIFAR-10 images, showing that the Baseline DCGAN successfully learned the general colour and intensity range of the dataset.
# 
# However, differences are still visible between the two distributions. The generated distribution is smoother and does not fully reproduce some of the patterns found in the real images.
# 
# This suggests that the Baseline Generator learned the general appearance of CIFAR-10 but still had limitations in representing more detailed image characteristics.

# ### Discriminator Predictions Graph
# 
# This graph compares the **Discriminator's predictions for real and generated images**.
# 
# A prediction closer to **1** means the image is considered real, while a prediction closer to **0** means it is considered generated. If the two distributions overlap around **0.5**, the Discriminator has difficulty distinguishing between real and generated images, showing that the Generator is able to confuse it.

# In[31]:


# Select real images
real_images = x_test[:1000]

# Generate fake images
noise = tf.random.normal(
    shape=(1000, latent_dim)
)

generated_images = baseline_generator(
    noise,
    training=False
)

# Get discriminator predictions
real_predictions = baseline_discriminator(
    real_images,
    training=False
).numpy().flatten()

fake_predictions = baseline_discriminator(
    generated_images,
    training=False
).numpy().flatten()

plt.figure(figsize=(8, 5))

plt.hist(
    real_predictions,
    bins=30,
    alpha=0.6,
    label="Real Images"
)

plt.hist(
    fake_predictions,
    bins=30,
    alpha=0.6,
    label="Generated Images"
)

plt.xlabel("Discriminator Prediction")
plt.ylabel("Number of Images")
plt.title("Discriminator Predictions: Real vs Generated Images")
plt.legend()

plt.show()


# ### Discriminator Prediction Distribution
# 
# The discriminator predictions for real and generated images overlap strongly, with most predictions concentrated around **0.50 to 0.55**.
# 
# A prediction close to 0.5 indicates that the Discriminator is uncertain whether an image is real or generated. The strong overlap therefore suggests that the Generator became capable of confusing the Baseline Discriminator.
# 
# However, this does not necessarily mean that the generated images are visually realistic. The Baseline DCGAN still achieved an FID score of **121.45** and an Inception Score of **3.61 ± 0.13**, while the generated images remained limited in clarity and detail.
# 
# Therefore, although the adversarial training became balanced, the results suggest that the model required greater learning capacity to produce higher-quality images.

# ### Baseline FID Score

# In[32]:


baseline_dcgan_fid = fid_evaluator.evaluate(
    generator=baseline_generator,
    latent_dim=latent_dim,
    model_name="Baseline DCGAN",
    generated_image_range="-1_to_1"
)


# ## Evaluation
# 
# ### Fréchet Inception Distance (FID)
# 
# The Baseline DCGAN achieved an **FID score of 119.42** using **1,000 real images and 1,000 generated images**.
# 
# **Observation:**
# 
# - FID measures the difference between the feature distributions of real and generated images.
# - A **lower FID score is better**, as it indicates that the generated images are more similar to the real CIFAR-10 images.
# - The score of **119.42** shows that the Baseline DCGAN has learned some characteristics of the CIFAR-10 distribution, but there is still a noticeable difference between the real and generated images.
# - This suggests that the generated images may capture general colours, shapes or object-like structures but still lack the clarity and detail of real CIFAR-10 images.
# 
# ### Overall Evaluation
# 
# The Baseline DCGAN FID score of **119.42** provides a quantitative benchmark for evaluating the Improved DCGAN.
# 
# If the Improved DCGAN achieves a **lower FID score**, this would indicate that its generated images are more similar to the real CIFAR-10 images and that the proposed improvements were effective.
# 
# The FID score should also be considered together with the **generated image quality, training-loss behaviour and Inception Score** when evaluating the overall performance of the model.

# ### Baseline IS Score

# In[73]:


baseline_is_mean, baseline_is_std = (
    is_evaluator.evaluate(
        generator=baseline_generator,
        latent_dim=latent_dim,
        model_name="Baseline DCGAN",
        generated_image_range="-1_to_1"
    )
)


# ## Evaluation
# 
# ### Inception Score (IS)
# 
# The Baseline DCGAN achieved an **Inception Score of 3.61 ± 0.13** using **1,000 generated images**.
# 
# **Observation:**
# 
# - Inception Score evaluates both the **quality** and **diversity** of generated images.
# - A **higher Inception Score is better**, as it suggests that the generated images are more recognisable while still containing variation across different outputs.
# - The score of **3.61 ± 0.13** indicates that the Baseline DCGAN is able to generate some meaningful and varied image features.
# - However, the score is still relatively limited, suggesting that many generated images may remain unclear or lack strong object structure.
# - The small standard deviation of **0.13** suggests that the score is fairly consistent across the evaluation splits.
# 
# ### Overall Evaluation
# 
# The Inception Score of **3.61 ± 0.13** provides another benchmark for comparison with the Improved DCGAN.
# 
# If the Improved DCGAN achieves a **higher Inception Score**, this would suggest that its generated images are more recognisable and diverse.
# 
# The Inception Score should be considered together with the **FID score of 119.42**, generated image samples and training-loss behaviour to provide a more complete evaluation of the Baseline DCGAN.

# ### Baseline DCGAN Conclusion
# 
# The Baseline DCGAN was able to learn the general visual distribution of the CIFAR-10 dataset and generate varied synthetic images. Its training remained relatively stable, with the Generator and Discriminator reaching a balanced loss.
# 
# However, the generated images were still limited in clarity and detail. The model achieved an **FID score of 119.42** and an **Inception Score of 3.61 ± 0.13**, which provide useful baseline values for comparison with the Improved DCGAN.
# 
# Overall, the Baseline DCGAN provides a suitable reference model for evaluating whether the proposed improvements can produce more realistic and diverse images.

# # 7. Second Model: Improved DCGAN

# # Improvements Made to the Baseline DCGAN
# 
# The Improved DCGAN was developed based on the results from the **Baseline DCGAN loss graph, pixel-value distribution, discriminator predictions, generated image samples, FID score and Inception Score**.
# 
# The Baseline DCGAN achieved relatively stable adversarial training. The Generator and Discriminator losses became close to each other, while the discriminator prediction graph showed considerable overlap between real and generated images around **0.5**. This suggests that neither network strongly dominated the training process.
# 
# However, stable training did not necessarily result in high-quality images. The generated images were still blurry and lacked clear object structures. The model also achieved an **FID score of 121.45** and an **Inception Score of 3.61 ± 0.13**, showing that image quality and diversity could still be improved.
# 
# The pixel-value distribution showed that the Generator had learned the general colour and intensity characteristics of CIFAR-10, but differences from the real-image distribution were still present.
# 
# Therefore, the Improved DCGAN focuses on increasing **feature-learning capacity and image-generation depth** while maintaining stable adversarial training.
# 
# ## Baseline and Improved DCGAN Comparison
# 
# | Improvement | Baseline DCGAN | Improved DCGAN | Why It Was Changed |
# |---|---|---|---|
# | **Generator starting feature map** | `8 × 8 × 128` | `4 × 4 × 256` | The baseline images lacked fine details. A smaller but deeper starting representation gives the Generator greater capacity to learn high-level features before increasing image resolution. |
# | **Generator upsampling stages** | 2 stages | 3 stages | The baseline images remained blurry. Using `4 × 4 → 8 × 8 → 16 × 16 → 32 × 32` allows image features to be developed more gradually. |
# | **Upsampling method** | `Conv2DTranspose` | `UpSampling2D + Conv2D` | Separates image resizing from feature learning, allowing the convolution layers to focus on refining visual features after each increase in resolution. |
# | **Batch Normalisation** | Not used | Added throughout Generator | Helps maintain stable activations and gradient flow as the Generator becomes deeper. |
# | **Generator depth** | 2 upsampling stages | 3 upsampling stages | Provides more stages for learning and refining shapes, textures and object structures. |
# | **Discriminator depth** | 2 Conv2D layers | 3 Conv2D layers | The baseline prediction graph showed considerable uncertainty between real and generated images. A deeper Discriminator can learn more complex features and provide more informative feedback. |
# | **Discriminator filters** | `64 → 128` | `64 → 128 → 256` | The additional 256-filter layer allows deeper patterns such as complex textures and object structures to be examined. |
# | **Generator learning rate** | `0.0002` | `0.0001` | The deeper Generator uses smaller and more controlled weight updates to support stable training. |
# | **Training epochs** | 35 | 100 | The baseline became stable but image quality remained limited. The deeper improved model is given more time to learn and refine its outputs. |
# 
# ### Why Some Settings Were Kept
# 
# Not every part of the Baseline DCGAN required changing.
# 
# The Baseline loss graph showed that adversarial training was already relatively stable. Therefore, settings that worked successfully were retained, including:
# 
# - **Discriminator learning rate:** `0.0001`
# - **Adam beta 1:** `0.5`
# - **LeakyReLU:** negative slope of `0.2`
# - **Dropout:** `0.3`
# - **Generator output:** `tanh`
# 
# Keeping these settings allows the Improved DCGAN to focus on addressing the main weakness identified in the baseline: **limited image quality rather than severe training instability**.
# 
# ### Overall Justification
# 
# The Baseline DCGAN successfully learned the general distribution of CIFAR-10 and achieved balanced Generator and Discriminator training. However, the generated images remained unclear, with an **FID of 121.45** and **Inception Score of 3.61 ± 0.13**.
# 
# Therefore, the Improved DCGAN introduces a **deeper Generator, progressive upsampling, Batch Normalisation, a deeper Discriminator, a lower Generator learning rate and longer training**.
# 
# The Improved model will be considered successful if these changes produce **clearer images, a lower FID score and a higher Inception Score** than the Baseline DCGAN.

# ## Improved DCGAN Generator
# 
# The Improved Generator was designed to produce more detailed CIFAR-10 images by using a deeper architecture and a more gradual upsampling process.
# 
# It begins with a **100-dimensional latent vector**, which is expanded into a `4 × 4 × 256` feature map. The image resolution is then progressively increased through three upsampling stages:
# 
# `4 × 4 → 8 × 8 → 16 × 16 → 32 × 32`
# 
# At each stage, `UpSampling2D` increases the image resolution while `Conv2D` learns and refines image features.
# 
# | Component | Setting Used | Purpose |
# |---|---|---|
# | **Input** | `100` latent features | Provides a random starting representation for image generation. |
# | **Dense Layer** | `4 × 4 × 256` | Converts the latent vector into a small but deep feature representation. |
# | **Batch Normalisation** | Used after Dense and Conv2D layers | Helps stabilise activations and supports more consistent training. |
# | **LeakyReLU** | Negative slope = `0.2` | Maintains gradient flow and supports learning of complex features. |
# | **First Upsampling Stage** | `4 × 4 → 8 × 8` | Begins increasing image resolution. |
# | **Conv2D** | 256 filters | Learns high-level image structures at low resolution. |
# | **Second Upsampling Stage** | `8 × 8 → 16 × 16` | Gradually increases spatial detail. |
# | **Conv2D** | 128 filters | Refines shapes, textures and intermediate features. |
# | **Third Upsampling Stage** | `16 × 16 → 32 × 32` | Produces the final CIFAR-10 image resolution. |
# | **Conv2D** | 64 filters | Refines lower-level visual details before image generation. |
# | **Output Conv2D** | 3 filters | Produces the three RGB colour channels. |
# | **Output Activation** | `tanh` | Produces pixel values between `-1` and `1`, matching the normalised CIFAR-10 images. |
# 
# ### Generator Flow
# 
# `Latent Vector (100)`  
# → `4 × 4 × 256`  
# → `8 × 8 × 256`  
# → `16 × 16 × 128`  
# → `32 × 32 × 64`  
# → `32 × 32 × 3`
# 
# ### Why This Improves the Baseline Generator
# 
# The Baseline Generator started from `8 × 8 × 128` and required only two upsampling stages to reach `32 × 32`. Although it learned the general colours and structures of CIFAR-10, the generated images still lacked clear details.
# 
# The Improved Generator instead begins with a smaller but deeper `4 × 4 × 256` representation and uses **three progressive upsampling stages**. This gives the network more opportunities to gradually build and refine image features.
# 
# Batch Normalisation is also introduced throughout the Generator to support stable training as the network becomes deeper.
# 
# These changes were intended to improve the image quality identified as limited in the Baseline DCGAN while maintaining stable adversarial training.

# In[33]:


improved_generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),

        # Latent vector → 4 × 4 × 256
        layers.Dense(
            4 * 4 * 256,
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((4, 4, 256)),

        # 4 × 4 → 8 × 8
        layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        ),
        layers.Conv2D(
            256,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        # 8 × 8 → 16 × 16
        layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        ),
        layers.Conv2D(
            128,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        # 16 × 16 → 32 × 32
        layers.UpSampling2D(
            size=(2, 2),
            interpolation="nearest"
        ),
        layers.Conv2D(
            64,
            kernel_size=3,
            padding="same",
            use_bias=False
        ),
        layers.BatchNormalization(),
        layers.LeakyReLU(negative_slope=0.2),

        # Produce 32 × 32 RGB image
        layers.Conv2D(
            3,
            kernel_size=3,
            padding="same",
            activation="tanh"
        )
    ],
    name="improved_generator"
)

improved_generator.summary()


# ## Improved DCGAN Discriminator
# 
# The Improved Discriminator was designed to learn more complex image features and provide stronger feedback to the Generator.
# 
# The Baseline Discriminator used two convolutional layers with **64 and 128 filters**. The Improved Discriminator adds a third convolutional layer with **256 filters**, allowing deeper visual features to be learned before the final real or fake prediction.
# 
# ### Discriminator Structure
# 
# | Stage | Output Shape | Main Purpose |
# |---|---|---|
# | Input image | `32 × 32 × 3` | Receives a CIFAR-10 RGB image. |
# | First convolutional block | `16 × 16 × 64` | Learns basic features such as colours, edges and textures. |
# | Second convolutional block | `8 × 8 × 128` | Learns more complex textures and image structures. |
# | Third convolutional block | `4 × 4 × 256` | Learns deeper features such as object shapes and unrealistic generated patterns. |
# | Flatten | 1D feature vector | Prepares the extracted features for classification. |
# | Final output | 1 prediction | Uses `sigmoid` to predict whether the image is real or generated. |
# 
# Each convolutional layer uses a **4 × 4 kernel with stride 2**, which gradually reduces the image size while increasing the number of learned feature maps from:
# 
# `64 → 128 → 256`
# 
# `LeakyReLU(0.2)` is used to maintain gradient flow, while `Dropout(0.3)` helps reduce overfitting and prevents the Discriminator from becoming too confident.
# 
# ### Why This Improves the Baseline Discriminator
# 
# The Baseline discriminator prediction graph showed strong overlap between real and generated images, with many predictions around **0.5**. This indicates that the Baseline Discriminator had difficulty separating some generated images from real CIFAR-10 images.
# 
# Therefore, an additional **256-filter convolutional layer** was added to increase the Discriminator's learning capacity.
# 
# The earlier layers can identify simpler features such as colours, edges and textures, while the deeper layer can combine these features to identify more complex object structures and unrealistic generated patterns.
# 
# This allows the Improved Discriminator to provide more informative feedback to the Generator, encouraging it to produce more realistic images.

# In[34]:


improved_discriminator = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),

        # 32 × 32 → 16 × 16
        layers.Conv2D(
            64,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        # 16 × 16 → 8 × 8
        layers.Conv2D(
            128,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        # 8 × 8 → 4 × 4
        layers.Conv2D(
            256,
            kernel_size=4,
            strides=2,
            padding="same"
        ),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(
            1,
            activation="sigmoid"
        )
    ],
    name="improved_discriminator"
)

improved_discriminator.summary()


# ## Improved DCGAN Training Configuration
# ### Why Reduce the Generator Learning Rate?
# 
# The Baseline Generator used a learning rate of `0.0002`, while the Improved Generator uses `0.0001`.
# 
# Since the Improved Generator is deeper and contains additional upsampling, convolutional and Batch Normalisation layers, a lower learning rate allows its weights to change more gradually.
# 
# This was intended to support **more controlled and stable training** while the larger Generator learns more detailed image features.
# 
# ### Why Keep the Discriminator Learning Rate at `0.0001`?
# 
# The Baseline training-loss graph showed that the Discriminator already trained relatively stably using a learning rate of `0.0001`.
# 
# Therefore, this value was retained instead of unnecessarily increasing its learning speed. The main improvement to the Discriminator comes from its **additional convolutional layer and increased feature-learning capacity**, rather than a higher learning rate.

# In[35]:


improved_gan = GAN(
    discriminator=improved_discriminator,
    generator=improved_generator,
    latent_dim=latent_dim
)

improved_gan.compile(
    d_optimizer=keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.5
    ),
    g_optimizer=keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.5
    ),
    loss_fn=keras.losses.BinaryCrossentropy()
)


# ## Improved DCGAN Training
# ### Why Train for 100 Epochs?
# 
# The Baseline DCGAN was trained for **35 epochs** and achieved relatively stable losses, but its generated images were still blurry and lacked clear object structures.
# 
# Since the Improved DCGAN contains a deeper Generator and Discriminator, it was trained for **100 epochs** to give the additional layers more time to learn complex features and refine the generated images.
# 
# The longer training period allows the effectiveness of the architectural improvements to be evaluated through the **generated images, loss graph, FID score and Inception Score**.
# 
# ### Why Use a Batch Size of 128?
# 
# A batch size of **128** was retained from the Baseline DCGAN to keep the training conditions consistent between both models.
# 
# This makes the comparison fairer, as improvements in performance can be linked more clearly to the changes in architecture, learning rate and training duration rather than a different batch size.

# In[36]:


improved_history = improved_gan.fit(
    x_train,
    epochs=100,
    batch_size=128
)


# ## Images produced by Improved DCGAN

# In[39]:


display_generated_images(
    improved_generator,
    latent_dim,
    num_images=25,
    classifier=classifier,
    title="Improved DCGAN Generated Images"
)



# ## Generated Image Evaluation
# 
# The Improved DCGAN generated images were evaluated using **post-generation CIFAR-10 class predictions** and visual inspection of the generated outputs.
# 
# ### Class Prediction Evaluation
# 
# The CIFAR-10 classifier predicted images from **all 10 CIFAR-10 classes** within this sample of 25 generated images.
# 
# | Predicted Class | Number of Images |
# |---|---:|
# | Airplane | 2 |
# | Automobile | 5 |
# | Bird | 3 |
# | Cat | 2 |
# | Deer | 3 |
# | Dog | 4 |
# | Frog | 1 |
# | Horse | 2 |
# | Ship | 2 |
# | Truck | 1 |
# | **Total** | **25** |
# 
# The most frequently predicted class was **Automobile**, with 5 images, followed by **Dog**, with 4 images. The remaining predictions were spread across the other CIFAR-10 classes.
# 
# Several generated images also received very high classifier confidence scores, including:
# 
# - **Automobile — 100.0%**
# - **Airplane — 99.7%**
# - **Horse — 98.8%**
# - **Ship — 97.3%**
# - **Dog — 95.9%**
# - **Frog — 94.2%**
# - **Truck — 91.6%**
# 
# Compared with the Baseline DCGAN sample, the Improved DCGAN produces more visually defined shapes and clearer object-like structures. Cars, airplanes, ships and some animal-like objects can be identified more easily in several of the generated images.
# 
# However, classifier confidence does not always match human visual judgement. Some images receive high-confidence class predictions even when the object is still blurry or does not clearly resemble the predicted class. Therefore, these predictions are treated as supporting evidence rather than ground-truth labels.
# 
# ### Visual Evaluation
# 
# The generated images show an improvement in **object structure, colour separation and recognisability** compared with the Baseline DCGAN.
# 
# Several outputs contain identifiable structures such as:
# 
# - Cars and other vehicles
# - Airplanes
# - Ships
# - Animal-like faces and bodies
# 
# However, some images remain distorted or ambiguous, particularly for several animal classes. This shows that the Improved DCGAN has learned stronger CIFAR-10 visual features but still does not consistently generate completely clear objects.
# 
# ### Overall Evaluation
# 
# The Improved DCGAN demonstrates better qualitative image generation than the Baseline DCGAN. The appearance of all ten predicted CIFAR-10 classes within the 25-image sample also suggests that the Generator is capable of producing a diverse range of visual features.
# 
# The improved object structures and high-confidence predictions indicate that the architectural and training changes helped the Generator learn more meaningful CIFAR-10 characteristics.
# 
# However, the results should still be evaluated together with the **Predicted Class Distribution, FID score, Inception Score and training-loss behaviour** before determining whether the Improved DCGAN performs better overall.

# ### Predicted Class Distribution of Generated Images
# 
# To further evaluate the diversity of the Baseline DCGAN, a larger set of generated images is passed through the CIFAR-10 classifier.
# 
# The predicted classes are counted and displayed using a bar chart. This provides an indication of whether the Generator produces visual features associated with a variety of CIFAR-10 classes or concentrates mainly on a small number of classes.
# 
# Since the DCGAN is unconditional, these predictions are used only for post-generation analysis and do not represent ground-truth labels.

# In[41]:


# Number of generated images to analyse
num_samples = 1000

# Generate random latent vectors
noise = tf.random.normal(
    shape=(num_samples, latent_dim)
)

# Generate images using the Improved DCGAN
generated_images = improved_generator(
    noise,
    training=False
).numpy()

# Predict CIFAR-10 classes
predictions = classifier.predict(
    generated_images,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# Count predicted classes
class_counts = np.bincount(
    predicted_classes,
    minlength=10
)

# Plot predicted class distribution
plt.figure(figsize=(10, 5))

plt.bar(
    class_names,
    class_counts
)

plt.xlabel("Predicted CIFAR-10 Class")
plt.ylabel("Number of Generated Images")
plt.title(
    "Improved DCGAN: Predicted Class Distribution"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Predicted Class Distribution of Generated Images
# 
# A total of **1,000 images** were generated by the Improved DCGAN and passed through the CIFAR-10 classifier to examine the diversity of the generated outputs.
# 
# | Predicted Class | Number of Images | Percentage |
# |---|---:|---:|
# | Airplane | 58 | 5.8% |
# | Automobile | 140 | 14.0% |
# | Bird | 153 | 15.3% |
# | Cat | 136 | 13.6% |
# | Deer | 73 | 7.3% |
# | Dog | 165 | 16.5% |
# | Frog | 82 | 8.2% |
# | Horse | 86 | 8.6% |
# | Ship | 41 | 4.1% |
# | Truck | 66 | 6.6% |
# | **Total** | **1,000** | **100%** |
# 
# The classifier predicted generated images across **all 10 CIFAR-10 classes**, showing that the Improved DCGAN produces a variety of class-related visual features.
# 
# **Dog** was the most frequently predicted class at **16.5%**, followed by **Bird (15.3%)**, **Automobile (14.0%)** and **Cat (13.6%)**. **Ship** was the least frequently predicted class at **4.1%**.
# 
# Compared with the Baseline DCGAN, the predictions are less heavily concentrated in a single class and are more widely distributed across the ten categories. This suggests that the Improved DCGAN produces a more diverse range of CIFAR-10-like features.
# 
# However, the distribution is still not completely balanced. Since the Improved DCGAN is an **unconditional model**, equal representation of every class is not expected. These predicted classes are therefore used to evaluate **output diversity** rather than class-generation accuracy.

# In[43]:


plt.figure(figsize=(10, 5))

plt.plot(
    improved_history.history["d_loss"],
    label="Discriminator Loss"
)

plt.plot(
    improved_history.history["g_loss"],
    label="Generator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved DCGAN Training Loss")
plt.legend()
plt.grid()
plt.show()


# ## Evaluation
# 
# ### Generator Loss
# 
# The Generator loss started at approximately **0.74**, increased sharply to around **0.80** during the early epochs, and then decreased to approximately **0.75**. After this early adjustment period, the Generator loss remained relatively stable before gradually increasing to around **0.79–0.80** towards the end of training.
# 
# **Observation:**
# 
# - The early increase shows that the Generator initially struggled while adapting to the Improved Discriminator.
# - After approximately the first 20 epochs, the Generator loss became more stable.
# - The gradual increase towards the later epochs suggests that the Discriminator became slightly better at distinguishing generated images.
# - However, there were no extreme spikes or repeated large fluctuations during the later training stages.
# 
# ### Discriminator Loss
# 
# The Discriminator loss remained very stable at approximately **0.68–0.69** throughout the 100 epochs.
# 
# **Observation:**
# 
# - The stable loss suggests that the Discriminator did not become excessively dominant.
# - Only small fluctuations occurred during training.
# - The loss gradually decreased slightly towards the end, showing that the Discriminator continued improving without causing severe instability.
# - The consistency of the loss indicates that the deeper Discriminator was able to learn while maintaining a relatively balanced adversarial training process.
# 
# ### Overall Evaluation
# 
# The Improved DCGAN shows a **relatively stable adversarial training process** across the 100 epochs.
# 
# The Discriminator loss remained close to **0.68**, while the Generator loss stayed mainly between approximately **0.75 and 0.80** after the initial adjustment period. Although the Generator loss gradually increased towards the end, neither network showed signs of completely overpowering the other.
# 
# The stable loss behaviour suggests that the Improved DCGAN was able to train successfully despite using a **deeper Generator and Discriminator**.
# 
# However, GAN loss values alone cannot determine whether image quality has improved. Therefore, the training-loss graph should be considered together with the **generated image samples, predicted class distribution, FID score and Inception Score** when comparing the Improved DCGAN with the Baseline DCGAN.

# ### Improved DCGAN: Real vs Generated Pixel Value Distribution

# In[44]:


# Generate 1000 images
noise = tf.random.normal(
    shape=(1000, latent_dim)
)

generated_images = improved_generator(
    noise,
    training=False
).numpy()

# Flatten pixel values
real_pixels = x_test[:1000].flatten()
generated_pixels = generated_images.flatten()

plt.figure(figsize=(8, 5))

plt.hist(
    real_pixels,
    bins=50,
    alpha=0.5,
    density=True,
    label="Real CIFAR-10"
)

plt.hist(
    generated_pixels,
    bins=50,
    alpha=0.5,
    density=True,
    label="Improved DCGAN"
)

plt.xlabel("Pixel Value")
plt.ylabel("Density")
plt.title("Real vs Improved DCGAN Pixel Value Distribution")
plt.legend()

plt.show()


# ## Evaluation
# 
# ### Pixel Value Distribution
# 
# The pixel-value distribution of the Improved DCGAN generated images closely overlaps with the distribution of the real CIFAR-10 images across most of the range from **−1 to 1**.
# 
# **Observation:**
# 
# - The strong overlap indicates that the Improved DCGAN learned a pixel-value distribution that is similar to the real CIFAR-10 dataset.
# - Both real and generated images contain a wide range of dark, medium and bright pixel values rather than concentrating within a narrow range.
# - The generated-image distribution appears smoother than the real-image distribution, while the real images contain several sharper peaks at particular pixel values.
# - Some differences are still visible near the extreme pixel values, particularly close to **1.0**, where the real images have a higher concentration.
# - This suggests that the Generator has learned the overall colour and brightness distribution well, although it does not reproduce every characteristic of the real-image distribution exactly.
# 
# ### Overall Pixel Distribution Evaluation
# 
# The close overlap between the two distributions provides supporting evidence that the Improved DCGAN is generating images with pixel characteristics similar to real CIFAR-10 images.
# 
# However, a similar pixel distribution does not necessarily mean that the generated objects are visually realistic. Two images can have similar colour and brightness distributions while containing very different object structures.
# 
# Therefore, this result should be considered together with the **generated image samples, FID score, Inception Score and class-distribution analysis**.
# 
# ---

# ### Improved DCGAN Discriminator Predictions

# In[45]:


# Select real images
real_images = x_test[:1000]

# Generate fake images
noise = tf.random.normal(
    shape=(1000, latent_dim)
)

generated_images = improved_generator(
    noise,
    training=False
)

# Get discriminator predictions
real_predictions = improved_discriminator(
    real_images,
    training=False
).numpy().flatten()

fake_predictions = improved_discriminator(
    generated_images,
    training=False
).numpy().flatten()

plt.figure(figsize=(8, 5))

plt.hist(
    real_predictions,
    bins=30,
    alpha=0.6,
    label="Real Images"
)

plt.hist(
    fake_predictions,
    bins=30,
    alpha=0.6,
    label="Generated Images"
)

plt.xlabel("Discriminator Prediction")
plt.ylabel("Number of Images")
plt.title(
    "Improved DCGAN Discriminator Predictions: "
    "Real vs Generated Images"
)

plt.legend()

plt.show()


# ### Discriminator Prediction Distribution
# 
# The second graph compares the Improved DCGAN Discriminator predictions for **real and generated images**.
# 
# Both distributions are concentrated mainly around approximately **0.45–0.50**, with a large amount of overlap between them.
# 
# **Observation:**
# 
# - The strong overlap shows that the Discriminator gives similar prediction values to many real and generated images.
# - Most predictions remain close to **0.5**, indicating that the Discriminator is often uncertain about whether an image is real or generated.
# - The Discriminator does not consistently assign extremely high confidence to real images or extremely low confidence to generated images.
# - This suggests that the Generator is producing images that are sufficiently similar to the training distribution to make discrimination more difficult.
# - The generated-image predictions also show a slightly wider spread towards lower values, indicating that some generated images are still easier for the Discriminator to identify as fake.
# 
# ### Overall Discriminator Evaluation
# 
# The large overlap between the real and generated prediction distributions suggests that the **Generator and Discriminator have reached a relatively competitive state**.
# 
# The Discriminator is not completely overpowering the Generator, since it cannot clearly separate the two distributions. At the same time, the distributions are not perfectly identical, showing that differences between real and generated images still remain.
# 
# This result is consistent with the relatively stable **Generator and Discriminator loss curves** observed during training.
# 
# However, Discriminator predictions alone cannot confirm generated-image quality. The final performance of the Improved DCGAN should therefore be evaluated using the **visual results, predicted class distribution, FID score and Inception Score** together.

# ### Improved DCGAN FID Score

# In[46]:


improved_dcgan_fid = fid_evaluator.evaluate(
    generator=improved_generator,
    latent_dim=latent_dim,
    model_name="Improved DCGAN",
    generated_image_range="-1_to_1"
)


# ## Evaluation
# 
# ### Fréchet Inception Distance (FID)
# 
# The Improved DCGAN achieved an **FID score of 94.77** using **1,000 real images and 1,000 generated images**.
# 
# **Observation:**
# 
# - FID measures the difference between the feature distributions of real and generated images.
# - A **lower FID score is better**, as it indicates that the generated images are more similar to the real CIFAR-10 images.
# - The Improved DCGAN achieved a lower FID score compared with the Baseline DCGAN score of **121.45**.
# - This suggests that the architectural improvements helped the model generate images with features that are closer to the real CIFAR-10 distribution.
# 
# ### Overall Evaluation
# 
# The Improved DCGAN achieved an FID score of **94.77**, compared with **121.45** for the Baseline DCGAN.
# 
# The reduction in FID indicates that the Improved DCGAN generated images that were more similar to real CIFAR-10 images.
# 
# This suggests that improvements such as the **deeper Generator, progressive upsampling, Batch Normalisation and deeper Discriminator** were effective in improving image generation.
# 
# The FID result should also be considered together with the **generated image quality, training-loss behaviour and Inception Score** when evaluating the overall performance of the Improved DCGAN.

# ### Improved DCGAN IS Score

# In[47]:


improved_is_mean, improved_is_std = (
    is_evaluator.evaluate(
        generator=improved_generator,
        latent_dim=latent_dim,
        model_name="Improved DCGAN",
        generated_image_range="-1_to_1"
    )
)


# ## Evaluation
# 
# ### Inception Score (IS)
# 
# The Improved DCGAN achieved an **Inception Score of 5.36 ± 0.29** using **1,000 generated images**.
# 
# **Observation:**
# 
# - Inception Score evaluates both the **quality** and **diversity** of generated images.
# - A **higher Inception Score is better**, as it suggests that the generated images are more recognisable and varied.
# - The Improved DCGAN achieved a much higher score than the Baseline DCGAN, which obtained **3.61 ± 0.13**.
# - This improvement suggests that the deeper Generator, progressive upsampling and Batch Normalisation helped the model produce more meaningful and diverse image features.
# - The score also indicates that the Improved DCGAN generated images with stronger class-related characteristics compared with the Baseline model.
# 
# ### Overall Evaluation
# 
# The Improved DCGAN achieved an Inception Score of **5.36 ± 0.29**, compared with **3.61 ± 0.13** for the Baseline DCGAN.
# 
# This increase shows that the Improved DCGAN generated images that were generally **more recognisable and diverse**.
# 
# The result suggests that the architectural improvements were effective in improving the quality of the generated CIFAR-10 images.
# 
# The Inception Score should also be considered together with the **FID score, generated image samples and training-loss behaviour** when evaluating the overall performance of the Improved DCGAN.

# ## Overall Comparison of Baseline and Improved DCGAN
# 
# The Baseline and Improved DCGAN models were compared using several evaluation methods:
# 
# - Generated-image visualisation
# - **Clear / Marginal / Nonsense eye-power assessment**
# - Post-generation CIFAR-10 class predictions
# - Predicted class distribution
# - Fréchet Inception Distance (FID)
# - Inception Score (IS)
# - Generator and Discriminator loss behaviour
# 
# Using multiple evaluation methods provides a more complete comparison because no single metric can fully represent the quality and diversity of generated images.
# 
# ---
# 
# ### Quantitative Score Comparison
# 
# | Evaluation Metric | Baseline DCGAN | Improved DCGAN | Better Model |
# |---|---:|---:|---|
# | **FID Score** | **119.42** | **87.60** | **Improved DCGAN** |
# | **Inception Score** | **3.61 ± 0.13** | **5.36 ± 0.29** | **Improved DCGAN** |
# 
# The Improved DCGAN achieved an **FID score of 87.60**, compared with **119.42** for the Baseline DCGAN.
# 
# Since a **lower FID score is better**, this shows that the feature distribution of the Improved DCGAN generated images is closer to the real CIFAR-10 image distribution.
# 
# The Improved DCGAN also achieved an **Inception Score of 5.36 ± 0.29**, compared with **3.61 ± 0.13** for the Baseline DCGAN.
# 
# Since a **higher Inception Score is better**, this suggests that the Improved DCGAN generated images with more recognisable and diverse visual features.
# 
# Both quantitative metrics therefore show an improvement over the Baseline DCGAN.
# 
# ---
# 
# ### Manual Eye-Power Evaluation
# 
# The generated images were also manually assessed using the **Clear / Marginal / Nonsense eye-power method**.
# 
# | Category | Definition |
# |---|---|
# | **Clear** | A reasonably recognisable object with meaningful colour and structure. |
# | **Marginal** | Some object-like features are visible, but the image remains blurry, distorted or difficult to identify. |
# | **Nonsense** | No meaningful or recognisable object can be clearly identified. |
# 
# The same criteria were applied consistently to 25 generated images from each model.
# 
# | Model | Clear | Marginal | Nonsense |
# |---|---:|---:|---:|
# | **Baseline DCGAN** | 6 (24%) | 15 (60%) | 4 (16%) |
# | **Improved DCGAN** | **13 (52%)** | 10 (40%) | **2 (8%)** |
# 
# The Improved DCGAN produced **13 clear images (52%)**, compared with **6 clear images (24%)** from the Baseline DCGAN.
# 
# The percentage of **Clear** images therefore increased, while the proportion of **Nonsense** images decreased from **16% to 8%**.
# 
# This supports the quantitative results by showing that the Improved DCGAN also produced more recognisable images during manual visual inspection.
# 
# However, manual assessment is subjective, so these results are considered together with FID, Inception Score and the other evaluation methods.
# 
# ---
# 
# ### Post-Generation Class Prediction
# 
# A separate CIFAR-10 classifier was used to estimate which class each generated image most closely resembled.
# 
# The predicted class and confidence score were displayed above each generated image. This provides additional information about whether the generated images contain visual features associated with CIFAR-10 classes.
# 
# However, classifier confidence does **not necessarily mean that an image is visually clear**. Some blurry images can still receive high-confidence predictions because they contain colours, textures or shapes strongly associated with a particular class.
# 
# Therefore, class predictions are used only as a **supporting post-generation evaluation method** and are not treated as ground-truth labels.
# 
# ---
# 
# ### Predicted Class Distribution
# 
# The predicted class distributions of **1,000 generated images** were also examined for both models.
# 
# The Baseline DCGAN produced images across all ten CIFAR-10 classes, but its predictions were more strongly concentrated in certain classes, particularly **Cat**.
# 
# The Improved DCGAN also produced predictions across all ten classes, but the generated images were distributed more widely across categories such as **Dog, Bird, Automobile, Cat, Frog and Horse**.
# 
# This suggests that the Improved DCGAN learned a broader range of class-related visual characteristics and reduced some of the concentration seen in the Baseline model.
# 
# Since both models are **unconditional DCGANs**, a perfectly balanced class distribution is not expected. The class distribution is therefore used to evaluate **output diversity**, rather than class-generation accuracy.
# 
# ---
# 
# ### Training Loss Comparison
# 
# Both models achieved relatively stable adversarial training.
# 
# The Baseline DCGAN maintained Generator and Discriminator losses without extreme fluctuations, showing that neither network completely dominated the other.
# 
# The Improved DCGAN also remained relatively stable across **100 epochs**. Its Discriminator loss remained close to approximately **0.68–0.69**, while the Generator loss stayed mainly around **0.75–0.80** after the initial adjustment period.
# 
# The deeper architecture therefore did not cause severe adversarial instability.
# 
# However, GAN loss values alone cannot determine image quality. A higher or lower Generator loss does not automatically mean that the generated images are better or worse.
# 
# For this reason, the loss curves are interpreted together with the generated images, FID, Inception Score and visual assessments.
# 
# ---
# 
# ### Overall Comparison
# 
# The results show that the **Improved DCGAN performed better than the Baseline DCGAN overall**.
# 
# The Baseline DCGAN successfully learned general CIFAR-10 colours, textures and object-like structures, but many generated images remained blurry or difficult to recognise.
# 
# The Improved DCGAN introduced several changes:
# 
# - A deeper Generator
# - More gradual image upsampling
# - `UpSampling2D + Conv2D`
# - Batch Normalisation
# - A deeper Discriminator with an additional 256-filter layer
# - A lower Generator learning rate
# - Longer training of 100 epochs
# 
# These changes resulted in:
# 
# - A lower **FID score of 87.60**, compared with **119.42**
# - A higher **Inception Score of 5.36 ± 0.29**, compared with **3.61 ± 0.13**
# - An increase in **Clear images from 24% to 52%**
# - A reduction in **Nonsense images from 16% to 8%**
# - More recognisable object structures
# - A wider predicted class distribution
# - Relatively stable adversarial training
# 
# The improvement in both **FID and Inception Score** is particularly important because the two metrics measure different aspects of generated-image performance. The lower FID indicates that the Improved DCGAN generated images are closer to the real CIFAR-10 distribution, while the higher Inception Score indicates improved recognisability and diversity.
# 
# These quantitative results are also supported by the visual evaluation, where the Improved DCGAN produced more clearly recognisable objects and fewer meaningless outputs.
# 
# Therefore, the **Improved DCGAN is selected as the better-performing DCGAN model** for this experiment.
# 
# However, some generated images remain blurry or ambiguous. This shows that although the architectural and training improvements were effective, generating detailed and consistently recognisable objects remains challenging at the small **32 × 32 pixel resolution** of CIFAR-10.

# 
# ## Generating Images from a Specific Class using DCGAN
# 
# A standard DCGAN can generate images that resemble the different CIFAR-10 classes, but it cannot be instructed to generate only one selected class, such as ships, cats, or automobiles.
# 
# This is because the DCGAN is trained using the images without their class labels. The generator receives only a random noise vector and learns the overall distribution of all ten CIFAR-10 classes together. Therefore, the class of the generated image depends on the random input and cannot be directly controlled.
# 
# For example:
# 
# **Random noise → Generator → Random CIFAR-10-like image**
# 
# The output may resemble a ship, cat, truck, or another object, but the user cannot choose the class beforehand.
# 
# To generate images from a specific class, the model could be extended into a **Conditional GAN (CGAN)**. A CGAN supplies both a random noise vector and a class label to the generator.
# 
# For example:
# 
# **Random noise + "ship" label → Generator → Generated ship image**
# 
# The discriminator also receives the image and its class label, allowing it to determine whether the image is both realistic and consistent with the requested class.
# 
# A CGAN was not used in this assignment because the main objective was to compare standard DCGAN and VAE models in generating general CIFAR-10 images. Adding class labels would change the model into a conditional architecture and introduce additional complexity beyond the required scope. Using standard models also allows a fairer comparison between the baseline and improved versions because they are trained under the same unconditional image-generation setting.
# 
# Therefore, a CGAN or Conditional VAE could be considered as a future improvement when controlled generation of a specific CIFAR-10 class is required.
# `
# 

# # 8. Third Model: Baseline Variational Autoencoder (VAE)
# 
# The third model implemented in this project is a **Baseline Variational Autoencoder (VAE)**. Unlike the DCGAN models, which learn through competition between a Generator and Discriminator, a VAE learns by encoding images into a structured latent space and reconstructing them through a Decoder.
# 
# ## Why Use a VAE?
# 
# A VAE was included to provide a different approach to image generation and to compare it with the GAN-based models.
# 
# The DCGAN focuses mainly on generating realistic-looking images through adversarial training. In comparison, the VAE learns a probability-based latent representation of the CIFAR-10 images. This allows new images to be generated by sampling from the learned latent space.
# 
# Using a VAE is useful for this project because it allows the following areas to be studied:
# 
# - **Reconstruction ability** – the original CIFAR-10 image can be compared directly with its reconstructed version.
# - **Structured latent space** – similar image features are represented within a continuous latent space.
# - **Stable training** – the VAE does not require a Generator and Discriminator to compete against each other, so training is generally easier to monitor.
# - **Different image-generation approach** – it provides a useful comparison against the DCGAN models.
# - **Generative capability** – new images can still be produced by sampling latent vectors and passing them through the Decoder.
# 
# However, VAEs commonly produce smoother or blurrier images because their objective focuses on both reconstruction accuracy and maintaining a structured latent distribution.
# 
# Therefore, including both **DCGAN and VAE models** allows the project to compare two different generative modelling approaches rather than relying on only one type of model.
# 
# ---
# 
# ## 8.1 Baseline VAE Architecture
# 
# The Baseline VAE consists of three main components:
# 
# 1. **Encoder** – compresses the input CIFAR-10 image into a latent representation.
# 2. **Sampling Layer** – samples a latent vector using the learned mean and variance.
# 3. **Decoder** – reconstructs or generates an image from the latent vector.
# 
# The overall process is:
# 
# `Input Image → Encoder → Latent Distribution → Sampling → Decoder → Reconstructed Image`
# 
# ---
# 
# ## 8.2 Baseline VAE Training
# 
# The Baseline VAE is trained using two main loss components:
# 
# - **Reconstruction Loss** – measures the difference between the original and reconstructed images.
# - **KL Divergence Loss** – encourages the latent representation to follow a structured probability distribution.
# 
# The total loss is calculated as:
# 
# **Total Loss = Reconstruction Loss + β × KL Loss**
# 
# For the Baseline VAE, `β = 1.0` is used.
# 
# ---
# 
# ## 8.3 Baseline VAE Evaluation
# 
# The Baseline VAE is evaluated using:
# 
# - Total VAE loss
# - Reconstruction loss
# - KL divergence loss
# - Original vs reconstructed images
# - Generated-image visualisation
# - **Clear / Marginal / Nonsense eye-power assessment**
# - Fréchet Inception Distance (FID)
# - Inception Score (IS)
# 
# These evaluation methods allow both the **reconstruction performance** and **generation quality** of the VAE to be assessed.

# ### Sampling Layer
# 
# Unlike a normal Autoencoder, the VAE does not encode each image into a single fixed latent vector. Instead, the Encoder learns two values:
# 
# - `z_mean` – the mean of the latent distribution
# - `z_log_var` – the logarithm of the variance of the latent distribution
# 
# A latent vector `z` is then sampled from this distribution using the **reparameterisation trick**:
# 
# \[
# z = z_{mean} + e^{0.5z_{log\_var}} \times \epsilon
# \]
# 
# where \(\epsilon\) is randomly sampled from a standard normal distribution.
# 
# The reparameterisation trick allows randomness to be included in the latent representation while still allowing gradients to pass through the network during training.
# 
# This is important because it allows the VAE to learn a **continuous and structured latent space**, rather than simply memorising fixed representations of the training images.
# 
# The sampling process can be summarised as:
# 
# `z_mean + z_log_var → Sampling → z → Decoder`

# In[48]:


class Sampling(layers.Layer):
    """Sample a latent vector using the reparameterisation trick."""

    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch_size = tf.shape(z_mean)[0]
        latent_size = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(
            shape=(batch_size, latent_size)
        )

        return (
            z_mean
            + tf.exp(0.5 * z_log_var) * epsilon
        )


# ## 8.1 Baseline VAE Encoder
# 
# The Encoder compresses each CIFAR-10 image from its original size of `32 × 32 × 3` into a lower-dimensional latent representation.
# 
# A latent dimension of **100** is used so that each image is represented using 100 latent features.
# 
# ### Encoder Architecture
# 
# | Layer | Output Shape | Purpose |
# |---|---|---|
# | **Input** | `32 × 32 × 3` | Receives the CIFAR-10 RGB image. |
# | **Conv2D (32 filters)** | `16 × 16 × 32` | Extracts basic image features while reducing the spatial size. |
# | **Conv2D (64 filters)** | `8 × 8 × 64` | Learns more complex visual features and further reduces the image dimensions. |
# | **Flatten** | `4096` | Converts the feature maps into a 1D feature vector. |
# | **Dense (128)** | `128` | Combines the extracted features before producing the latent distribution. |
# | **z_mean** | `100` | Represents the mean of the latent distribution. |
# | **z_log_var** | `100` | Represents the log variance of the latent distribution. |
# | **Sampling** | `100` | Samples the final latent vector `z` using `z_mean` and `z_log_var`. |
# 
# ### Encoder Process
# 
# The spatial dimensions are gradually reduced:
# 
# `32 × 32 × 3 → 16 × 16 × 32 → 8 × 8 × 64 → 4096 → 128`
# 
# The network then produces two vectors:
# 
# - `z_mean`
# - `z_log_var`
# 
# Both vectors contain **100 values**, matching the selected latent dimension.
# 
# These two outputs are passed into the Sampling layer, which creates the final latent vector `z`.
# 
# Therefore, the complete Encoder flow is:
# 
# `Input Image → Convolution Layers → Flatten → Dense → z_mean + z_log_var → Sampling → z`
# 
# Using separate mean and variance vectors allows the VAE to learn a **probability distribution** in the latent space instead of storing each image as one fixed representation.

# In[49]:


latent_dim = 100
baseline_vae_encoder_inputs = keras.Input(
    shape=(32, 32, 3),
    name="baseline_vae_encoder_input"
)

# 32 × 32 → 16 × 16
x = layers.Conv2D(
    32,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(baseline_vae_encoder_inputs)

# 16 × 16 → 8 × 8
x = layers.Conv2D(
    64,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

x = layers.Flatten()(x)
x = layers.Dense(
    128,
    activation="relu"
)(x)

baseline_z_mean = layers.Dense(
    latent_dim,
    name="baseline_z_mean"
)(x)

baseline_z_log_var = layers.Dense(
    latent_dim,
    name="baseline_z_log_var"
)(x)

baseline_z = Sampling()(
    [baseline_z_mean, baseline_z_log_var]
)

baseline_vae_encoder = keras.Model(
    baseline_vae_encoder_inputs,
    [
        baseline_z_mean,
        baseline_z_log_var,
        baseline_z
    ],
    name="baseline_vae_encoder"
)

baseline_vae_encoder.summary()


# ## 8.2 Baseline VAE Decoder
# 
# The Decoder takes the sampled latent vector `z` and converts it back into a `32 × 32 × 3` CIFAR-10 image.
# 
# Its purpose is to reverse the compression performed by the Encoder and reconstruct the important visual information stored in the latent representation.
# 
# ### Decoder Architecture
# 
# | Layer | Output Shape | Purpose |
# |---|---|---|
# | **Latent Input** | `100` | Receives the sampled latent vector `z`. |
# | **Dense** | `4096` | Expands the compressed latent representation into enough features for image reconstruction. |
# | **Reshape** | `8 × 8 × 64` | Converts the 1D feature vector back into spatial feature maps. |
# | **Conv2DTranspose (64 filters)** | `16 × 16 × 64` | Doubles the image resolution and begins reconstructing larger visual structures. |
# | **Conv2DTranspose (32 filters)** | `32 × 32 × 32` | Restores the image to the original CIFAR-10 spatial resolution. |
# | **Conv2D + Tanh** | `32 × 32 × 3` | Produces the final RGB image with three colour channels. |
# 
# ### Why This Decoder Design Was Used
# 
# #### Starting from `8 × 8 × 64`
# 
# The Encoder reduces the input image to an `8 × 8` feature representation before producing the latent variables. Therefore, the Decoder begins reconstruction from an `8 × 8 × 64` feature map.
# 
# The Dense layer expands the 100-dimensional latent vector into:
# 
# `8 × 8 × 64 = 4096`
# 
# features before reshaping them into spatial feature maps.
# 
# This gives the Decoder a suitable starting point for rebuilding the image.
# 
# ---
# 
# #### Conv2DTranspose for Upsampling
# 
# Two `Conv2DTranspose` layers are used to gradually increase the image resolution:
# 
# `8 × 8 → 16 × 16 → 32 × 32`
# 
# This was chosen because the Decoder needs to reverse the spatial reduction performed by the Encoder.
# 
# Gradual upsampling allows the model to reconstruct image structure in stages instead of attempting to create the full `32 × 32` image immediately.
# 
# ---
# 
# #### Filters Reduce from 64 to 32
# 
# The first reconstruction stage uses **64 filters** to learn and combine a larger number of visual features.
# 
# As the image becomes larger, the number of filters is reduced to **32** because the network is moving closer to the final RGB image.
# 
# The overall progression is:
# 
# `64 feature maps → 32 feature maps → 3 RGB channels`
# 
# ---
# 
# #### ReLU Activation
# 
# `ReLU` is used in the hidden Decoder layers because it introduces non-linearity and allows the network to learn more complex relationships between latent features and image structures.
# 
# ---
# 
# #### Tanh Output Activation
# 
# The final layer uses a `tanh` activation:
# 
# `Conv2D(3, activation="tanh")`
# 
# This was selected because the CIFAR-10 images were normalised to the range:
# 
# `[-1, 1]`
# 
# The `tanh` activation also produces values within `[-1, 1]`, ensuring that the Decoder output is consistent with the scale of the training images.
# 
# ---
# 
# ### Decoder Process
# 
# The complete Decoder process is:
# 
# `Latent Vector (100) → Dense (4096) → 8 × 8 × 64 → 16 × 16 × 64 → 32 × 32 × 32 → 32 × 32 × 3`
# 
# Together, the Encoder and Decoder form the main reconstruction pathway of the Baseline VAE:
# 
# `Input Image → Encoder → Latent Space → Sampling → Decoder → Reconstructed Image`

# In[50]:


baseline_vae_latent_inputs = keras.Input(
    shape=(latent_dim,),
    name="baseline_vae_latent_input"
)

x = layers.Dense(
    8 * 8 * 64,
    activation="relu"
)(baseline_vae_latent_inputs)

x = layers.Reshape(
    (8, 8, 64)
)(x)

# 8 × 8 → 16 × 16
x = layers.Conv2DTranspose(
    64,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

# 16 × 16 → 32 × 32
x = layers.Conv2DTranspose(
    32,
    kernel_size=3,
    strides=2,
    padding="same",
    activation="relu"
)(x)

# Output remains in [-1, 1] to match x_train
baseline_vae_decoder_outputs = layers.Conv2D(
    3,
    kernel_size=3,
    padding="same",
    activation="tanh"
)(x)

baseline_vae_decoder = keras.Model(
    baseline_vae_latent_inputs,
    baseline_vae_decoder_outputs,
    name="baseline_vae_decoder"
)

baseline_vae_decoder.summary()


# ## 8.3 Baseline VAE Training and Loss Function
# 
# A custom `VAE` model class was created to control how the Encoder and Decoder are trained together.
# 
# Unlike the DCGAN, which trains a Generator and Discriminator separately, the VAE is trained by minimising two main loss components:
# 
# 1. **Reconstruction Loss**
# 2. **KL Divergence Loss**
# 
# These losses are combined to form the **Total VAE Loss**.
# 
# ### Training Process
# 
# For each batch of CIFAR-10 images, the following process is performed:
# 
# `Input Image → Encoder → z_mean + z_log_var → Sampling → z → Decoder → Reconstruction`
# 
# The reconstructed image is then compared with the original image to calculate the loss.
# 
# ---
# 
# ### Reconstruction Loss
# 
# The reconstruction loss measures the difference between the original CIFAR-10 image and the image reconstructed by the Decoder.
# 
# The model uses the squared difference between each original and reconstructed pixel:
# 
# \[
# L_{reconstruction} =
# \sum (x - \hat{x})^2
# \]
# 
# where:
# 
# - \(x\) represents the original image.
# - \(\hat{x}\) represents the reconstructed image.
# 
# The error is summed across the image height, width and RGB channels before being averaged across the batch.
# 
# **Why use reconstruction loss?**
# 
# The VAE needs to preserve important visual information while compressing an image into the latent space. Minimising reconstruction loss encourages the Decoder to reproduce the colours, shapes and structures found in the original CIFAR-10 images.
# 
# A **lower reconstruction loss** generally indicates that the reconstructed images are closer to the originals.
# 
# ---
# 
# ### KL Divergence Loss
# 
# The second component is the **Kullback-Leibler (KL) Divergence Loss**.
# 
# It encourages the latent distributions produced by the Encoder to remain close to a standard normal distribution:
# 
# \[
# N(0,1)
# \]
# 
# The KL loss is calculated using `z_mean` and `z_log_var`.
# 
# **Why use KL loss?**
# 
# Without KL loss, the Encoder could place different images at unrelated locations in the latent space. This would make random sampling difficult and the model would behave more like a normal Autoencoder.
# 
# KL loss encourages the latent space to remain **continuous and structured**, allowing new latent vectors to be sampled and passed through the Decoder to generate new images.
# 
# However, KL loss should not simply be made as small as possible. There needs to be a balance between maintaining a structured latent space and preserving enough information for good reconstruction.
# 
# ---
# 
# ### Total VAE Loss
# 
# The two losses are combined using:
# 
# \[
# L_{total} = L_{reconstruction} + \beta L_{KL}
# \]
# 
# For the Baseline VAE:
# 
# `β = 1.0`
# 
# Therefore, the KL loss is included at its full weighting during training.
# 
# **Why use β?**
# 
# The `beta` parameter controls the balance between:
# 
# - Reconstructing the input image accurately.
# - Maintaining a structured latent space.
# 
# A larger beta gives more importance to the KL loss, while a smaller beta gives more importance to reconstruction quality.
# 
# A value of **1.0** was selected for the Baseline VAE because it represents the standard VAE objective and provides a suitable starting point before testing improvements.
# 
# ---
# 
# ### Gradient-Based Training
# 
# `tf.GradientTape()` is used to record the operations involved in calculating the total loss.
# 
# The gradients of the total loss are then calculated and applied to all trainable weights in both the Encoder and Decoder.
# 
# This allows both parts of the VAE to learn together from the same objective.
# 
# ---
# 
# ### Loss Tracking
# 
# Three metrics are recorded during training:
# 
# | Metric | Purpose |
# |---|---|
# | **Total Loss** | Overall VAE training objective |
# | **Reconstruction Loss** | Measures how closely reconstructed images match the originals |
# | **KL Loss** | Measures how closely the latent distribution follows the desired normal distribution |
# 
# Tracking the three losses separately is important because the Total Loss alone does not show whether poor performance comes from image reconstruction or latent-space regularisation.
# 
# These values will later be plotted across the training epochs to evaluate the stability and learning behaviour of the Baseline VAE.

# In[51]:


class VAE(keras.Model):
    def __init__(
        self,
        encoder,
        decoder,
        beta=1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

        self.total_loss_tracker = keras.metrics.Mean(
            name="total_loss"
        )

        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )

        self.kl_loss_tracker = keras.metrics.Mean(
            name="kl_loss"
        )

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(
                data,
                training=True
            )

            reconstruction = self.decoder(
                z,
                training=True
            )

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(
                        data - reconstruction
                    ),
                    axis=(1, 2, 3)
                )
            )

            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1
                    + z_log_var
                    - tf.square(z_mean)
                    - tf.exp(z_log_var),
                    axis=1
                )
            )

            total_loss = (
                reconstruction_loss
                + self.beta * kl_loss
            )

        gradients = tape.gradient(
            total_loss,
            self.trainable_weights
        )

        self.optimizer.apply_gradients(
            zip(
                gradients,
                self.trainable_weights
            )
        )

        self.total_loss_tracker.update_state(
            total_loss
        )

        self.reconstruction_loss_tracker.update_state(
            reconstruction_loss
        )

        self.kl_loss_tracker.update_state(
            kl_loss
        )

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss":
                self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }


# ## 8.4 Baseline VAE Compilation and Training
# 
# The Baseline VAE was created using the previously defined Encoder and Decoder.
# 
# The model was trained with the following settings:
# 
# | Parameter | Value | Reason |
# |---|---:|---|
# | **Beta (β)** | `1.0` | Uses the standard VAE balance between reconstruction loss and KL divergence loss. |
# | **Optimizer** | Adam | Provides adaptive learning rates and works well for deep neural-network training. |
# | **Learning Rate** | `0.0005` | Allows the model to update its weights gradually without making excessively large changes. |
# | **Epochs** | `35` | Provides enough training time for the baseline model while keeping it suitable as a reference model. |
# | **Batch Size** | `128` | Provides a balance between stable gradient updates and training efficiency. |
# | **Shuffle** | `True` | Randomises the training-image order each epoch to reduce dependence on the original data ordering. |
# 
# ### Why β = 1.0?
# 
# The Baseline VAE uses:
# 
# \[
# \beta = 1.0
# \]
# 
# This represents the standard VAE objective:
# 
# \[
# L_{total} = L_{reconstruction} + L_{KL}
# \]
# 
# Using `β = 1.0` gives the KL divergence its normal weighting and provides a suitable baseline before experimenting with different values in the Improved VAE.
# 
# ### Why Adam?
# 
# The **Adam optimizer** was selected because it automatically adjusts the learning rate for individual model parameters during training.
# 
# This makes it suitable for training the Encoder and Decoder together while handling the different gradients produced by the reconstruction and KL losses.
# 
# ### Why a Learning Rate of 0.0005?
# 
# A learning rate of `0.0005` was selected to allow the model to learn at a reasonable speed while avoiding very large weight updates that could make training unstable.
# 
# This value also provides a baseline that can later be adjusted when improving the VAE.
# 
# ### Why 35 Epochs?
# 
# The Baseline VAE was trained for **35 epochs** to establish an initial level of performance without making the baseline unnecessarily complex or expensive to train.
# 
# Its performance after 35 epochs can then be compared against the Improved VAE, which can use additional training or architectural changes.
# 
# ### Why a Batch Size of 128?
# 
# A batch size of `128` was used because it provides reasonably stable gradient estimates while remaining computationally efficient.
# 
# It is also consistent with the batch size used for the DCGAN experiments, helping to keep the experimental setup more consistent across the generative models.
# 
# ### Training Process
# 
# During each epoch, the VAE:
# 
# `Input Image → Encoder → Sampling → Decoder → Reconstruction → Loss Calculation → Weight Update`
# 
# The training history stores:
# 
# - Total loss
# - Reconstruction loss
# - KL divergence loss
# 
# These values are later plotted to evaluate whether the Baseline VAE learned steadily and whether the reconstruction and latent-space objectives remained balanced.

# In[52]:


baseline_vae = VAE(
    encoder=baseline_vae_encoder,
    decoder=baseline_vae_decoder,
    beta=1.0,
    name="baseline_vae"
)

baseline_vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0005
    )
)

baseline_vae_history = baseline_vae.fit(
    x_train,
    epochs=35,
    batch_size=128,
    shuffle=True
)


# ## Images Generated by the Baseline VAE
# 
# The existing image-display function can be reused because the VAE decoder accepts a random latent vector and produces images in the same format as the DCGAN generators.
# 

# In[53]:


display_generated_images(
    baseline_vae_decoder,
    latent_dim,
    classifier=classifier,
    num_images=25,
    title="Baseline VAE Generated Images"
)


# ### Generated Image and Class Prediction Evaluation
# 
# The Baseline VAE generated images were visually inspected and passed through the CIFAR-10 classifier to estimate which classes the outputs most closely resembled.
# 
# Within the sample of **25 generated images**, the predicted classes were:
# 
# | Predicted Class | Number of Images | Percentage |
# |---|---:|---:|
# | **Cat** | 12 | 48% |
# | **Dog** | 5 | 20% |
# | **Ship** | 4 | 16% |
# | **Deer** | 2 | 8% |
# | **Frog** | 1 | 4% |
# | **Bird** | 1 | 4% |
# | **Airplane** | 0 | 0% |
# | **Automobile** | 0 | 0% |
# | **Horse** | 0 | 0% |
# | **Truck** | 0 | 0% |
# | **Total** | **25** | **100%** |
# 
# The predictions are concentrated mainly in the **Cat class (48%)**, followed by **Dog (20%)** and **Ship (16%)**. Only six of the ten CIFAR-10 classes appeared within this sample.
# 
# Several images received relatively high classifier confidence scores, including:
# 
# - **Ship — 96.0%**
# - **Ship — 95.3%**
# - **Ship — 94.6%**
# - **Cat — 92.9%**
# - **Cat — 87.8%**
# - **Dog — 81.3%**
# 
# However, many of these images remain heavily blurred when inspected visually. This shows that a high classifier confidence score does not necessarily mean that the generated object is clearly recognisable to a human.
# 
# The classifier may be detecting particular **colours, textures or broad shapes** associated with a CIFAR-10 class even when the complete object structure is unclear.
# 
# The strong concentration of predictions in a small number of classes also suggests that the Baseline VAE has **limited output diversity** within this sample.
# 
# ---
# 
# ### Eye-Power Visual Evaluation
# 
# The same 25 generated images were manually evaluated using the **Clear / Marginal / Nonsense eye-power method**.
# 
# | Category | Number of Images | Percentage | Observation |
# |---|---:|---:|---|
# | **Clear** | 3 | 12% | A recognisable object or structure could be identified with reasonable confidence. |
# | **Marginal** | 16 | 64% | Some meaningful colours, shapes or object-like features were visible, but the images remained heavily blurred or ambiguous. |
# | **Nonsense** | 6 | 24% | No sufficiently recognisable or coherent object could be identified. |
# | **Total** | **25** | **100%** | |
# 
# Most outputs were classified as **Marginal (64%)**, showing that the Baseline VAE learned general colours and object-like patterns but struggled to generate clearly defined structures.
# 
# Only **3 images (12%)** were considered Clear, while **6 images (24%)** were classified as Nonsense.
# 
# Overall, the results suggest that the Baseline VAE learned some meaningful CIFAR-10 characteristics but produced images that were generally **blurry, weakly structured and concentrated around a limited number of predicted classes**.
# 
# This is consistent with its relatively high **FID score of 190.08** and lower **Inception Score of 2.40 ± 0.13**.

# ## Baseline VAE Training Loss Graph
# 

# In[54]:


plt.figure(figsize=(10, 5))

plt.plot(
    baseline_vae_history.history["total_loss"],
    label="Total Loss"
)

plt.plot(
    baseline_vae_history.history["reconstruction_loss"],
    label="Reconstruction Loss"
)

plt.plot(
    baseline_vae_history.history["kl_loss"],
    label="KL Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline VAE Training Loss")
plt.legend()
plt.grid()
plt.show()


# ## Evaluation
# 
# ### Total Loss
# 
# The Total Loss decreased sharply during the first few epochs, falling from approximately **465** to around **320**, before decreasing more gradually to approximately **275** by the end of training.
# 
# **Observation:**
# 
# - The large decrease during the early epochs shows that the Baseline VAE learned the main image patterns quickly.
# - After approximately 5–10 epochs, the decrease became much slower.
# - The gradual flattening of the curve suggests that the model was approaching convergence.
# - No large fluctuations occurred during the later epochs, indicating relatively stable VAE training.
# 
# ### Reconstruction Loss
# 
# The Reconstruction Loss started at approximately **435** and decreased rapidly to below **300** during the first few epochs. It then gradually decreased to approximately **250–255** towards the end of training.
# 
# **Observation:**
# 
# - The reduction in Reconstruction Loss shows that the Decoder became better at reconstructing CIFAR-10 images from the latent representation.
# - Most of the improvement occurred during the early stages of training.
# - The curve gradually flattened, suggesting that further epochs produced smaller improvements.
# - Reconstruction Loss remained the largest component of the Total Loss throughout training.
# 
# ### KL Loss
# 
# The KL Loss initially increased from approximately **25** to around **35**, before decreasing and stabilising at approximately **15–17**.
# 
# **Observation:**
# 
# - The early increase occurred while the Encoder was adjusting the structure of the latent space.
# - The later decrease shows that the learned latent distribution became more consistent with the desired normal distribution.
# - The KL Loss remained relatively stable during the later epochs.
# - This suggests that the model maintained latent-space regularisation without causing unstable training.
# 
# ### Overall Evaluation
# 
# The Baseline VAE shows a **stable learning process** across the training epochs.
# 
# Both the **Total Loss and Reconstruction Loss decreased substantially**, showing that the model improved its ability to reconstruct CIFAR-10 images. The **KL Loss also stabilised**, indicating that the latent representation became reasonably organised.
# 
# However, the loss curves mainly show that the model successfully optimised its training objectives. They do not directly indicate whether newly generated images are sharp or recognisable.
# 
# Therefore, the training-loss results should be considered together with the **generated image samples, eye-power evaluation, FID score of 190.08 and Inception Score of 2.40 ± 0.13** when evaluating the overall performance of the Baseline VAE.

# ## Baseline VAE: Predicted Class Distribution
# 

# In[56]:


# Number of generated images to analyse
num_samples = 1000

# Sample random latent vectors
z_samples = tf.random.normal(
    shape=(num_samples, latent_dim)
)

# Generate images using the Baseline VAE Decoder
generated_images = baseline_vae_decoder(
    z_samples,
    training=False
).numpy()

# Convert generated images from [-1, 1] to [0, 255]
# to match the input range used to train the CIFAR-10 classifier
classifier_images = (
    (generated_images + 1.0) / 2.0
) * 255.0

classifier_images = np.clip(
    classifier_images,
    0.0,
    255.0
)

# Predict CIFAR-10 classes
predictions = classifier.predict(
    classifier_images,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# Count predicted classes
class_counts = np.bincount(
    predicted_classes,
    minlength=10
)

# Plot predicted class distribution
plt.figure(figsize=(10, 5))

plt.bar(
    class_names,
    class_counts
)

plt.xlabel("Predicted CIFAR-10 Class")
plt.ylabel("Number of Generated Images")
plt.title(
    "Baseline VAE: Predicted Class Distribution"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Predicted Class Distribution of Generated Images
# 
# A total of **1,000 images** were generated by the Baseline VAE and passed through the CIFAR-10 classifier to examine the diversity of the generated outputs.
# 
# The predicted class distribution is highly uneven. The classifier predicted the large majority of generated images as **Cat**, while **Truck** was the next most common prediction. Only a small number of images were predicted as classes such as **Airplane, Bird, Dog, Frog and Ship**, while some classes were rarely or never predicted.
# 
# ### Evaluation
# 
# The strong concentration in the **Cat** class suggests that the Baseline VAE has **limited output diversity** when new latent vectors are sampled for image generation.
# 
# Although the VAE is trained using images from all ten CIFAR-10 classes, its generated outputs appear to contain visual features that the classifier repeatedly associates with only a small number of classes.
# 
# This result is consistent with the generated-image grid, where many outputs contain similar blurry colours, shapes and textures rather than clearly different object structures.
# 
# The distribution also supports the relatively low **Inception Score of 2.40 ± 0.13**, since Inception Score considers both recognisability and diversity. Generating many outputs that resemble the same predicted class limits the diversity component of the score.
# 
# Overall, the Baseline VAE appears to have learned some general CIFAR-10 characteristics, but it struggles to generate a **wide and balanced variety of recognisable objects**. This limitation is also reflected in its **FID score of 190.08** and the large number of Marginal and Nonsense images identified during the eye-power evaluation.

# ### Observation
# 
# The top row shows the original CIFAR-10 images, while the bottom row shows the VAE reconstructions.
# 
# The reconstructed images preserve the main colours and general object shapes, but they are noticeably blurred and lack fine details. Some objects remain partly recognisable, while others are difficult to identify.
# 
# This suggests that the baseline VAE has learned the overall image structure, but its compressed latent representation is not strong enough to reconstruct clear and detailed CIFAR-10 images.

# ## Baseline VAE FID Score
# 

# In[57]:


baseline_vae_fid = fid_evaluator.evaluate(
    generator=baseline_vae_decoder,
    latent_dim=latent_dim,
    model_name="Baseline VAE",
    generated_image_range="-1_to_1"
)


# ## 8.7 Baseline VAE FID Evaluation
# 
# The Baseline VAE achieved an **FID score of 256.83** using **1,000 real CIFAR-10 images and 1,000 generated images**.
# 
# FID measures the similarity between the feature distribution of real images and generated images.
# 
# A **lower FID score is better**, because it indicates that the generated images are closer to the real CIFAR-10 distribution.
# 
# ### Result
# 
# | Model | FID Score |
# |---|---:|
# | **Baseline VAE** | **256.83** |
# 
# The relatively high FID score of **256.83** suggests that the Baseline VAE generated images that were still considerably different from real CIFAR-10 images.
# 
# This is consistent with the visual results, where many generated images appeared heavily blurred and lacked clear object structures.
# 
# The result may also reflect the trade-off in VAE training between:
# 
# - reconstructing the input images accurately
# - maintaining a structured latent distribution through KL divergence
# 
# Although the Baseline VAE was able to learn general colours and object-like patterns from CIFAR-10, the high FID score indicates that the generated-image distribution remained far from the real-image distribution.
# 
# This score will be used as a benchmark when evaluating the Improved VAE. A **lower FID score** in the Improved VAE would indicate that the proposed improvements produced generated images that are more similar to real CIFAR-10 images.

# ## Baseline VAE Inception Score
# 

# In[59]:


baseline_vae_is_mean, baseline_vae_is_std = (
    is_evaluator.evaluate(
        generator=baseline_vae_decoder,
        latent_dim=latent_dim,
        model_name="Baseline VAE",
        generated_image_range="-1_to_1"
    )
)


# ## 8.8 Baseline VAE Inception Score Evaluation
# 
# The Baseline VAE achieved an **Inception Score of 1.87 ± 0.06** using **1,000 generated CIFAR-10 images**.
# 
# The Inception Score evaluates two main aspects of generated images:
# 
# - **Recognisability** – whether the generated images contain features that can be associated with meaningful object classes.
# - **Diversity** – whether the model generates a variety of different image types rather than very similar outputs.
# 
# A **higher Inception Score is better**.
# 
# ### Result
# 
# | Model | Inception Score |
# |---|---:|
# | **Baseline VAE** | **1.87 ± 0.06** |
# 
# The relatively low Inception Score of **1.87 ± 0.06** suggests that the Baseline VAE generated images with limited recognisability and diversity.
# 
# This result is consistent with the **FID score of 256.83**, which also indicates that the generated-image distribution remains considerably different from the real CIFAR-10 distribution.
# 
# Compared with the Baseline DCGAN, which achieved an Inception Score of **3.61 ± 0.13**, the Baseline VAE performed worse in terms of generated-image recognisability and diversity.
# 
# This is also consistent with the generated-image samples and predicted class distribution, where many outputs appeared blurry and a large proportion of generated images were predicted as the same class.
# 
# The low score may reflect the trade-off in VAE training between:
# 
# - reconstructing images accurately
# - maintaining a structured latent space through KL-divergence regularisation
# 
# This can result in smoother and less detailed generated images.
# 
# The Baseline VAE results will therefore be used as a benchmark for evaluating whether the Improved VAE can generate **clearer, more realistic and more diverse images**.

# # 9. Model 4: Improved VAE
# This version keeps the same **100-dimensional latent space** and `tanh` image range used by the baseline VAE, while addressing two issues found in the first improved model:
# 
# 1. A very low KL weight improved reconstruction but weakened the match between the learned latent distribution and the standard normal distribution used for generation.
# 2. Pixel-only mean squared error encouraged smooth reconstructions and did not directly preserve image structure.
# 
# The revised model uses residual convolutional blocks, a combined pixel-and-structure reconstruction loss, deterministic reconstruction during inference, and a moderate KL weight.
# 

# ## Changes Made in Version 2 and Why They Were Made
# 
# Version 2 was designed to address the main weaknesses observed in the earlier VAE, especially **blurry generated images, weak object structure, limited diversity and poor FID/Inception Score results**.
# 
# The latent dimension remains fixed at **100**, so any improvement is mainly caused by the architectural and training changes rather than giving the model a larger latent space.
# 
# | Change | Why It Was Changed | Expected Benefit |
# |---|---|---|
# | **Residual Encoder blocks** | The earlier VAE may lose useful image information as features pass through multiple convolutional layers. Residual connections allow information and gradients to pass more directly through the network. | Helps preserve useful visual features and supports deeper feature extraction. |
# | **Residual Decoder blocks** | The earlier generated images were blurry and lacked clear object structures. More processing is therefore added at each image scale before producing the final RGB image. | Gives the Decoder more opportunity to refine shapes, textures and object structures. |
# | **MSE + SSIM reconstruction loss** | MSE focuses mainly on pixel-by-pixel differences and may still produce visually blurry images. SSIM is added because it also considers local structure, contrast and similarity between image regions. | Encourages both accurate pixel values and stronger visual structure. |
# | **Deterministic reconstruction using `z_mean`** | Random sampling can introduce additional variation when reconstructing the same image. Using `z_mean` removes this randomness during reconstruction evaluation. | Produces more consistent reconstructions and makes comparison with the original image clearer. |
# | **Moderate KL target of `0.3`** | A strong KL weight can force the latent distribution towards the normal distribution too aggressively, which may reduce reconstruction quality. | Maintains latent-space regularisation while allowing more focus on image reconstruction. |
# | **40-epoch KL warm-up** | Applying the full KL effect from the beginning may make it harder for the model to first learn useful image features. | Allows the model to focus more on reconstruction early in training before stronger latent regularisation is introduced. |
# | **Adaptive learning rate** | A learning rate that works well during early training may become too large when the model approaches convergence. | Reduces the learning rate when improvement slows, allowing smaller and more precise updates. |
# | **100 epochs** | The deeper model requires enough training time to learn and refine its additional layers. | Gives the model sufficient time to converge and also keeps the training duration consistent with the previous improved experiment. |
# 
# ### Why Residual Connections Were Added
# 
# Residual connections were introduced because deeper networks can sometimes lose useful information or become more difficult to optimise.
# 
# Instead of forcing every feature to pass only through several transformations, a residual connection provides an additional path for information to move through the network.
# 
# This is especially useful for the VAE because the Encoder needs to preserve important CIFAR-10 features during compression, while the Decoder needs to recover useful structures when reconstructing or generating images.
# 
# ---
# 
# ### Why MSE and SSIM Were Combined
# 
# The earlier VAE produced images that contained general colours and shapes but remained heavily blurred.
# 
# **MSE** measures the difference between individual pixel values, which helps the reconstructed image remain numerically close to the original image.
# 
# However, two images can have similar pixel values while still differing in visible structure.
# 
# **SSIM** therefore provides an additional measure that considers local image structure, contrast and patterns.
# 
# By combining MSE and SSIM, Version 2 attempts to improve both:
# 
# - **Pixel-level accuracy**
# - **Perceptual image structure**
# 
# ---
# 
# ### Why `z_mean` Is Used for Reconstruction
# 
# During normal VAE training, a random latent vector is sampled using `z_mean`, `z_log_var` and random noise.
# 
# This randomness is necessary because the VAE needs to learn a continuous latent distribution.
# 
# However, when evaluating reconstruction quality, this randomness can make the reconstructed image slightly different each time.
# 
# Using `z_mean` directly provides a deterministic latent representation, allowing reconstruction quality to be evaluated more consistently.
# 
# This is used only for **reconstruction comparison**. Random latent sampling is still used when evaluating the model's actual image-generation ability.
# 
# ---
# 
# ### Why the KL Target Was Reduced to 0.3
# 
# The VAE needs KL divergence to keep the latent space organised and close to a normal distribution.
# 
# However, applying a full KL weight of `1.0` may cause the model to focus too strongly on organising the latent space instead of preserving detailed image information.
# 
# A moderate KL target of **0.3** was therefore selected to create a better balance between:
# 
# - Reconstruction quality
# - Latent-space regularisation
# 
# The goal is not to remove KL divergence, but to prevent it from overpowering the reconstruction objective.
# 
# ---
# 
# ### Why KL Warm-Up Was Added
# 
# Instead of immediately applying the full KL target, its influence is gradually increased over the first **40 epochs**.
# 
# During the early stages of training, the model therefore focuses more strongly on learning how to reconstruct CIFAR-10 images.
# 
# As training progresses, the KL contribution increases and begins to organise the latent space more strongly.
# 
# This gradual approach aims to prevent strong regularisation from disrupting reconstruction learning too early.
# 
# ---
# 
# ### Why an Adaptive Learning Rate Was Added
# 
# Large learning-rate updates are useful during the earlier stages of training because the model needs to learn quickly.
# 
# However, once the loss begins to stabilise, large updates may cause the model to move around the minimum instead of refining its solution.
# 
# The adaptive learning-rate strategy therefore reduces the learning rate when the Total Loss stops improving.
# 
# This allows smaller parameter updates during the later stages of training and may help the model achieve a more refined solution.
# 
# ---
# 
# ### Why 100 Epochs Were Used
# 
# Version 2 contains a deeper architecture and additional residual processing, so sufficient training time is required for the model to learn these additional parameters.
# 
# Training for **100 epochs** also keeps the experiment consistent with the previous improved model, allowing the architectural and training changes to be compared more fairly.
# 
# ---
# 
# ### Overall Reason for Version 2
# 
# The main objective of Version 2 is to improve the balance between **image quality and latent-space structure**.
# 
# The changes can be grouped into three areas:
# 
# 1. **Architecture improvements**  
#    Residual Encoder and Decoder blocks increase the model's ability to preserve and refine visual information.
# 
# 2. **Loss improvements**  
#    MSE + SSIM focuses on both pixel accuracy and image structure, while the moderate KL target and KL warm-up prevent latent regularisation from becoming too strong too early.
# 
# 3. **Training improvements**  
#    Adaptive learning rate and 100 training epochs provide more opportunity for the deeper model to converge and refine its outputs.
# 
# Together, these changes aim to produce **clearer, more structured and more diverse CIFAR-10 images while maintaining a useful continuous latent space**.

# ### Sampling Layer
# 
# The Sampling layer performs the **reparameterisation trick** used by the VAE.
# 
# The Encoder does not directly produce a single latent vector. Instead, it produces:
# 
# - `z_mean` – the mean of the latent distribution
# - `z_log_var` – the logarithm of the variance of the latent distribution
# 
# A random value `epsilon` is sampled from a standard normal distribution and combined with the learned mean and variance to produce the latent vector `z`.
# 
# This allows the VAE to introduce randomness into the latent space while still allowing gradients to pass through the network during training.
# 
# The sampling process can be represented as:
# 
# `z = z_mean + exp(0.5 × z_log_var) × epsilon`
# 
# This is important because it allows the VAE to learn a **continuous and structured latent space**, where new latent vectors can later be sampled to generate new CIFAR-10 images.

# In[60]:


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]

        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# ## Improved VAE Encoder
# 
# The Improved VAE Encoder compresses each **32 × 32 × 3 CIFAR-10 image** into a latent representation with a fixed latent dimension of **100**.
# 
# The Encoder gradually reduces the spatial dimensions of the image while increasing the number of feature maps. This allows the model to move from learning simple visual features to more complex object structures.
# 
# | Component / Parameter | Setting Used | Why It Was Used |
# |---|---|---|
# | **Input Shape** | `32 × 32 × 3` | Matches the original dimensions of CIFAR-10 RGB images. |
# | **First Conv2D Layer** | 64 filters, `3 × 3` kernel, stride `2` | Reduces the image from **32 × 32 to 16 × 16** while learning basic features such as edges, colours and textures. |
# | **Second Conv2D Layer** | 128 filters, `3 × 3` kernel, stride `2` | Reduces the feature maps from **16 × 16 to 8 × 8** and allows the Encoder to learn more complex patterns. |
# | **Third Conv2D Layer** | 256 filters, `3 × 3` kernel, stride `2` | Reduces the feature maps from **8 × 8 to 4 × 4** while increasing the capacity to learn higher-level object structures. |
# | **Increasing Filters** | `64 → 128 → 256` | As spatial dimensions become smaller, more filters are used so that the network can represent a larger variety of increasingly complex visual features. |
# | **Kernel Size** | `3 × 3` | Provides a good balance between capturing local visual patterns and keeping the number of parameters manageable. |
# | **Stride** | `2` | Performs downsampling directly within the convolutional layers, reducing the image dimensions gradually without requiring separate pooling layers. |
# | **Padding** | `same` | Maintains predictable feature-map dimensions so that each stride-2 layer approximately halves the image resolution. |
# | **LeakyReLU** | Negative slope = `0.2` | Allows a small gradient for negative values and helps maintain gradient flow during training. |
# | **Flatten Layer** | Converts `4 × 4 × 256` into a 1D vector | Prepares the extracted convolutional features for the Dense layers used to construct the latent representation. |
# | **Dense Layer** | 256 neurons with ReLU | Combines the extracted image features before estimating the parameters of the latent distribution. |
# | **Latent Dimension** | `100` | Provides sufficient capacity to represent different CIFAR-10 visual characteristics while keeping the latent space manageable. It is also kept constant across the VAE experiments for a fair comparison. |
# | **`z_mean`** | 100 values | Represents the mean of the learned latent probability distribution. |
# | **`z_log_var`** | 100 values | Represents the logarithm of the variance of the learned latent probability distribution. |
# | **Sampling Layer** | Reparameterisation trick | Samples a latent vector from the learned distribution while still allowing gradients to flow through the Encoder during training. |
# 
# 
# 
# ### Why the Improved Encoder Is Deeper
# 
# The Improved VAE uses **three convolutional stages** so that the Encoder can learn visual information at different levels.
# 
# The earlier layers focus on simpler characteristics such as **edges, colours and textures**, while deeper layers can capture more complex information such as **object shapes and larger structures**.
# 
# Increasing the number of filters from **64 to 128 to 256** also gives the deeper layers greater capacity to represent these more complex patterns.
# 
# This deeper feature extraction is intended to create a more informative latent representation, which can help the Decoder reconstruct and generate more detailed CIFAR-10 images.
# 
# ### Why the Latent Dimension Remains at 100
# 
# The latent dimension remains fixed at **100** rather than being increased for the Improved VAE.
# 
# Keeping the same latent dimension ensures that improvements are mainly caused by the **Encoder, Decoder and training changes**, rather than simply giving the improved model more latent-space capacity.
# 
# This supports a fairer comparison between the Baseline and Improved VAE models.

# In[61]:


latent_dim = 100

improved_vae_encoder_inputs = keras.Input(
    shape=(32, 32, 3)
)

x = layers.Conv2D(
    64, 3,
    strides=2,
    padding="same"
)(improved_vae_encoder_inputs)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Conv2D(
    128, 3,
    strides=2,
    padding="same"
)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Conv2D(
    256, 3,
    strides=2,
    padding="same"
)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Flatten()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

improved_z_mean = layers.Dense(
    latent_dim,
    name="improved_z_mean"
)(x)

improved_z_log_var = layers.Dense(
    latent_dim,
    name="improved_z_log_var"
)(x)

improved_z = Sampling()(
    [improved_z_mean, improved_z_log_var]
)

improved_vae_encoder = keras.Model(
    improved_vae_encoder_inputs,
    [
        improved_z_mean,
        improved_z_log_var,
        improved_z
    ],
    name="improved_vae_encoder"
)


# ## Improved VAE Decoder
# 
# The Improved VAE Decoder converts the **100-dimensional latent vector** back into a **32 × 32 × 3 RGB image**.
# 
# The Decoder performs the opposite operation of the Encoder. Instead of gradually reducing the image dimensions, it progressively increases the spatial resolution from **4 × 4 to 32 × 32** while reducing the number of feature maps from **256 to 64**.
# 
# | Component / Parameter | Setting Used | Why It Was Used |
# |---|---|---|
# | **Latent Input** | `100` dimensions | Matches the latent representation produced by the Encoder and keeps the latent size consistent across experiments. |
# | **Dense Layer** | `4 × 4 × 256` neurons | Expands the compact latent vector into enough values to form the initial spatial feature map used for image reconstruction. |
# | **Reshape Layer** | `4 × 4 × 256` | Converts the 1D Dense output into a 3D feature map that can be processed by convolutional layers. |
# | **First Conv2DTranspose** | 256 filters, `4 × 4`, stride `2` | Upsamples the feature map from **4 × 4 to 8 × 8** while maintaining a high number of feature channels for complex image information. |
# | **Second Conv2DTranspose** | 128 filters, `4 × 4`, stride `2` | Upsamples the feature map from **8 × 8 to 16 × 16** while gradually converting abstract latent features into more visible image structures. |
# | **Third Conv2DTranspose** | 64 filters, `4 × 4`, stride `2` | Upsamples from **16 × 16 to 32 × 32**, bringing the output back to the original CIFAR-10 image resolution. |
# | **Decreasing Filters** | `256 → 128 → 64` | Early Decoder layers need many channels to represent complex latent features. As spatial resolution increases, fewer filters are needed to refine these features into visible image details. |
# | **Kernel Size** | `4 × 4` | Works effectively with stride `2` for progressive upsampling and allows each layer to use information from a wider local region when constructing the image. |
# | **Stride** | `2` | Doubles the spatial resolution at each stage, allowing the Decoder to progress smoothly from `4 × 4` to `32 × 32`. |
# | **Padding** | `same` | Keeps the upsampling dimensions predictable and ensures the required image sizes are produced at each stage. |
# | **LeakyReLU** | Negative slope = `0.2` | Keeps a small gradient for negative values, which helps maintain gradient flow and avoids completely inactive neurons during training. |
# | **Final Conv2D Layer** | 3 filters, `3 × 3` | Produces the final three channels required for an RGB CIFAR-10 image. |
# | **Final Activation** | `tanh` | Produces output values between **-1 and 1**, matching the normalisation applied to the CIFAR-10 training images. |
# 
# 
# 
# ### Why Three Upsampling Stages Were Used
# 
# The Decoder starts with a small **4 × 4 feature map**, so the spatial resolution needs to be increased until it reaches the original CIFAR-10 size of **32 × 32**.
# 
# Using three stride-2 transposed convolution layers gives the following progression:
# 
# `4 × 4 → 8 × 8 → 16 × 16 → 32 × 32`
# 
# This gradual upsampling allows the Decoder to reconstruct the image progressively instead of attempting to directly transform the latent vector into a full-resolution image.
# 
# Earlier layers focus more on the overall image representation, while later layers progressively introduce finer spatial information.
# 
# ---
# 
# ### Why the Number of Filters Decreases
# 
# The Decoder uses:
# 
# `256 → 128 → 64`
# 
# At the beginning of decoding, the spatial feature map is very small, so a larger number of filters is used to represent different high-level features stored in the latent vector.
# 
# As the image becomes larger, these abstract features are gradually converted into visible shapes, colours and textures.
# 
# Therefore, the number of channels can be reduced as the spatial resolution increases.
# 
# This also mirrors the general structure of the Encoder, which performs the opposite process:
# 
# **Encoder:**
# 
# `64 → 128 → 256`
# 
# **Decoder:**
# 
# `256 → 128 → 64`
# 
# This creates a relatively symmetrical Encoder-Decoder structure.
# 
# ---
# 
# ### Why Conv2DTranspose Was Used
# 
# `Conv2DTranspose` was selected because the Decoder needs to both **increase image resolution and learn how the new spatial features should be constructed**.
# 
# Unlike simple resizing, transposed convolution contains trainable filters.
# 
# This means that the model learns how latent features should be expanded into larger spatial representations during training.
# 
# By using a stride of `2`, each layer doubles the image dimensions until the required **32 × 32** output size is reached.
# 
# ---
# 
# ### Why a 4 × 4 Kernel Was Used for Upsampling
# 
# A `4 × 4` kernel was used together with a stride of `2`.
# 
# The larger kernel gives each output region access to information from a wider neighbouring area of the previous feature map.
# 
# This is useful during generation because object structures in CIFAR-10 depend on relationships between nearby colours, shapes and textures.
# 
# The combination of:
# 
# - `kernel_size = 4`
# - `strides = 2`
# - `padding = "same"`
# 
# also provides a simple and predictable doubling of the spatial resolution at each Decoder stage.
# 
# ---
# 
# ### Why LeakyReLU Was Used
# 
# LeakyReLU with a negative slope of **0.2** was used after each transposed convolution.
# 
# A normal ReLU outputs zero for all negative inputs. This can cause some neurons to stop contributing to learning if they repeatedly receive negative values.
# 
# LeakyReLU keeps a small portion of the negative values instead.
# 
# This helps maintain gradient flow throughout the Decoder and supports more stable learning while the model reconstructs complex image features.
# 
# 
# ---
# 
# ### Overall Decoder Design
# 
# The Decoder was designed to progressively transform the compact latent representation into a full CIFAR-10 image.
# 
# The process can be summarised as:
# 
# `Latent Vector → High-Level Features → Larger Spatial Features → RGB Image`
# 
# The progressive upsampling structure gives the network multiple stages to transform abstract latent information into increasingly detailed image features.
# 
# Together with the deeper Encoder, this Decoder is intended to improve the model's ability to generate **more structured and recognisable CIFAR-10 images** while keeping the latent dimension fixed at **100** for a fair comparison.

# In[62]:


improved_vae_latent_inputs = keras.Input(
    shape=(latent_dim,)
)

x = layers.Dense(
    4 * 4 * 256,
    activation="relu"
)(improved_vae_latent_inputs)

x = layers.Reshape(
    (4, 4, 256)
)(x)

x = layers.Conv2DTranspose(
    256, 4,
    strides=2,
    padding="same"
)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Conv2DTranspose(
    128, 4,
    strides=2,
    padding="same"
)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

x = layers.Conv2DTranspose(
    64, 4,
    strides=2,
    padding="same"
)(x)
x = layers.LeakyReLU(negative_slope=0.2)(x)

improved_vae_decoder_outputs = layers.Conv2D(
    3,
    3,
    padding="same",
    activation="tanh"
)(x)

improved_vae_decoder = keras.Model(
    improved_vae_latent_inputs,
    improved_vae_decoder_outputs,
    name="improved_vae_decoder"
)


# ## Improved VAE Training Configuration
# 
# 
# 
# The VAE loss contains two main components:
# 
# - **Reconstruction Loss** – encourages the reconstructed image to remain similar to the original image.
# - **KL Divergence Loss** – encourages the learned latent distribution to remain close to a standard normal distribution.
# 
# If the KL contribution is too strong, the model may focus heavily on organising the latent space and lose useful image information, which can result in weaker or blurrier reconstructions.
# 
# Therefore, `beta = 0.5` was selected instead of the standard value of `1.0`.
# 
# This reduces the influence of KL divergence while still keeping the latent space regularised.
# 
# The aim is to achieve a better balance between:
# 
# `Image Reconstruction Quality ↔ Structured Latent Space`
# 
# This is particularly important for CIFAR-10 because the images contain varied colours, textures and object structures that need to be preserved during compression.
# 
# ---
# 
# ### Why Adam Was Used
# 
# The **Adam optimizer** was selected because it adapts the size of parameter updates during training.
# 
# Instead of applying exactly the same learning rate to every parameter, Adam keeps track of previous gradients and adjusts the updates accordingly.
# 
# This is useful for the VAE because the Encoder and Decoder are being optimised together while also balancing reconstruction loss and KL divergence.
# 
# Adam therefore provides a practical and stable optimisation method for this type of deep generative model.
# 
# ---
# 
# ### Why the Learning Rate Was Set to `0.0003`
# 
# A learning rate of **0.0003** was chosen to provide relatively small and controlled parameter updates.
# 
# If the learning rate is too high, the model may:
# 
# - overshoot useful solutions
# - produce unstable loss values
# - fail to converge properly
# 
# If it is too low, training may take unnecessarily long.
# 
# Therefore, `0.0003` provides a reasonable balance between **training speed and stability** for the Improved VAE.
# 
# ---
# 
# ### Why 100 Epochs Were Used
# 
# The Improved VAE contains a deeper feature-extraction and reconstruction process than the simpler Baseline VAE.
# 
# Training for **100 epochs** gives the model enough opportunities to repeatedly observe the CIFAR-10 training images and refine its learned parameters.
# 
# Using the same training duration as the other improved experiments also makes the comparison fairer because better performance cannot simply be attributed to giving one model substantially more training time.
# 
# ---
# 
# ### Why a Batch Size of 128 Was Used
# 
# A batch size of **128** means that the model processes 128 CIFAR-10 images before updating its parameters.
# 
# This provides a compromise between:
# 
# - computational efficiency
# - memory consumption
# - stability of the gradient estimates
# 
# A very small batch can produce noisy updates, while a very large batch requires more memory and may reduce the variation between updates.
# 
# Therefore, `128` was selected as a practical batch size for training the VAE.
# 
# ---
# 
# ### Why Training Data Was Shuffled
# 
# `shuffle=True` randomises the order of the CIFAR-10 training images at the beginning of each epoch.
# 
# This prevents the model from repeatedly seeing images in exactly the same sequence.
# 
# As a result, each batch contains a different mixture of images, which helps reduce the possibility of the model learning patterns caused by dataset ordering rather than the actual visual characteristics of CIFAR-10.
# 
# ---
# 
# ### Overall Training Strategy
# 
# The training configuration was designed to provide a balance between **stable optimisation, reconstruction quality and latent-space regularisation**.
# 
# The moderate KL weight of `0.5` prevents the KL term from becoming too dominant, while the Adam optimizer and learning rate of `0.0003` provide controlled parameter updates.
# 
# Training for **100 epochs with a batch size of 128 and shuffled data** gives the Improved VAE sufficient opportunity to learn the CIFAR-10 distribution while maintaining a consistent experimental setup for comparison with the other models.

# In[63]:


improved_vae = VAE(
    encoder=improved_vae_encoder,
    decoder=improved_vae_decoder,
    beta=0.5,
    name="improved_vae"
)

improved_vae.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.0003
    )
)

improved_vae_history = improved_vae.fit(
    x_train,
    epochs=100,
    batch_size=128,
    shuffle=True
)


# ## Images Generated by Improved VAE Version 2
# 
# Random vectors are sampled from the same 100-dimensional standard normal distribution used for the baseline and previous improved VAE experiments.
# 

# In[65]:


display_generated_images(
    improved_vae_decoder,
    latent_dim,
    num_images=25,
    classifier=classifier,
    title="Improved VAE Version 2 Generated Images"
)


# ## Improved VAE Generated Image Evaluation
# 
# A sample of **25 images** was generated using the Improved VAE and passed through the CIFAR-10 classifier.
# 
# The predicted classes within the sample were:
# 
# | Predicted Class | Number of Images | Percentage |
# |---|---:|---:|
# | **Dog** | 11 | 44% |
# | **Cat** | 7 | 28% |
# | **Deer** | 3 | 12% |
# | **Airplane** | 1 | 4% |
# | **Automobile** | 1 | 4% |
# | **Bird** | 1 | 4% |
# | **Horse** | 1 | 4% |
# | **Frog** | 0 | 0% |
# | **Ship** | 0 | 0% |
# | **Truck** | 0 | 0% |
# | **Total** | **25** | **100%** |
# 
# ### Observation
# 
# The generated images show more visible object-like structures compared with the Baseline VAE.
# 
# Several images also received relatively high classifier confidence scores, including:
# 
# - **Dog – 99.2%**
# - **Dog – 99.0%**
# - **Dog – 98.9%**
# - **Cat – 90.7%**
# - **Dog – 81.9%**
# - **Dog – 77.4%**
# 
# Although the images remain blurry, many contain stronger shapes, colour regions and object-like features that allow the classifier to identify them with relatively high confidence.
# 
# However, the sample is still dominated by **Dog and Cat predictions**, which together account for **72% of the 25 generated images**.
# 
# Therefore, the Improved VAE has improved the recognisability of the generated images, but some class imbalance and visual blurriness are still present.
# 
# It is also important to note that classifier confidence does not directly measure human visual quality. A classifier may recognise particular colours, textures or broad shapes even when the object is not completely clear to a human observer.

# ## Improved VAE Version 2 Training Loss
# 
# The first graph displays the combined reconstruction loss, pixel loss, KL loss and total loss. SSIM loss is shown separately because its numerical scale is much smaller than the other losses.
# 

# In[50]:


plt.figure(figsize=(10, 5))

plt.plot(
    improved_vae_history.history["total_loss"],
    label="Total Loss"
)

plt.plot(
    improved_vae_history.history["reconstruction_loss"],
    label="Reconstruction Loss"
)

plt.plot(
    improved_vae_history.history["kl_loss"],
    label="KL Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Improved VAE Training Loss")

plt.legend()
plt.grid()

plt.show()


# ## Improved VAE Training Loss Evaluation
# 
# The Improved VAE training loss shows a stable learning pattern across the **100 training epochs**.
# 
# ### Reconstruction Loss
# 
# The Reconstruction Loss decreased sharply during the early epochs and continued decreasing throughout training, from approximately **300** to around **80**.
# 
# **Observation:**
# 
# - The large reduction shows that the Decoder became progressively better at reconstructing the CIFAR-10 images.
# - Most of the improvement occurred during the earlier epochs.
# - The loss continued to decrease gradually until the end of training, suggesting that the model was still refining its reconstructions.
# - Compared with the Baseline VAE, the much lower final Reconstruction Loss suggests improved reconstruction capability.
# 
# ### KL Loss
# 
# The KL Loss increased from approximately **60** to around **90** during the early stages of training before gradually stabilising at approximately **95–100**.
# 
# **Observation:**
# 
# - The increasing KL Loss shows that the Encoder increasingly used the latent variables to represent image information.
# - Because the Improved VAE uses `beta = 0.5`, the KL component has a reduced contribution to the Total Loss compared with using the full weight of `1.0`.
# - The KL Loss eventually stabilised rather than increasing continuously, indicating that the balance between reconstruction and latent regularisation became relatively stable.
# 
# ### Total Loss
# 
# The Total Loss decreased strongly from approximately **335** at the beginning of training to around **128** by the final epoch.
# 
# **Observation:**
# 
# - The rapid early decrease shows that the model learned useful image representations quickly.
# - The Total Loss continued to decrease gradually throughout training.
# - No large fluctuations were observed, suggesting stable optimisation.
# - The smooth convergence indicates that the selected learning rate and training configuration were suitable for the Improved VAE.
# 
# ### Overall Evaluation
# 
# The training curves indicate that the Improved VAE successfully reduced its reconstruction error while maintaining KL regularisation.
# 
# The strong reduction in Reconstruction Loss is particularly important because one of the main objectives of the Improved VAE was to preserve more useful image information.
# 
# Although the KL Loss increased, the Total Loss continued to decrease because the reconstruction improvement was sufficiently large.
# 
# Overall, the loss curves suggest **stable convergence and improved reconstruction learning**, although generated-image quality must still be evaluated using visual inspection, FID, Inception Score and class diversity rather than training loss alone.

# ## Improved VAE: Predicted Class Distribution
# 

# In[66]:


# Improved VAE: Predicted Class Distribution

num_samples = 1000

# Generate random latent vectors
z_samples = tf.random.normal(
    shape=(num_samples, latent_dim)
)

# Generate images using Improved VAE decoder
generated_images = improved_vae_decoder(
    z_samples,
    training=False
).numpy()

# Predict CIFAR-10 classes
predictions = classifier.predict(
    generated_images,
    verbose=0
)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

# Count predictions for each class
class_counts = np.bincount(
    predicted_classes,
    minlength=10
)

# Print class counts and percentages
for i, class_name in enumerate(class_names):
    percentage = (
        class_counts[i] / num_samples
    ) * 100

    print(
        f"{class_name}: "
        f"{class_counts[i]} "
        f"({percentage:.1f}%)"
    )

# Plot class distribution
plt.figure(figsize=(10, 5))

plt.bar(
    class_names,
    class_counts
)

plt.xlabel("Predicted CIFAR-10 Class")
plt.ylabel("Number of Generated Images")
plt.title(
    "Improved VAE: Predicted Class Distribution"
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Improved VAE: Predicted Class Distribution
# 
# A total of **1,000 images** were generated by the Improved VAE and classified using the CIFAR-10 classifier.
# 
# ### Result
# 
# | Predicted Class | Images | Percentage |
# |---|---:|---:|
# | Airplane | 22 | 2.2% |
# | Automobile | 36 | 3.6% |
# | Bird | 76 | 7.6% |
# | **Cat** | **334** | **33.4%** |
# | Deer | 50 | 5.0% |
# | **Dog** | **348** | **34.8%** |
# | Frog | 12 | 1.2% |
# | Horse | 35 | 3.5% |
# | Ship | 59 | 5.9% |
# | Truck | 28 | 2.8% |
# | **Total** | **1000** | **100%** |
# 
# ### Evaluation
# 
# The Improved VAE generated images that were classified into **all ten CIFAR-10 classes**, showing greater output diversity than the Baseline VAE.
# 
# However, the distribution remains uneven.
# 
# The two dominant predicted classes were:
# 
# - **Dog – 348 images (34.8%)**
# - **Cat – 334 images (33.4%)**
# 
# Together, these classes account for approximately **68.2% of all generated images**.
# 
# Other classes such as Bird, Ship and Deer were also represented, while Frog appeared least frequently at only **1.2%**.
# 
# ### Why This Is an Improvement
# 
# The Baseline VAE class distribution was extremely concentrated around a small number of predicted classes, particularly Cat.
# 
# In contrast, the Improved VAE produces predictions across **all ten CIFAR-10 classes** and is considerably less concentrated around a single class.
# 
# This suggests that the architectural and training improvements helped the model learn a broader range of visual characteristics from CIFAR-10.
# 
# The distribution is still not balanced, especially because Dog and Cat dominate the predictions. However, an exactly balanced class distribution is not required because the VAE is an **unconditional generative model** and is not explicitly instructed to generate equal numbers of each class.
# 
# Therefore, this graph should mainly be interpreted as evidence of **improved output diversity rather than classification accuracy**.
# 
# Overall, the Improved VAE demonstrates substantially better diversity than the Baseline VAE, although further improvements would be required to produce a more evenly distributed and visually distinct range of CIFAR-10 objects.

# ## Improved VAE FID Score
# 
# The same FID evaluator, number of images, real-image sample and random noise seed are reused so that the baseline and improved VAE scores are directly comparable.
# 
# A **lower FID score** indicates that the generated-image distribution is closer to the real CIFAR-10 distribution.

# In[67]:


improved_vae_fid = fid_evaluator.evaluate(
    generator=improved_vae_decoder,
    latent_dim=latent_dim,
    model_name="Improved VAE Version 2",
    generated_image_range="-1_to_1"
)


# ## Improved VAE Version 2 FID Evaluation
# 
# The Improved VAE Version 2 achieved an **FID score of 155.54** using **1,000 real CIFAR-10 images and 1,000 generated images**.
# 
# A lower FID score indicates that the generated-image distribution is more similar to the real-image distribution.
# 
# ### Result
# 
# | Model | FID Score |
# |---|---:|
# | Baseline VAE | 256.83 |
# | **Improved VAE Version 2** | **155.54** |
# 
# The FID score decreased from **256.83 to 155.54**, representing an improvement of approximately **39.4%**.
# 
# This substantial reduction suggests that the Improved VAE generated images with features that were considerably closer to real CIFAR-10 images than those produced by the Baseline VAE.
# 
# The result is consistent with the generated-image samples, where the Improved VAE produced stronger object-like shapes, clearer colour regions and more recognisable structures.
# 
# The improvement may be attributed to the deeper Encoder and Decoder architecture together with the revised VAE training configuration, which allowed the model to learn richer visual representations.
# 
# However, an FID score of **155.54** still indicates that noticeable differences remain between the generated and real CIFAR-10 image distributions.
# 
# Overall, the FID result provides strong evidence that the Improved VAE Version 2 achieved better image realism than the Baseline VAE.

# ## Improved VAE Inception Score
# 
# The same Inception Score evaluator is reused.
# 
# A **higher Inception Score** generally indicates that the generated images are more recognisable and diverse.

# In[68]:


improved_vae_is_mean, improved_vae_is_std = (
    is_evaluator.evaluate(
        generator=improved_vae_decoder,
        latent_dim=latent_dim,
        model_name="Improved VAE Version 2",
        generated_image_range="-1_to_1"
    )
)


# ## Improved VAE Version 2 Inception Score Evaluation
# 
# The Improved VAE Version 2 achieved an **Inception Score of 2.99 ± 0.20** using **1,000 generated CIFAR-10 images**.
# 
# A higher Inception Score generally indicates that generated images are both more recognisable and more diverse.
# 
# ### Result
# 
# | Model | Inception Score |
# |---|---:|
# | Baseline VAE | 1.87 ± 0.06 |
# | **Improved VAE Version 2** | **2.99 ± 0.20** |
# 
# The Inception Score increased from **1.87 to 2.99**, representing an improvement of approximately **59.9%**.
# 
# This suggests that the Improved VAE generated images with better recognisability and greater diversity than the Baseline VAE.
# 
# This result is supported by the predicted class distribution, where generated images were classified across **all ten CIFAR-10 classes**, although Cat and Dog remained the dominant predictions.
# 
# The higher score is also consistent with the generated-image samples, where stronger object-like structures can be observed compared with the much blurrier Baseline VAE outputs.
# 
# However, an Inception Score of **2.99 ± 0.20** still indicates that the generated images are not consistently clear or highly diverse.
# 
# Overall, the Improved VAE Version 2 demonstrates a clear improvement in both the **recognisability and diversity** of generated images compared with the Baseline VAE.

# ## Baseline and Improved VAE Comparison
# 

# In[69]:


print("VAE PERFORMANCE COMPARISON")
print("-" * 45)

print(
    f"Baseline VAE FID: "
    f"{baseline_vae_fid:.2f}"
)
print(
    f"Improved VAE FID: "
    f"{improved_vae_fid:.2f}"
)

fid_improvement = (
    baseline_vae_fid - improved_vae_fid
)

if fid_improvement > 0:
    print(
        f"FID improved by "
        f"{fid_improvement:.2f} points."
    )
else:
    print(
        f"FID increased by "
        f"{abs(fid_improvement):.2f} points."
    )

print()

print(
    f"Baseline VAE Inception Score: "
    f"{baseline_vae_is_mean:.2f} ± "
    f"{baseline_vae_is_std:.2f}"
)
print(
    f"Improved VAE Inception Score: "
    f"{improved_vae_is_mean:.2f} ± "
    f"{improved_vae_is_std:.2f}"
)

is_improvement = (
    improved_vae_is_mean
    - baseline_vae_is_mean
)

if is_improvement > 0:
    print(
        f"Inception Score improved by "
        f"{is_improvement:.2f} points."
    )
else:
    print(
        f"Inception Score decreased by "
        f"{abs(is_improvement):.2f} points."
    )


# ## VAE Performance Comparison
# 
# The Baseline VAE and Improved VAE Version 2 were compared using **FID** and **Inception Score** to evaluate whether the proposed improvements increased the quality and diversity of the generated CIFAR-10 images.
# 
# ### Quantitative Comparison
# 
# | Metric | Baseline VAE | Improved VAE Version 2 | Improvement |
# |---|---:|---:|---:|
# | **FID ↓** | 256.83 | **155.54** | **101.29 points lower** |
# | **Inception Score ↑** | 1.87 ± 0.06 | **2.99 ± 0.20** | **1.12 points higher** |
# 
# ### FID Comparison
# 
# The FID score decreased from **256.83 to 155.54**, an improvement of **101.29 points**, or approximately **39.4%**.
# 
# Since a lower FID score is better, this result suggests that the Improved VAE produced images whose feature distribution was substantially closer to that of real CIFAR-10 images.
# 
# This improvement is consistent with the visual results, where the Improved VAE generated images with stronger object-like shapes and more noticeable structures compared with the heavily blurred outputs of the Baseline VAE.
# 
# ### Inception Score Comparison
# 
# The Inception Score increased from **1.87 ± 0.06 to 2.99 ± 0.20**, an improvement of **1.12 points**, or approximately **59.9%**.
# 
# Since a higher Inception Score is better, this indicates that the Improved VAE generated images with better **recognisability and diversity**.
# 
# This is also supported by the predicted class distribution. The Improved VAE produced images that were classified across all ten CIFAR-10 classes, while the Baseline VAE outputs were much more heavily concentrated around a small number of predicted classes.
# 
# ### Overall Evaluation
# 
# Both quantitative metrics show that the changes made to the Improved VAE were effective.
# 
# The **101.29-point reduction in FID** indicates a substantial improvement in how closely the generated images resemble the real CIFAR-10 distribution, while the **1.12-point increase in Inception Score** indicates improved recognisability and diversity.
# 
# These improvements are also supported by the generated-image samples, training-loss behaviour and predicted class distribution.
# 
# Although the Improved VAE images are still somewhat blurry and the class distribution remains dominated by Cat and Dog predictions, the model performs significantly better than the Baseline VAE.
# 
# Therefore, the deeper Encoder and Decoder, revised KL weighting and longer training configuration successfully improved the overall generative performance of the VAE while keeping the latent dimension fixed at **100** for a fair comparison.M

# # Overall GAN and VAE Performance Comparison
# 
# The performance of all four generative models was compared using **Fréchet Inception Distance (FID)** and **Inception Score (IS)**.
# 
# - **Lower FID is better** because it indicates that the generated-image distribution is closer to the real CIFAR-10 image distribution.
# - **Higher Inception Score is better** because it indicates better recognisability and diversity in the generated images.
# 
# ## Quantitative Comparison
# 
# | Model | FID ↓ | Inception Score ↑ |
# |---|---:|---:|
# | Baseline DCGAN | 119.42 | 3.61 ± 0.13 |
# | **Improved DCGAN** | **87.60** | **5.36 ± 0.29** |
# | Baseline VAE | 256.83 | 1.87 ± 0.06 |
# | Improved VAE Version 2 | 155.54 | 2.99 ± 0.20 |
# 
# ## Model Ranking
# 
# Based on both evaluation metrics, the models can be ranked as:
# 
# 1. **Improved DCGAN**
# 2. **Baseline DCGAN**
# 3. **Improved VAE Version 2**
# 4. **Baseline VAE**
# 
# The Improved DCGAN achieved both the **lowest FID score of 87.60** and the **highest Inception Score of 5.36 ± 0.29**, making it the strongest overall model in this experiment.
# 
# ---
# 
# ## DCGAN Improvement
# 
# The Baseline DCGAN achieved an FID score of **119.42** and an Inception Score of **3.61 ± 0.13**.
# 
# After improving the architecture and training configuration, the Improved DCGAN achieved:
# 
# - FID: **87.60**
# - Inception Score: **5.36 ± 0.29**
# 
# The FID decreased by **31.82 points**, which is approximately a **26.6% improvement**.
# 
# The Inception Score increased by **1.75 points**, which is approximately a **48.5% improvement**.
# 
# This shows that the changes made to the DCGAN successfully improved both the similarity of generated images to real CIFAR-10 images and the recognisability and diversity of the outputs.
# 
# The visual evaluation also supports this result, as the Improved DCGAN produced clearer and more recognisable objects than the Baseline DCGAN.
# 
# ---
# 
# ## VAE Improvement
# 
# The Baseline VAE achieved an FID score of **256.83** and an Inception Score of **1.87 ± 0.06**.
# 
# After improving the VAE, Version 2 achieved:
# 
# - FID: **155.54**
# - Inception Score: **2.99 ± 0.20**
# 
# The FID decreased by **101.29 points**, representing an improvement of approximately **39.4%**.
# 
# The Inception Score increased by **1.12 points**, representing an improvement of approximately **59.9%**.
# 
# Therefore, the modifications made to the VAE were also successful.
# 
# The improved model produced stronger object-like structures and greater predicted class diversity than the Baseline VAE.
# 
# However, many generated images remained blurry, showing that the VAE still struggled to produce the same level of visual sharpness as the DCGAN models.
# 
# ---
# 
# ## GAN vs VAE Comparison
# 
# The results show a clear difference between the performance of the GAN and VAE frameworks.
# 
# Both DCGAN models achieved better FID and Inception Scores than both VAE models.
# 
# For example, the best models from each framework achieved:
# 
# | Best Model | FID ↓ | Inception Score ↑ |
# |---|---:|---:|
# | **Improved DCGAN** | **87.60** | **5.36 ± 0.29** |
# | Improved VAE Version 2 | 155.54 | 2.99 ± 0.20 |
# 
# The Improved DCGAN achieved an FID score that was **67.94 points lower** than the Improved VAE and an Inception Score that was **2.37 points higher**.
# 
# ### Why did the DCGAN perform better?
# 
# The main objective of the DCGAN is to generate images that are realistic enough to fool the Discriminator.
# 
# The Generator is therefore directly encouraged to produce increasingly realistic visual features during adversarial training.
# 
# In contrast, the VAE has to optimise two objectives:
# 
# - reconstructing image information
# - maintaining a structured latent distribution through KL divergence
# 
# This trade-off can cause VAE outputs to become smoother or blurrier.
# 
# This behaviour was visible in the generated images, where the VAE generally produced weaker object boundaries and less detailed structures than the DCGAN.
# 
# ---
# 
# ## Effectiveness of the Improvements
# 
# Importantly, **both improved models performed better than their respective baseline models**.
# 
# | Framework | FID Improvement | IS Improvement |
# |---|---:|---:|
# | DCGAN | 119.42 → **87.60** | 3.61 → **5.36** |
# | VAE | 256.83 → **155.54** | 1.87 → **2.99** |
# 
# This demonstrates that the proposed improvements were effective rather than simply increasing model complexity without improving performance.
# 
# The DCGAN improvements increased image realism and diversity, while the VAE improvements substantially reduced the difference between generated and real-image distributions.
# 
# ---
# 
# # Final Conclusion
# 
# Overall, the experiment demonstrates that both **DCGAN and VAE can learn the CIFAR-10 image distribution and generate new images**, but their performance differs significantly.
# 
# The **Baseline VAE produced the weakest results**, with the highest FID score of **256.83** and the lowest Inception Score of **1.87 ± 0.06**. Its generated images were generally blurry and showed limited diversity.
# 
# The **Improved VAE Version 2 significantly improved these results**, reducing FID to **155.54** and increasing the Inception Score to **2.99 ± 0.20**. This demonstrates that the deeper architecture and revised training configuration improved the VAE's generative ability. However, the generated images remained relatively blurry.
# 
# The **Baseline DCGAN already performed better than both VAE models**, achieving an FID of **119.42** and an Inception Score of **3.61 ± 0.13**.
# 
# The **Improved DCGAN achieved the best overall performance**, with the lowest FID of **87.60** and highest Inception Score of **5.36 ± 0.29**.
# 
# Therefore, the **Improved DCGAN is selected as the best-performing model for this problem**.
# 
# For the objective of generating realistic and diverse CIFAR-10 images, the adversarial learning approach of the DCGAN was more effective than the reconstruction-based VAE approach.
# 
# At the same time, the improvement from both baseline models demonstrates that architectural design, training parameters and loss configuration have a significant effect on generative-model performance.
# 
# Overall, the results successfully demonstrate the process of **building, evaluating and improving GAN and VAE models**, with the Improved DCGAN providing the strongest final solution for CIFAR-10 image generation.

# In[71]:


### Saving weights for improved DCGAN because it is the best model!!! YAY
# Save Improved DCGAN weights

improved_generator.save_weights(
    "improved_dcgan_generator.weights.h5"
)

improved_discriminator.save_weights(
    "improved_dcgan_discriminator.weights.h5"
)

print("Improved DCGAN weights saved successfully.")

