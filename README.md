# CIFAR-10 Image Classification with ViT, ResNet, and CNN-MLP Hybrid


## Description

This project implements and evaluates three different deep learning architectures for classifying images from the CIFAR-10 dataset. The primary objective is to compare the performance of a **Vision Transformer (ViT)** built from scratch, a **pre-trained ResNet-18 model** using transfer learning, and a novel **hybrid CNN-MLP model**.

The project explores different approaches to feature extraction and classification in computer vision, from the convolutional strengths of ResNet and CNNs to the attention-based mechanisms of Transformers. The models are trained and evaluated on key metrics including accuracy, precision, recall, and F1-score, with results visualized using loss curves and confusion matrices. The entire project is built using PyTorch.

## Features

- **Vision Transformer (ViT) from Scratch**: A complete implementation including patch embedding, positional encoding, and multiple Transformer encoder layers.
- **Hybrid CNN-MLP Model**: A custom architecture that uses a CNN for feature extraction from image patches, followed by an MLP for classification.
- **Pre-trained ResNet-18**: Utilizes transfer learning by fine-tuning the final classifier layer of a ResNet-18 model pre-trained on ImageNet.
- **Data Augmentation**: Standard techniques like random cropping, horizontal flipping, and color jitter are applied to improve model generalization.
- **Comprehensive Training & Evaluation**: Includes complete training loops with learning rate scheduling and early stopping to prevent overfitting.
- **Performance Comparison**: The performance of all three models is analyzed based on classification metrics and training behavior.

## Model Architectures

1.  **Vision Transformer (`vit-2.ipynb`)**:
    - **Image Size**: 32x32
    - **Patch Size**: 4x4
    - **Embedding Dimension**: 64
    - **Transformer Layers**: 8
    - **Attention Heads**: 4
    - Divides images into patches, linearly embeds them, adds positional encodings, and processes them through a series of Transformer encoder blocks.

2.  **Hybrid CNN-MLP (`hybrid-mlp-cnn.ipynb`)**:
    - **Patch Size**: 4x4
    - **Embedding Dimension**: 64
    - A unique model that first creates patch embeddings, then uses a series of convolutional and max-pooling layers to extract features before a final MLP classifier.

3.  **ResNet-18 (`resnet18.ipynb`)**:
    - A pre-trained ResNet-18 model is used.
    - The feature extraction layers are frozen, and only the final fully connected layer is replaced and trained for CIFAR-10's 10 classes.

## Results

The models were trained and evaluated on the CIFAR-10 test set. The ResNet-18 model achieved the highest accuracy, demonstrating the power of transfer learning from large-scale datasets.

| Model                           | Test Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
| ------------------------------- | :-----------: | :---------------: | :------------: | :--------------: |
| **ResNet-18 (Transfer Learning)** |  **~81.4%**   |      ~0.8180      |    ~0.8139     |     ~0.8122      |
| **Hybrid CNN-MLP**              |    ~74.0%     |      ~0.7381      |    ~0.7403     |     ~0.7383      |
| **Vision Transformer (ViT)**    |    ~53.8%     |      ~0.5339      |    ~0.5385     |     ~0.5305      |

## Visualizations

## Dependencies

The project requires the following Python libraries. You can install them all using `pip install -r requirements.txt`.

- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

_This list is provided in `requirements.txt`._

## Dataset

- **Name**: **CIFAR-10 Dataset**
- **Source**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Description**: The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
