# garbage-classifier

**Garbage Image Classification (CNN)**

 <p>&nbsp;</p>

**## Overview**

This project develops image classification models to classify waste images into 6 categories.  
The goal is to explore transfer learning and data augmentation techniques to improve classification accuracy on a relatively small dataset (~2,500 images).

 <p>&nbsp;</p>

**## Dataset**

　- Source: Garbage classification dataset (not included here)

　- Size: ~2,500 images

　- Classes: 6 categories (e.g., cardboard, glass, metal, paper, plastic, trash)

　- Note: Since the dataset does not provide a separate test set, part of the training data was split and used for evaluation.

 <p>&nbsp;</p>

**## Preprocessing steps**

　- Applied data augmentation using `torchvision.transforms`:
 
　    Resize, random rotation, color jitter, random crop, normalization

　- Constructed custom `Dataset` and `DataLoader`

　- Split dataset into training and test sets

 <p>&nbsp;</p>

**## Models & Methods**

　- Compared multiple pre-trained CNN architectures:
 
　 ResNet18, ResNet50, EfficientNet-B0, MobileNet-V2, VGG16

　- Introduced dropout in fully connected layers to evaluate generalization effect

　- Used weighted cross-entropy loss to address class imbalance

　- Compared optimizers (Adam, RMSprop, SGD)

 <p>&nbsp;</p>

**## Results**

　- Best test accuracy: **~77.5%**

　- Evaluated confusion matrix and classification report to analyze misclassifications

　- Data augmentation contributed to improved robustness

 <p>&nbsp;</p>

**## Technologies Used**

　- Python, Pandas

　- PyTorch, Torchvision

　- Pre-trained CNNs (ResNet, EfficientNet, MobileNet, VGG)

　- scikit-learn (evaluation)

　- Matplotlib

　- Jupyter Notebook

 <p>&nbsp;</p>

**## Repository Structure**

```

garbage-classifier/

├── garbage_classifier.ipynb   # Main notebook

├── README.md            # Project description

└── data/                # Dataset (not included, see below)

```

<p>&nbsp;</p>

**## About Dataset**

The dataset is not included in this repository due to license restrictions. Please download it directly from Kaggle.

https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

<p>&nbsp;</p>

**## Note**

This notebook was originally developed and executed in a local Jupyter/Colab environment. 

Due to the use of custom folder structures (e.g., `data/`, `notebook/`, `model/`), it may not run directly without modifications.  

The main purpose of this repository is to showcase the analysis process and results, rather than to provide a fully reproducible environment.
