# Breast Cancer Classification using KNN

- [Breast Cancer Classification using KNN](#breast-cancer-classification-using-knn)
  - [About](#about)
  - [K-Nearest Neighbors Classification](#k-nearest-neighbors-classification)
  - [Results](#results)
  - [References](#references)


## About

This dataset is provided by UCI Machine Learning on Kaggle and is available [here](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). 

The task is to predict whether the cancer is benign or malignant. Features describe characteristics of the cell nuclei present in the image. They are:
* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter^2 / area - 1.0)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension ("coastline approximation" - 1)


## K-Nearest Neighbors Classification

The KNN algorithm is a simple, easy-to-implement supervised machine learning algorithm that's used to solve both regression and classification tasks. 

Similar data points are grouped together. This follows the assumption that similar things exist in close proximity (calculate euclidean distance).

## Results

The notebook containing all steps: [BCC_kNN.ipynb](BCC_kNN.ipynb)

The KNN model (K=3) achieved an accuracy of **97%** on both training and testing set and an ROC AUC score of **0.967**

## References
1. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761
2. https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization