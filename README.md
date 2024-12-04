# **Handwritten Digit Classification Using SVM, KNN, and PCA**

This project implements handwritten digit classification using Support Vector Machines (SVM) and k-Nearest Neighbors (KNN) with Principal Component Analysis (PCA) for dimensionality reduction. The goal is to accurately classify handwritten digits from a dataset, optimizing both model performance and computational efficiency.

---

## **Project Overview**

- **Objective**: Classify handwritten digits using machine learning models.
- **Models Used**: 
  - **SVM** with Radial Basis Function (RBF) kernel and linear kernel.
  - **KNN** with different values of k.
- **Dimensionality Reduction**: PCA was applied to reduce feature dimensions while retaining key patterns for classification.

---

## **Results**

- **SVM with RBF Kernel**: Achieved 98% accuracy on the test set.
- **KNN**: Achieved 97% accuracy on the test set.
- **PCA**: Improved computational efficiency without significant loss in classification accuracy.

---

## **Key Features**

- Efficient use of PCA for dimensionality reduction.
- Comparison of linear and non-linear kernels in SVM.
- Robust evaluation metrics: precision, recall, F1-score.
- Confusion matrix analysis to visualize classification performance.

---

## **Usage Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/digit-classification-SVM-KNN.git
cd digit-classification-SVM-KNN
```

### **2. Install Required Packages**
Ensure you have the necessary Python libraries installed:
```bash
pip install numpy pandas scikit-learn matplotlib
```

### **3. Run the Jupyter Notebook**
```bash
jupyter notebook digit_classifier.ipynb
```

---

## **Technologies Used**

- **Python**: Core language for implementation.
- **Jupyter Notebook**: Interactive development and visualization.
- **scikit-learn**: Machine learning models (SVM, KNN, PCA).
- **matplotlib**: Data visualization.

---

## **Future Work**

- **Hyperparameter Optimization**: Explore techniques like grid search or Bayesian optimization.
- **Ensemble Methods**: Implement methods like bagging or boosting for improved accuracy.
- **Advanced Feature Extraction**: Integrate Convolutional Neural Networks (CNNs) for better feature representation.
- **Real-time Deployment**: Explore real-time applications such as mobile digit recognition.

---

## **Contributors**

- **Ileana Bocage** - Developer and Author
