# Diabetes ML Model Training & Evaluation Pipeline

A lightweight machine learning project demonstrating dataset handling, model training, evaluation metrics, and retraining workflows using `scikit-learn`.

## 📌 Overview

This small project showcases a fundamental Machine Learning workflow using Python and Scikit-Learn. It loads the built-in Scikit-Learn Diabetes dataset, trains a baseline `LinearRegression` model, evaluates performance across multiple regression metrics, and demonstrates conditional model retraining logic.

## 🚀 Key Features

* **Data Preprocessing & Splitting**: Splitting tabular datasets into train and test sets using `train_test_split`.
* **Supervised Learning**: Training a `LinearRegression` baseline model.
* **Model Evaluation**: Assessing predictions using:
  * **$R^2$ Score** (Coefficient of Determination)
  * **MAE** (Mean Absolute Error)
  * **MSE** (Mean Squared Error)
* **Retraining Pipeline**: Functionality to trigger model retraining based on custom evaluation thresholds.

## 🛠️ Technologies Used

* **Python 3.11**
* **scikit-learn**

## 📂 Project Structure

```text
.
├── train.py   # Dataset loading, train/test split, model training, and retraining functions
└── eval.py    # Model evaluation metrics and performance checks
```

## ⚙️ Getting Started

### Prerequisites

Ensure you have Python 3.11+ installed. Install the required dependency:

```bash
pip install scikit-learn
```

### Running the Project

Run the evaluation script to execute the training and evaluation workflow:

```bash
python eval.py
```

## 📊 Workflow Summary

1. **Load Data**: Imports the Diabetes dataset ($X$ features and $y$ target variable).
2. **Train**: Fits a `LinearRegression` model on 80% of the data (`X_train`, `y_train`).
3. **Evaluate**: Evaluates predictions against the remaining 20% (`X_test`, `y_test`).
4. **Conditional Retrain**: If evaluation metrics fall below defined performance thresholds ($R^2 < 0.8$, $\text{MSE} > 1000$, or $\text{MAE} > 40$), a retraining step is triggered.