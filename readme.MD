<p align="center">
  <img src="assets/nitr.png" alt="NIT Rourkela Logo" width="170"/>
</p>

<h1 align="center">
Diabetes Prediction Using Supervised Machine Learning Classification Algorithms
</h1>

<p align="center">
<b>A Project Report Submitted for</b><br>
<b>One-Month Internship</b>
</p>

<p align="center">
<b>Submitted by</b><br><br>

<b>Md Amanullah</b><br>
md.amanullah.cse.2024@nist.edu
</p>

<p align="center">
Department of Computer Science and Engineering<br>
National Institute of Technology Rourkela<br>
Rourkela, Odisha 769008, India
</p>

---

## About the Project

This repository contains the implementation of a **Diabetes Prediction System** developed as part of a **One-Month Internship**. The project applies and compares multiple supervised machine learning classification algorithms to predict diabetes using demographic and clinical patient data.

The project implements a complete machine learning pipeline including:

- Data preprocessing
- Feature engineering
- Model training
- Cross-validation
- Performance evaluation
- Comparative analysis

A total of **seven supervised learning algorithms** are trained and evaluated using a common preprocessing pipeline to ensure a fair comparison.

---

## Objectives

- Develop an accurate diabetes prediction model.
- Compare multiple supervised machine learning algorithms.
- Evaluate each classifier using standard performance metrics.
- Identify the most suitable algorithm for diabetes prediction.

---

## Dataset

**Source:** Kaggle – Diabetes Prediction Dataset

The dataset contains approximately **100,000 patient records** consisting of demographic and clinical attributes.

### Features

| Feature | Description |
|----------|-------------|
| Gender | Patient Gender |
| Age | Age |
| Hypertension | Hypertension Status |
| Heart Disease | Heart Disease Status |
| Smoking History | Smoking Status |
| BMI | Body Mass Index |
| HbA1c Level | Average Blood Sugar Level |
| Blood Glucose Level | Blood Glucose Measurement |
| Diabetes | Target Variable (0 = Non-Diabetic, 1 = Diabetic) |

---

## Data Preprocessing

The preprocessing pipeline consists of:

- Duplicate record removal
- One-hot encoding of categorical variables
- Stratified 80:20 train-test split
- Feature standardization using StandardScaler
- Class imbalance handling using `class_weight="balanced"` where applicable

Algorithms requiring feature scaling:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine

Tree-based models were trained using unscaled features.

---

## Machine Learning Models

| Algorithm | Category |
|------------|----------|
| Logistic Regression | Linear Classification |
| K-Nearest Neighbors | Instance-Based Learning |
| Decision Tree | Decision Tree |
| Random Forest | Bagging Ensemble |
| Support Vector Machine | Margin-Based Classification |
| XGBoost | Gradient Boosting Ensemble |
| Gradient Boosting | Boosting Ensemble |

---

## Evaluation Metrics

The following metrics are used to evaluate each model:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- 5-Fold Cross Validation

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | 5-Fold CV |
|:-------------------------------|:-------:|:---------:|:------:|:-------:|:---------:|
| Logistic Regression | 88.46% | 42.54% | **87.91%** | 57.34% | 88.72% |
| Support Vector Machine | 88.48% | 42.59% | **87.97%** | 57.40% | 88.73% |
| K-Nearest Neighbors | 95.96% | 88.98% | 61.91% | 73.02% | 95.90% |
| Decision Tree | 95.04% | 71.14% | 73.70% | 72.40% | 95.05% |
| Random Forest | 96.95% | 94.90% | 69.10% | 79.97% | 96.84% |
| XGBoost | 97.05% | 95.34% | 69.99% | 80.72% | 96.93% |
| **Gradient Boosting** | **97.18%** | **98.65%** | 68.99% | **81.19%** | **97.08%** |

Gradient Boosting achieved the highest overall accuracy and F1-score. Logistic Regression and Support Vector Machine achieved the highest recall, making them suitable for high-sensitivity screening applications.

---

## Confusion Matrix Comparison

| Model | TN | FP | FN | TP |
|:-------------------------------|------:|------:|------:|------:|
| Logistic Regression | 15,520 | 2,014 | 205 | 1,491 |
| Support Vector Machine | 15,523 | 2,011 | 204 | 1,492 |
| K-Nearest Neighbors | 17,404 | 130 | 646 | 1,050 |
| Decision Tree | 17,027 | 507 | 446 | 1,250 |
| Random Forest | 17,471 | 63 | 524 | 1,172 |
| XGBoost | 17,476 | 58 | 509 | 1,187 |
| **Gradient Boosting** | **17,518** | **16** | 526 | 1,170 |

### Key Observations

- Gradient Boosting achieved the highest accuracy and F1-score.
- XGBoost and Random Forest performed comparably to Gradient Boosting.
- Logistic Regression and Support Vector Machine achieved the highest recall, making them effective for screening applications.
- Ensemble learning methods outperformed individual classifiers in overall predictive performance.



## Installation

Clone the repository:

```bash
git clone https://github.com/amaan-exe/diabetes_prediction.git
```

Navigate to the project directory:

```bash
cd diabetes_prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute all models:

```bash
python main.py
```

Run individual models:

```bash
python logistic_reg.py
python knn.py
python decisiontree.py
python rf.py
python svmm.py
python gb.py
python xboost.py
```

---

## Results

The repository includes:

- Dataset distribution
- Performance comparison charts
- Confusion matrices for all models
- Confusion matrix of the best-performing model

---

## Future Improvements

- Hyperparameter tuning
- SMOTE for class imbalance
- Feature selection
- Explainable AI using SHAP or LIME
- Deep learning models
- Flask/FastAPI deployment
- Docker support
- Cloud deployment

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib

---

## Project Report

The complete internship report describing the methodology, implementation, experimental setup, evaluation metrics, and comparative analysis is included in this repository.

---

## Author

**Md Amanullah**

Department of Computer Science and Engineering

National Institute of Technology Rourkela

Email: **md.amanullah.cse.2024@nist.edu**

GitHub: **https://github.com/amaan-exe**

---

If you find this project useful, consider starring the repository.