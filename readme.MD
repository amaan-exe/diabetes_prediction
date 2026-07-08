# 🩺 Diabetes Prediction Using Supervised Machine Learning Classification Algorithms

A machine learning project that compares the performance of seven supervised classification algorithms for predicting diabetes using patient demographic and clinical data.

## Overview

Early detection of diabetes can significantly reduce the risk of severe complications and improve patient outcomes. This project develops and evaluates multiple supervised machine learning models to classify whether a patient is diabetic based on health-related attributes.

All models are trained and evaluated using the same preprocessing pipeline, allowing a fair comparison of their predictive performance.

---

##  Features

- Data preprocessing and cleaning
- One-hot encoding for categorical features
- Feature scaling where required
- 80:20 stratified train-test split
- Comparison of 7 supervised ML algorithms
- 5-Fold Cross Validation
- Performance evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

##  Dataset

**Source:** [Kaggle - Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

**Dataset Characteristics**

- ~100,000 patient records
- Demographic and clinical attributes
- Binary classification problem

### Features

| Feature | Description |
|----------|-------------|
| Gender | Patient gender |
| Age | Age in years |
| Hypertension | Presence of hypertension |
| Heart Disease | Presence of heart disease |
| Smoking History | Smoking status |
| BMI | Body Mass Index |
| HbA1c Level | Average blood glucose level |
| Blood Glucose Level | Blood glucose measurement |
| Diabetes | Target variable (0 = No, 1 = Yes) |

---

##  Data Preprocessing

The preprocessing pipeline includes:

- Removal of duplicate records
- One-hot encoding of categorical variables
- Stratified train-test split (80:20)
- Standardization of numerical features for distance-based algorithms
- Handling class imbalance using class weights where applicable

---

##  Machine Learning Models

The following supervised learning algorithms were implemented:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (Linear SVM)
- XGBoost
- Gradient Boosting

---

##  Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- 5-Fold Cross Validation

---

##  Project Structure

```text
.
├── main.py
├── diabetes.py
├── evaluation.py
├── logistic_reg.py
├── knn.py
├── decisiontree.py
├── rf.py
├── svmm.py
├── xboost.py
├── gb.py
├── diabetes.csv
├── requirements.txt
└── README.md
```

---

##  Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 88.46% |
| SVM | 88.48% |
| KNN | 95.96% |
| Decision Tree | 95.04% |
| Random Forest | 96.95% |
| XGBoost | 97.05% |
| **Gradient Boosting** | **97.18%** |

Gradient Boosting achieved the highest overall accuracy, while Logistic Regression and SVM demonstrated higher recall for the diabetic class, making them suitable for screening-oriented applications. These results are documented in your report. :contentReference[oaicite:0]{index=0}

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/amaan-exe/diabetes_prediction.git
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

Or run an individual model:

```bash
python logistic_reg.py
python knn.py
python decisiontree.py
python rf.py
python svmm.py
python xboost.py
python gb.py
```
---

##  Technologies Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost

---

##  Future Improvements

- Hyperparameter optimization using Grid Search or Random Search
- SMOTE for class imbalance handling
- Feature selection techniques
- Model explainability using SHAP or LIME
- Deep Learning implementation
- Flask/FastAPI web deployment
- Docker containerization

---

##  Author

**Md Amanullah**

Computer Science & Engineering

GitHub: https://github.com/amaan-exe

Email : md.amanullah.cse.2024@nist.edu