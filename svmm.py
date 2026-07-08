from diabetes import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

from sklearn.svm import LinearSVC
from evaluation import evaluate_model


model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=5000
)

evaluate_model(
    model,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "SVM"
)