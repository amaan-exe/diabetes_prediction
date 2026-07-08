from diabetes import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

from sklearn.linear_model import LogisticRegression
from evaluation import evaluate_model
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)
evaluate_model(
    model=model,
    X_train=X_train_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_test=y_test,
    model_name="Logistic Regression"
)