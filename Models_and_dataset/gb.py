from diabetes import (
    X_train,
    X_test,
    y_train,
    y_test
)

from sklearn.ensemble import GradientBoostingClassifier
from evaluation import evaluate_model

model = GradientBoostingClassifier(
    random_state=42
)

evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Gradient Boosting"
)