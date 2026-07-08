from diabetes import (
    X_train,
    X_test,
    y_train,
    y_test
)

from sklearn.ensemble import RandomForestClassifier
from evaluation import evaluate_model

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Random Forest"
)