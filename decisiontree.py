from diabetes import (
    X_train,
    X_test,
    y_train,
    y_test
)

from sklearn.tree import DecisionTreeClassifier
from evaluation import evaluate_model

model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    "Decision Tree"
)