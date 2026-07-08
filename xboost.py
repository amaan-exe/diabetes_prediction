from diabetes import (
    X_train,
    X_test,
    y_train,
    y_test
)

from xgboost import XGBClassifier
from evaluation import evaluate_model

model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    "XGBoost"
)