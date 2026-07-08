from diabetes import (
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test
)

from sklearn.neighbors import KNeighborsClassifier
from evaluation import evaluate_model

model = KNeighborsClassifier(
    n_neighbors=5
)

evaluate_model(
    model,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    "KNN"
)