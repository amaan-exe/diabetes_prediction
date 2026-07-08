from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )
    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    print("Cross Validation Scores:", cv_scores)
    print("CV Mean :", cv_scores.mean())
    print("CV Std  :", cv_scores.std())

    print("=" * 60)