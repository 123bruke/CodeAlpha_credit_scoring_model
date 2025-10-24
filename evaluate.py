
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def evaluate_models(models, X_train, X_test, y_train, y_test, features):
    """Train, predict, and evaluate models"""
    accuracy = {}

    for name, model in models.items():
        print(f"\n==== {name} ====")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=features)
            feat_imp.sort_values().plot(kind='barh', figsize=(6, 4))
            plt.title(f"{name} Feature Importance")
            plt.show()

        accuracy[name] = model.score(X_test, y_test)

    return accuracy
