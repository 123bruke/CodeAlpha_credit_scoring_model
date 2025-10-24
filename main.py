#main format file 

from data_preparation import generate_dataset
from visualization import show_heatmap
from model_training import prepare_data, build_models
from evaluation import evaluate_models


df = generate_dataset()
print("Dataset created successfully!\n")
print(df.head())
show_heatmap(df)

X_train, X_test, y_train, y_test = prepare_data(df)
print("Data prepared successfully!\n")

models = build_models()

features = ["Income", "Debts", "Payment_History", "Age", "Employment_Status", "Credit_Utilization"]
results = evaluate_models(models, X_train, X_test, y_train, y_test, features)
print("\nModel Accuracy Comparison:")
for model_name, score in results.items():
    print(f"{model_name}: {round(score * 100, 2)}%")
