# data visualization library

import matplotlib.pyplot as plt
import seaborn as sns

def show_heatmap(df):
    """Display correlation heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
