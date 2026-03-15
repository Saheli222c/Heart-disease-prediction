import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset/heart.csv")

# Style
sns.set_style("whitegrid")

# 1 Age distribution
plt.figure()
sns.histplot(data["age"], kde=True)
plt.title("Age Distribution")
plt.show()

# 2 Heart disease count
plt.figure()
sns.countplot(x="heart disease", data=data)
plt.title("Heart Disease Count")
plt.show()

# 3 Chest pain vs heart disease
plt.figure()
sns.countplot(x="chest pain type", hue="heart disease", data=data)
plt.title("Chest Pain vs Heart Disease")
plt.show()

# 4 Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()