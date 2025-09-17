import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset with multi-level headers
df = pd.read_csv("AAPL_stock_data.csv", header=[0, 1], index_col=0, parse_dates=True)

# Flatten multi-level columns → ('Price', 'Close') becomes 'Close'
df.columns = df.columns.get_level_values(0)

# Reset index to get Date as a column
df = df.reset_index().rename(columns={'index': 'Date'})

# Sort by Date
df = df.sort_values(by="Date")

# Target: 1 if next day's Close > today's Close
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

# Drop last row (NaN target)
df = df.dropna()

# ✅ Only numeric features
X = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)
nb_acc = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", nb_acc)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_acc)

# Save models + scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("naive_bayes.pkl", "wb") as f:
    pickle.dump(nb, f)

with open("logistic_regression.pkl", "wb") as f:
    pickle.dump(lr, f)

# 📊 Plot accuracy comparison
models = ["Naive Bayes", "Logistic Regression"]
accuracy = np.array([nb_acc * 100, lr_acc * 100])  # NumPy array

plt.bar(models, accuracy, color=["skyblue", "orange"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(40, 55)

# Label bars with accuracy values
for i, acc in enumerate(accuracy):
    plt.text(i, acc + 0.5, f"{acc:.2f}%", ha="center")

# Save the graph as PNG
plt.savefig("model_accuracy_comparison.png", dpi=300, bbox_inches="tight")

plt.show()
