import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# dataset load
df = pd.read_csv("bank.csv")

# features aur target
X = df[["age", "balance", "duration", "campaign"]]
y = df["y"]

# data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier()

# training
model.fit(X_train, y_train)

# save model
joblib.dump(model, "model.pkl")

print("Model saved successfully as model.pkl")
