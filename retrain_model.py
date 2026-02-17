import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df["tenure_group"] = pd.cut(df.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)

df.drop(columns=["tenure"], inplace=True)

X = pd.get_dummies(df.drop("Churn", axis=1))
y = df["Churn"]

model = RandomForestClassifier()
model.fit(X, y)

# 🔥 Save both model AND columns
pickle.dump(model, open("model.sav", "wb"))
pickle.dump(X.columns, open("model_columns.pkl", "wb"))

print("✅ Model and columns saved successfully!")
