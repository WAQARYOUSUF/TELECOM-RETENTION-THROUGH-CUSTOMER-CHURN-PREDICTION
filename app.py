# coding: utf-8
import os
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# ----------------- Load Model and Dataset -----------------

df_1 = pd.read_csv("first_telc.csv")  # Ensure this is deployed
model = pickle.load(open("model.sav", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

# ----------------- Routes -----------------

@app.route("/")
def loadPage():
    return render_template("home.html")

@app.route("/", methods=["POST"])
def predict():
    app.logger.info("Received POST request")  # Log POST

    try:
        # ----------- GET INPUTS (Safe Parsing) -----------
        inputQuery1 = int(request.form.get("query1", 0))
        inputQuery2 = float(request.form.get("query2", 0.0))
        inputQuery3 = float(request.form.get("query3", 0.0))
        inputQuery4 = request.form.get("query4", "Male")
        inputQuery5 = request.form.get("query5", "No")
        inputQuery6 = request.form.get("query6", "No")
        inputQuery7 = request.form.get("query7", "No")
        inputQuery8 = request.form.get("query8", "No")
        inputQuery9 = request.form.get("query9", "No")
        inputQuery10 = request.form.get("query10", "No")
        inputQuery11 = request.form.get("query11", "No")
        inputQuery12 = request.form.get("query12", "No")
        inputQuery13 = request.form.get("query13", "No")
        inputQuery14 = request.form.get("query14", "No")
        inputQuery15 = request.form.get("query15", "No")
        inputQuery16 = request.form.get("query16", "Month-to-month")
        inputQuery17 = request.form.get("query17", "No")
        inputQuery18 = request.form.get("query18", "Electronic check")
        inputQuery19 = int(request.form.get("query19", 1))
    except Exception as e:
        return render_template("home.html", output1=f"Input Error: {e}", output2="")

    # ----------- CREATE DATAFRAME -----------
    data = [[
        inputQuery1, inputQuery2, inputQuery3, inputQuery4,
        inputQuery5, inputQuery6, inputQuery7, inputQuery8,
        inputQuery9, inputQuery10, inputQuery11, inputQuery12,
        inputQuery13, inputQuery14, inputQuery15, inputQuery16,
        inputQuery17, inputQuery18, inputQuery19
    ]]

    new_df = pd.DataFrame(data, columns=[
        "SeniorCitizen", "MonthlyCharges", "TotalCharges", "gender",
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "tenure"
    ])

    # ----------- Merge with Original Dataset -----------
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # ----------- Tenure Grouping -----------
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2["tenure"].astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels
    )
    df_2.drop(columns=["tenure"], inplace=True)

    # ----------- Create Dummies -----------
    new_df_dummies = pd.get_dummies(df_2[[
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure_group"
    ]])

    # ----------- Remove Duplicate Columns -----------
    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]
    model_columns_unique = list(dict.fromkeys(model_columns))
    new_df_dummies = new_df_dummies.reindex(
        columns=model_columns_unique,
        fill_value=0
    )

    # ----------- Prediction -----------
    prediction = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1][0]

    if probability > 0.5:
        result_text = "This customer is likely to be churned!!"
    else:
        result_text = "This customer is likely to continue!!"

    probability_text = "Churn Probability: {:.2f}%".format(probability * 100)

    return render_template("home.html", output1=result_text, output2=probability_text)

# ----------------- Main -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render dynamic port
    app.run(host="0.0.0.0", port=port, debug=True)
