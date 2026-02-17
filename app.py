# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__, debug=True)


# Load dataset (used for dummy alignment)
df_1 = pd.read_csv("first_telc.csv")

# Load trained model and saved columns
model = pickle.load(open("model.sav", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))


@app.route("/")
def loadPage():
    return render_template("home.html")


@app.route("/", methods=["POST"])
def predict():

    # ----------- GET INPUTS -----------

    inputQuery1 = int(request.form["query1"])          # SeniorCitizen
    inputQuery2 = float(request.form["query2"])        # MonthlyCharges
    inputQuery3 = float(request.form["query3"])        # TotalCharges
    inputQuery4 = request.form["query4"]               # gender
    inputQuery5 = request.form["query5"]               # Partner
    inputQuery6 = request.form["query6"]               # Dependents
    inputQuery7 = request.form["query7"]               # PhoneService
    inputQuery8 = request.form["query8"]               # MultipleLines
    inputQuery9 = request.form["query9"]               # InternetService
    inputQuery10 = request.form["query10"]             # OnlineSecurity
    inputQuery11 = request.form["query11"]             # OnlineBackup
    inputQuery12 = request.form["query12"]             # DeviceProtection
    inputQuery13 = request.form["query13"]             # TechSupport
    inputQuery14 = request.form["query14"]             # StreamingTV
    inputQuery15 = request.form["query15"]             # StreamingMovies
    inputQuery16 = request.form["query16"]             # Contract
    inputQuery17 = request.form["query17"]             # PaperlessBilling
    inputQuery18 = request.form["query18"]             # PaymentMethod
    inputQuery19 = int(request.form["query19"])        # tenure

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

    # ----------- MERGE WITH ORIGINAL DATA -----------

    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # ----------- TENURE GROUPING -----------

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    df_2["tenure_group"] = pd.cut(
        df_2["tenure"].astype(int),
        range(1, 80, 12),
        right=False,
        labels=labels
    )

    df_2.drop(columns=["tenure"], inplace=True)

    # ----------- CREATE DUMMIES -----------

    new_df_dummies = pd.get_dummies(df_2[[
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
        "tenure_group"
    ]])

    # ----------- REMOVE DUPLICATE COLUMNS -----------

    new_df_dummies = new_df_dummies.loc[:, ~new_df_dummies.columns.duplicated()]

    model_columns_unique = list(dict.fromkeys(model_columns))

    new_df_dummies = new_df_dummies.reindex(
        columns=model_columns_unique,
        fill_value=0
    )

    # ----------- PREDICTION -----------

    prediction = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1][0]

    # Custom threshold (optional improvement)
    if probability > 0.5:
        result_text = "This customer is likely to be churned!!"
    else:
        result_text = "This customer is likely to continue!!"

    probability_text = "Churn Probability: {:.2f}%".format(probability * 100)

    return render_template(
        "home.html",
        output1=result_text,
        output2=probability_text
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
