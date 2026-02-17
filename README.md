# 📊 Customer Churn Prediction System

> 🚀 A Machine Learning powered Flask web application that predicts
> telecom customer churn probability using a trained Random Forest
> model.

------------------------------------------------------------------------

## 🌟 Overview

Customer churn prediction is critical for telecom businesses to reduce
customer loss and improve retention strategies.

This project builds an end-to-end Machine Learning pipeline and deploys
it as a web application using Flask.

The system predicts:

-   ✅ Whether a customer is likely to churn\
-   📈 Churn Probability (%) score

------------------------------------------------------------------------

## 🎯 Business Objective

Telecom companies lose significant revenue due to customer churn.

By identifying high-risk customers early, businesses can:

-   Improve retention campaigns\
-   Offer targeted promotions\
-   Increase customer lifetime value\
-   Reduce acquisition costs

------------------------------------------------------------------------

## 🧠 Machine Learning Workflow

### 📌 Dataset

Telco Customer Churn Dataset

### 🔎 Steps Performed

1.  Data Cleaning\
2.  Handling Missing Values\
3.  Feature Engineering\
4.  Tenure Grouping\
5.  One-Hot Encoding\
6.  Model Training (Random Forest Classifier)\
7.  Model Evaluation\
8.  Model Serialization using Pickle

------------------------------------------------------------------------

## 🤖 Model Details

-   **Algorithm:** Random Forest Classifier\
-   **Problem Type:** Binary Classification\
-   **Input:** Customer service & billing features\
-   **Output:**
    -   Churn Prediction (Yes / No)\
    -   Churn Probability (%)

------------------------------------------------------------------------

## 💻 Web Application

Built using **Flask** and deployment-ready with **Gunicorn**.

### 🔄 User Flow

1.  User enters customer details\
2.  Clicks Submit\
3.  Model processes the input\
4.  Displays:
    -   Prediction Result\
    -   Churn Probability

------------------------------------------------------------------------

## 📂 Project Structure

Customer-Churn-Prediction/
│
├── app.py
├── retrain_model.py   (optional)
├── requirements.txt
├── Procfile
├── README.md
├── model.sav
├── model_columns.pkl
├── first_telc.csv
│
├── templates/
│   └── home.html
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Model_Building.ipynb
------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   Flask\
-   Gunicorn\
-   Pandas\
-   NumPy\
-   Scikit-Learn\
-   HTML & CSS\
-   Pickle

------------------------------------------------------------------------

## 🚀 How To Run Locally

### 1️⃣ Clone the Repository

``` bash
git clone https://github.com/WAQARYOUSUF/TELECOM-RETENTION-THROUGH-CUSTOMER-CHURN-PREDICTION.git
cd TELECOM-RETENTION-THROUGH-CUSTOMER-CHURN-PREDICTION
```

### 2️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application

``` bash
python app.py
```

Open in browser:

http://127.0.0.1:5000

------------------------------------------------------------------------

## 🌐 Deployment (Render)

This project is production-ready.

-   Procfile included\
-   Gunicorn configured\
-   requirements.txt configured\
-   No absolute file paths

To deploy on Render:

Build Command: pip install -r requirements.txt

Start Command: gunicorn app:app

------------------------------------------------------------------------

## 📊 Key Features

✔ Real-time churn prediction\
✔ Probability score calculation\
✔ Clean UI\
✔ Modular ML pipeline\
✔ Retrainable model\
✔ Deployment-ready structure

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Hyperparameter tuning\
-   Cross-validation\
-   Model comparison (Logistic Regression, XGBoost)\
-   Accuracy & confusion matrix display\
-   Cloud database integration

------------------------------------------------------------------------

## 👨‍💻 Author

**Waqar Yousuf**\
B.Tech Major Project\
Customer Churn Prediction System

------------------------------------------------------------------------

## ⭐ Support

If you found this project useful:

⭐ Star the repository\
🍴 Fork the project\
🔗 Share with others
