import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import *
from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


st.title("Customer Churn Prediction")


# Load the dataset
df = pd.read_csv("churn.csv")

# Drop the 'customerID' column as it is not relevant for prediction
df.drop(columns=['customerID'], inplace=True)

# Transform 'TotalCharges' to numeric, coercing errors to NaN, and fill NaN values with the median
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
df[['TotalCharges']] = imp.fit(df[['TotalCharges']])
df.info()

# Encode categorical variables using LabelEncoder
le = LabelEncoder()
obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    df[col] = le.fit_transform(df[col])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label mapping for {col}: {le_name_mapping}")

#print(df.head())

# Split the dataset into features (X) and target variable (Y)
X = df.drop(columns=['Churn'])
Y = df['Churn']
with st.spinner("Training Model...", show_time=True):
    for i in range(1, 10):
        total_score = 0
        total_loss = 0
        count = 0
        total_TP = 0
        total_TN = 0
        total_FP = 0
        total_FN = 0        # Split the dataset into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        training_data = X_train.join(Y_train)
        Best_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'threshold': 0.25}
        Best_params.pop("threshold")
        BEST_THRESHOLD = 0.3
        model = RandomForestClassifier(
            **Best_params,
            random_state=42, n_jobs=-1)
        model = model.fit(X_train, Y_train)
        score = model.score(X_test, Y_test)
        score = score * 100
        Y_prob = model.predict_proba(X_test)[:, 1] 
        Y_pred = (Y_prob > BEST_THRESHOLD).astype(int)
        loss = root_mean_squared_error(Y_test, Y_pred)
        matrix = confusion_matrix(Y_test, Y_pred)
        TN = matrix[0][0]
        TP = matrix[1][1]
        FP = matrix[0][1]
        FN = matrix[1][0]
        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN 
        total_score += score
        total_loss += loss
        count += 1
avg_TP = total_TP/count
avg_TN = total_TN/count
avg_FP = total_FP/count
avg_FN = total_FN/count
avg_score = total_score/count
avg_loss = total_loss/count
print(f" Average Model Accuracy: {score:.2f}%")
st.write(f" Average Model Accuracy: {score:.2f}%")
st.write(f" Average Model Loss (RMSE): {loss:.2f}")
matrix= ([[avg_TP, avg_FP],[avg_TN, avg_FN]])
st.write(matrix)


import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(5, 4))
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_aspect('equal')
ax.axis('off')

vals = [
    [avg_TN / (avg_TN + avg_FP), avg_FP / (avg_TN + avg_FP)],
    [avg_FN / (avg_FN + avg_TP), avg_TP / (avg_FN + avg_TP)]
]
colors = [["#4CAF50", "#FF5252"], ["#FF9800", "#2196F3"]]
labels = [["True Negative\n(Correctly kept)", "False Positive\n(Wrong alarm)"],
          ["False Negative\n(Missed churner)", "True Positive\n(Caught churner)"]]

for i in range(2):
    for j in range(2):
        rect = patches.FancyBboxPatch(
            (j + 0.05, 1 - i + 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=colors[i][j], alpha=0.85, edgecolor="white", linewidth=2
        )
        ax.add_patch(rect)
        ax.text(j + 0.5, 1 - i + 0.62, f"{vals[i][j]:.1%}",
        ha='center', va='center', fontsize=18, fontweight='bold', color='white')
        ax.text(j + 0.5, 1 - i + 0.28, labels[i][j],
                ha='center', va='center', fontsize=7.5, color='white', linespacing=1.4)

ax.text(0.5,  2.08, "No Churn", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#4CAF50')
ax.text(1.5,  2.08, "Churn",    ha='center', va='bottom', fontsize=11, fontweight='bold', color='#F44336')
ax.text(-0.22, 1.5, "No\nChurn", ha='center', va='center', fontsize=10, fontweight='bold', color='#4CAF50', rotation=90)
ax.text(-0.22, 0.5, "Churn",     ha='center', va='center', fontsize=10, fontweight='bold', color='#F44336', rotation=90)
ax.text(1.0,  2.22, "Predicted", ha='center', fontsize=12, fontweight='bold', color='gray')
ax.text(-0.5,  1.0, "Actual",    ha='center', fontsize=12, fontweight='bold', color='gray', rotation=90)

st.pyplot(fig)
plt.close()
# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        Y_test,
        display_labels=['Yes', 'No'],
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

#[[913 112][174 210]]

# streamlit form
def get_user_input():
    st.write("Please enter the following customer details:")
    # Create a form for user input
    with st.form(key="user_input"):
        gender = st.selectbox("Gender", options=["Female", "Male"],)
        senior_citizen = st.selectbox("Senior Citizen", options=[0, 1],)
        partner = st.selectbox("Partner", options=["No", "Yes"],)
        dependents = st.selectbox("Dependents", options=["No", "Yes"],)
        tenure = st.slider("Tenure (in months)", min_value=0, max_value=72, value=12)
        phone_service = st.selectbox("Phone Service", options=["No", "Yes"],)
        multiple_lines = st.selectbox("Multiple Lines", options=["No", "Yes", "No phone service"],)
        internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"],)
        online_security = st.selectbox("Online Security", options=["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", options=["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", options=["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", options=["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", options=["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", options=["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", options=["No", "Yes"])
        payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.slider("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.slider("Total Charges", min_value=0.0, max_value=10000.0, value=2000.0)
        st.form_submit_button('Submit')
    st.write(f"The customer you entered is a {gender} and has a tenure of {tenure} months.")
    # ── Encode inputs ─────────────────────────────────────────────────────────────
    gender_enc            = 1 if gender == "Male" else 0
    partner_enc           = 1 if partner == "Yes" else 0
    dependents_enc        = 1 if dependents == "Yes" else 0
    phone_service_enc     = 1 if phone_service == "Yes" else 0
    paperless_billing_enc = 1 if paperless_billing == "Yes" else 0

    multiple_lines_enc    = {"No": 0, "No phone service": 1, "Yes": 2}[multiple_lines]
    internet_service_enc  = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
    online_security_enc   = {"No": 0, "No internet service": 1, "Yes": 2}[online_security]
    online_backup_enc     = {"No": 0, "No internet service": 1, "Yes": 2}[online_backup]
    device_protection_enc = {"No": 0, "No internet service": 1, "Yes": 2}[device_protection]
    tech_support_enc      = {"No": 0, "No internet service": 1, "Yes": 2}[tech_support]
    streaming_tv_enc      = {"No": 0, "No internet service": 1, "Yes": 2}[streaming_tv]
    streaming_movies_enc  = {"No": 0, "No internet service": 1, "Yes": 2}[streaming_movies]
    contract_enc          = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    payment_method_enc    = {"Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3}[payment_method]

    # ── Build tuple ───────────────────────────────────────────────────────────────
    user_input = (
        gender_enc, senior_citizen, partner_enc, dependents_enc, tenure,
        phone_service_enc, multiple_lines_enc, internet_service_enc,
        online_security_enc, online_backup_enc, device_protection_enc,
        tech_support_enc, streaming_tv_enc, streaming_movies_enc,
        contract_enc, paperless_billing_enc, payment_method_enc, monthly_charges, total_charges
    )
    return user_input

# ── Run & predict ─────────────────────────────────────────────────────────────
user_input = get_user_input()
with st.spinner("Inference from Model...", show_time=True):
    prediction = model.predict([user_input])
    prediction_proba = model.predict_proba([user_input])

no  = prediction_proba[0][0]  # index 0 = class 0 = No
yes = prediction_proba[0][1]  # index 1 = class 1 = Yes

print(f"Probability of Churn:     {yes:.2f}")
print(f"Probability of Not Churn: {no:.2f}")
print("\n🔮 Churn Prediction:", "Yes ⚠️" if prediction[0] == 1 else "No ✅")
st.write(f"🔮 Churn Prediction: {'Yes ⚠️' if prediction[0] == 1 else 'No ✅'}")