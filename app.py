import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

h1, h2, h3 {
    color: #00F5FF;
}

.stButton>button {
    background-color: #00F5FF;
    color: black;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🩺 Breast Cancer Prediction using SVM")

st.markdown("---")

# ---------------- LOAD DATA ----------------

data = load_breast_cancer()

X = data.data
y = data.target

features = data.feature_names

df = pd.DataFrame(X, columns=features)

df['target'] = y

# ---------------- DATA PREVIEW ----------------

st.subheader("📊 Dataset Preview")

st.dataframe(df.head())

# ---------------- VISUALIZATION ----------------

st.subheader("📈 Dataset Visualization")

col1, col2 = st.columns(2)

# -------- TARGET DISTRIBUTION --------

with col1:

    fig1, ax1 = plt.subplots(figsize=(4,3))

    df['target'].value_counts().plot(
        kind='bar',
        ax=ax1
    )

    ax1.set_title("Target Distribution")

    ax1.set_xlabel("Class")

    ax1.set_ylabel("Count")

    st.pyplot(fig1)

# -------- CORRELATION --------

with col2:

    correlation = df.corr()['target'].sort_values()

    fig2, ax2 = plt.subplots(figsize=(4,3))

    correlation.plot(
        kind='barh',
        ax=ax2
    )

    ax2.set_title("Feature Correlation")

    st.pyplot(fig2)

# ---------------- OUTLIER REMOVAL ----------------

cleaned_df = df.copy()

for col in cleaned_df.columns[:-1]:

    Q1 = cleaned_df[col].quantile(0.25)

    Q3 = cleaned_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR

    upper_limit = Q3 + 1.5 * IQR

    cleaned_df = cleaned_df[
        (cleaned_df[col] >= lower_limit) &
        (cleaned_df[col] <= upper_limit)
    ]

# ---------------- IMPORTANT FEATURES ----------------

selected_features = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness'
]

# ---------------- FEATURES & TARGET ----------------

X = cleaned_df[selected_features]

y = cleaned_df['target']

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- FEATURE SCALING ----------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------

model = SVC(kernel='linear')

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------

accuracy = accuracy_score(y_test, y_pred)

# ---------------- PERFORMANCE ----------------

st.subheader("📊 Model Performance")

m1, m2 = st.columns(2)

with m1:

    st.metric(
        "Accuracy",
        round(accuracy, 4)
    )

with m2:

    st.metric(
        "Correct Predictions",
        f"{(y_pred == y_test).sum()}/{len(y_test)}"
    )

# ---------------- CONFUSION MATRIX ----------------

st.subheader("📉 Confusion Matrix")

fig3, ax3 = plt.subplots(figsize=(4,3))

cm = confusion_matrix(y_test, y_pred)

ax3.imshow(cm)

ax3.set_title("Confusion Matrix")

ax3.set_xlabel("Predicted")

ax3.set_ylabel("Actual")

st.pyplot(fig3)

# ---------------- CLASSIFICATION REPORT ----------------

st.subheader("🧠 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ---------------- USER INPUT ----------------

st.markdown("---")

st.subheader("🩺 Predict Cancer Type")

c1, c2 = st.columns(2)

input_values = []

with c1:

    mean_radius = st.slider(
        "Mean Radius",
        float(df['mean radius'].min()),
        float(df['mean radius'].max()),
        float(df['mean radius'].mean())
    )

    input_values.append(mean_radius)

    mean_texture = st.slider(
        "Mean Texture",
        float(df['mean texture'].min()),
        float(df['mean texture'].max()),
        float(df['mean texture'].mean())
    )

    input_values.append(mean_texture)

    mean_perimeter = st.slider(
        "Mean Perimeter",
        float(df['mean perimeter'].min()),
        float(df['mean perimeter'].max()),
        float(df['mean perimeter'].mean())
    )

    input_values.append(mean_perimeter)

with c2:

    mean_area = st.slider(
        "Mean Area",
        float(df['mean area'].min()),
        float(df['mean area'].max()),
        float(df['mean area'].mean())
    )

    input_values.append(mean_area)

    mean_smoothness = st.slider(
        "Mean Smoothness",
        float(df['mean smoothness'].min()),
        float(df['mean smoothness'].max()),
        float(df['mean smoothness'].mean())
    )

    input_values.append(mean_smoothness)

# ---------------- INPUT DATAFRAME ----------------

input_df = pd.DataFrame(
    [input_values],
    columns=X.columns
)

# ---------------- SCALE INPUT ----------------

input_scaled = scaler.transform(input_df)

# ---------------- PREDICT ----------------

prediction = model.predict(input_scaled)

result = "Benign (Non-Cancerous)" if prediction[0] == 1 else "Malignant (Cancerous)"

# ---------------- BUTTON ----------------

if st.button("Predict"):

    if prediction[0] == 1:

        st.success(f"🟢 Prediction: {result}")

    else:

        st.error(f"🔴 Prediction: {result}")

# ---------------- FOOTER ----------------

st.markdown("---")

st.info("Breast Cancer Dataset + SVM + Streamlit")