import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AtmoPredict", layout="wide")

# ----------------- STYLING -----------------
st.markdown("""
    <style>
    .nav-container { display: flex; justify-content: center; gap: 40px; font-size: 20px; }
    .nav-item { padding: 8px 20px; border-radius: 10px; background-color: #4CAF50; color: white;
                text-decoration: none; font-weight: bold; transition: background-color 0.3s; }
    .nav-item:hover { background-color: #45a049; }
    .info-box { border: 2px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px;
                background-color: #f9f9f9; }
    .info-title { font-size: 20px; font-weight: bold; color: #333; }
    </style>
""", unsafe_allow_html=True)

# ----------------- NAVIGATION -----------------
page = st.session_state.get("page", "About Project")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected = st.radio("", ["About Project", "Run Model", "Key Outcomes"], horizontal=True)
st.session_state.page = selected

# ----------------- LOAD DATA -----------------
file_path = os.path.join(os.path.dirname(__file__), "weatherHistory.csv")

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error(f"weatherHistory.csv not found in {os.path.dirname(__file__)}")
    st.stop()

df.columns = [c.strip().lower() for c in df.columns]
required_cols = {
    "temperature": "apparent temperature (c)",
    "humidity": "humidity",
    "windspeed": "wind speed (km/h)",
    "precip": "precip type"
}

try:
    X = df[[required_cols["temperature"], required_cols["humidity"], required_cols["windspeed"]]]
    y = df[required_cols["precip"]]
except KeyError:
    st.error("Dataset must contain: Apparent Temperature (C), Humidity, Wind Speed (km/h), Precip Type")
    st.stop()

y = y.fillna("Sunny")
X = X.fillna(X.mean())

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

models = {
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(kernel='linear', max_iter=2000)  # Faster SVM
}

def run_model(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    labels_present = unique_labels(y_test, y_pred)
    target_names = [str(le.classes_[i]) for i in labels_present]
    report = classification_report(y_test, y_pred, labels=labels_present, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="coolwarm", xticklabels=target_names, yticklabels=target_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name} — Confusion Matrix', fontsize=16)
    st.pyplot(fig)
    return acc, report

# ----------------- PAGES -----------------
if selected == "About Project":
    st.title("🌦️ AtmoPredict — Weather Classification")
    st.markdown("""
        **AtmoPredict** predicts precipitation type based on Apparent Temperature, Humidity, and Wind Speed.  
        It compares **Naive Bayes**, **KNN**, and **SVM** classifiers to identify the most effective model.
    """)

    st.subheader("📂 Dataset Information")
    st.markdown("""
        - **Source:** Historical weather dataset from Kaggle  
        - **Features:** Apparent Temperature (°C), Humidity, Wind Speed (km/h)  
        - **Target:** Precipitation Type (Rain, Snow, Sunny)
    """)

    st.subheader("⚙️ Workflow")
    st.markdown("""
        1. Preprocess data: handle missing values & scale features  
        2. Encode labels to numeric form  
        3. Train models (Naive Bayes, KNN, SVM)  
        4. Evaluate performance with accuracy, classification report, and confusion matrix
    """)

    st.subheader("📌 Algorithms Used")

    # Naive Bayes
    st.markdown("""
    ### 1. Naive Bayes
    **Definition:** Probabilistic classifier based on Bayes’ Theorem assuming feature independence.  
    **Formula:** 
    $$
    P(C|X) = \\frac{P(X|C)P(C)}{P(X)}
    $$
    **Advantages:**
    - Fast & efficient
    - Works well with small datasets 
                 
    **Disadvantages:**
    - Assumes feature independence
    - Can perform poorly if features are strongly correlated
    """)

    # KNN
    st.markdown("""
    ### 2. K-Nearest Neighbors (KNN)
    **Definition:** Classifies a data point based on the majority label of its *k* nearest neighbors.  
    **Formula:** Euclidean Distance:  
    $$
    d(p, q) = \\sqrt{\\sum_{i=1}^{n} (p_i - q_i)^2}
    $$
    **Advantages:**
    - Simple & intuitive  
    - No explicit training phase 
                 
    **Disadvantages:**
    - Slow for large datasets  
    - Sensitive to irrelevant features & noise
    """)

    # SVM
    st.markdown("""
    ### 3. Support Vector Machine (SVM)
    **Definition:** Finds an optimal hyperplane that maximizes the margin between classes.  
    **Formula:** Decision Function:  
    $$
    f(x) = sign(w \\cdot x + b)
    $$
    **Advantages:**
    - Works well in high-dimensional spaces  
    - Effective when classes are separable  
                
    **Disadvantages:**
    - Slower on large datasets  
    - Requires careful parameter & kernel choice
    """)

elif selected == "Run Model":
    st.title("⚙️ Run Weather Prediction Model")
    model_choice = st.selectbox("Choose a model", list(models.keys()))
    if st.button("Run Model"):
        acc, report = run_model(model_choice, models[model_choice])
        st.write(f"**Accuracy:** {acc*100:.2f}%")
        st.text(report)

elif selected == "Key Outcomes":
    st.title("📊 Key Outcomes & Real-Life Applications")
    st.write("""
    ### Key Outcomes:
    - SVM now runs faster with linear kernel
    - Naive Bayes is fastest for quick predictions
    - KNN is effective but slower with large datasets

    ### Real-Life Applications:
    - Weather forecasting
    - Agriculture planning
    - Transportation safety
    - Event scheduling
    """)