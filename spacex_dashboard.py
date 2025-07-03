import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="SpaceX Launch Analysis", layout="wide")

# === Sidebar Navigation ===
section = st.sidebar.radio("Navigation", [
    " Model Evaluation",
    " Reusability Impact",
    " Country Analysis",
    " Project Summary"
])

# === Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv("spacex_dataset.csv")
    df['Launch Date'] = pd.to_datetime(df['launch_date'])
    df['Launch Year'] = df['Launch Date'].dt.year
    return df

df = load_data()

# =================== 1. MODEL EVALUATION ===================
if section == " Model Evaluation":
    st.title("SpaceX Launch Success Prediction Dashboard")
    st.markdown("""
    This dashboard compares the performance of various machine learning models for predicting the success of SpaceX launches.
    Models are evaluated using accuracy, AUC, confusion matrix, ROC curve, and feature importance.
    """)

    # Model performance summary
    if os.path.exists("model_comparison_results.csv"):
        results_df = pd.read_csv("model_comparison_results.csv")
        st.header(" Model Performance Summary")
        st.dataframe(results_df)
    else:
        st.warning(" model_comparison_results.csv not found.")

    model_names = [
        "Logistic Regression", "Ridge Classifier",
        "Decision Tree", "Random Forest",
        "XGBoost", "AdaBoost", "LightGBM"
    ]

    # Confusion Matrices
    st.header("Confusion Matrices")
    for model_name in model_names:
        st.subheader(model_name)
        img_path = f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
        if os.path.exists(img_path):
            st.image(img_path)
        else:
            st.warning(f"Confusion matrix not found: {img_path}")

    # ROC Curves
    st.header("ROC Curves")
    for model_name in model_names:
        if model_name != "Ridge Classifier":
            st.subheader(model_name)
            img_path = f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
            if os.path.exists(img_path):
                st.image(img_path)
            else:
                st.warning(f"ROC curve not found: {img_path}")

    # Feature Importance
    st.header(" Feature Importances")
    for model_name in ["Decision Tree", "Random Forest", "XGBoost", "AdaBoost", "LightGBM"]:
        st.subheader(model_name)
        img_path = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        if os.path.exists(img_path):
            st.image(img_path)
        else:
            st.warning(f"Feature importance not found: {img_path}")

# =================== 2. REUSABILITY IMPACT ===================
elif section == " Reusability Impact":
    st.title(" Reusability Impact Analysis")

    # Launch success by reusability
    st.subheader("Launch Success by Booster Reusability")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x='Reusability', hue='Launch Success', ax=ax1)
    ax1.set_title('Launch Success by Booster Reusability')
    st.pyplot(fig1)
    st.markdown("- Reusable boosters show a 100% success rate, though used less frequently.\n- Non-reusable boosters had a ~97% success rate.")

    # Payload Mass Distribution
    st.subheader("Payload Mass Comparison")
    fig2, ax2 = plt.subplots()
    sns.boxplot(data=df, x='Reusability', y='Payload Mass (kg)', ax=ax2)
    ax2.set_title('Payload Mass by Booster Reusability')
    st.pyplot(fig2)

    from scipy.stats import ttest_ind
    reusable = df[df['Reusability'] == True]['Payload Mass (kg)'].dropna()
    non_reusable = df[df['Reusability'] == False]['Payload Mass (kg)'].dropna()
    t_stat, p_val = ttest_ind(reusable, non_reusable, equal_var=False)
    st.markdown(f"- T-test p-value: `{p_val:.4f}` → {'Significant' if p_val < 0.05 else 'Not significant'} difference.\n- Reusable boosters carry lighter payloads.")

    # Reusability trend over years
    st.subheader("Reusability Over Time")
    trend = df.groupby('Launch Year')['Reusability'].mean()
    fig3, ax3 = plt.subplots()
    trend.plot(marker='o', ax=ax3)
    ax3.set_ylabel("Proportion of Reusable Boosters")
    ax3.set_title("Trend of Reusability Over Years")
    st.pyplot(fig3)

    # Summary metrics
    total = df.shape[0]
    reusable_count = df[df['Reusability'] == True].shape[0]
    percent = (reusable_count / total) * 100
    st.markdown(f"**Reusable Launches**: `{reusable_count}` out of `{total}` ({percent:.2f}%)")

# =================== 3. COUNTRY ANALYSIS ===================
elif section == " Country Analysis":
    st.title(" Launches by Country")
    country_counts = df['Country'].value_counts()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=country_counts.values, y=country_counts.index, ax=ax4)
    ax4.set_title("Launch Count by Country")
    st.pyplot(fig4)
    st.markdown("- USA dominates the launch count.\n- Shows SpaceX’s centralized operations.")

# =================== 4. PROJECT SUMMARY ===================
st.title(" Project Summary")

st.markdown("""
###  Project Title:
**SpaceX Launch Outcome Prediction and Reusability Impact Analysis**

---

###  Definition:
This project aims to analyze and predict the outcome of SpaceX rocket launches using a combination of statistical analysis and machine learning.  
With growing interest in space commercialization and reusable rocket technology, the study investigates the role of reusability in mission success and payload performance.

---

###  Dataset Overview:
- **Source**: SpaceX launch data  
- **Size**: 186 rows * 10 columns  
- **Key Features**:
  - `mission_name`, `Rocket Type`, `Reusability`, `Payload Mass (kg)`, `Payload Type`, `Orbit`, `Country`, `Launch Success`, `launch_date`, `launch_year`  
- **Highlights**:
  - `Launch Success`: binary (1 = success, 0 = failure)  
  - `Reusability`: boolean (True = reused booster)

---

###  Steps Performed:

**1. Data Preprocessing**
- Verified no missing values
- Identified class imbalance (successes >> failures)

**2. Feature Analysis**
- Categorical analysis of `Reusability`, `Orbit`, `Rocket Type`
- Visualized launch success patterns

**3. Model Training & Evaluation**
- Trained 7 models across 3 categories:
  - *Linear*: Logistic Regression, Ridge Classifier  
  - *Tree-Based*: Decision Tree, Random Forest  
  - *Boosting*: XGBoost, AdaBoost, LightGBM  
- Evaluated using: accuracy, ROC curves, AUC, confusion matrices, and feature importance

---

###  Techniques Used:
- **Class Imbalance Handling**: Used SMOTE and SMOTETomek to generate synthetic samples and clean boundary overlaps  
- **Model Evaluation**: Balanced metrics used to address class imbalance (AUC, F1-score)

---

###  Reusability Impact Analysis:

- **100% Success for Reusable Rockets**  
  All launches using reusable boosters were successful in this dataset.
  
- **Lower Success in Non-Reusable Rockets**  
  Some non-reusable rockets failed, contributing to model learning.

- **Payload Mass Impact**  
  Reusable boosters carried lighter payloads — a statistically significant difference.

- **Reusability Over Time**  
  Reusability adoption began around 2017 and gradually increased.

- **Influence on Models**  
  Reusability emerged as a **highly influential feature**, boosting model accuracy and AUC scores.

---

###  Conclusion:
This project demonstrates that machine learning can accurately predict launch outcomes and that reusability is a strong indicator of mission success. The dashboard provides both predictive insights and interpretability for strategic decisions in space missions.
""")

