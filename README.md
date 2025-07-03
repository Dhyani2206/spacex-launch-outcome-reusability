# SpaceX Launch Outcome Prediction and Reusability Impact Analysis

This project analyzes and predicts the outcomes of SpaceX rocket launches using machine learning and statistical analysis.  
It also investigates the impact of **reusability** on mission success and payload performance, providing valuable insights into spaceflight reliability and emerging reusable technologies.

---

##  Project Overview
- **Objective**:
  - Predict SpaceX launch success using machine learning models.
  - Analyze the role of booster reusability in launch outcomes and payload capacity.

- **Dataset**:
  - SpaceX Launch Data (186 rows, 10 columns)
  - Key features: Mission Name, Rocket Type, Reusability, Payload Mass, Payload Type, Orbit, Country, Launch Success, Launch Date, Launch Year.

---

##  Project Highlights
- **Class Imbalance Handling**: SMOTE and SMOTETomek were used for balancing.
- **Models Used**:
  - **Linear Models**: Logistic Regression, Ridge Classifier
  - **Tree-Based Models**: Decision Tree, Random Forest
  - **Boosting Models**: XGBoost, AdaBoost, LightGBM
- **Evaluation**: Accuracy, ROC AUC, Confusion Matrix, Feature Importance.

---

##  Reusability Impact Insights:
- **100% Success Rate for Reusable Rockets** in this dataset.
- Reusable boosters carry **lighter payloads** (statistically significant difference).
- Reusability adoption started around **2017** and continues to rise.
- Reusability was a **highly influential feature** in predictive models.

---

##  Dashboard Features:
- Model performance comparison (Confusion Matrix, ROC Curves, Feature Importance).
- Interactive analysis of reusability impact.
- Country-wise launch analysis.

---

##  How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run your_dashboard_file.py
