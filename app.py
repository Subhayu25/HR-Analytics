import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, r2_score, mean_squared_error, silhouette_score
)
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance

st.set_page_config(
    layout="wide",
    page_title="HR Analytics Dashboard",
    page_icon="📊"
)

st.title("💼 HR Analytics Dashboard")

# ----------- DATA LOADING -----------
@st.cache_data
def load_data():
    # You can change this line to load from your own file
    df = pd.read_excel("Hr Analytics__.xlsx")
    return df

df = load_data()
st.sidebar.header("🔧 Data Preview & Info")
if st.sidebar.checkbox("Show Raw Data"):
    st.dataframe(df)

# ----------- DATA PREP -----------
# Label encoding & feature setup
df_clean = df.copy()
for col in df_clean.select_dtypes(include="object"):
    df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))

# Drop rows with missing target if exists
if 'Attrition' in df_clean.columns:
    df_clean = df_clean.dropna(subset=['Attrition'])

# Set up features and targets
feature_cols = [col for col in df_clean.columns if col != 'Attrition']
target_col = 'Attrition' if 'Attrition' in df_clean.columns else df_clean.columns[-1]
X = df_clean[feature_cols]
y = df_clean[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------- MAIN TABS -----------
tab1, tab2, tab3 = st.tabs(["Clustering", "Classification", "Regression"])

# ================== CLUSTERING TAB ==================
with tab1:
    st.header("🔹 KMeans Clustering & Silhouette Analysis")
    st.write("Visualize clusters and determine the optimal number of clusters using Silhouette Score.")

    # Range selection for clusters
    range_n_clusters = st.slider("Select range for number of clusters (k):", 2, 10, (2, 6))

    silhouette_scores = []
    for k in range(range_n_clusters[0], range_n_clusters[1]+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
    fig, ax = plt.subplots()
    ax.plot(range(range_n_clusters[0], range_n_clusters[1]+1), silhouette_scores, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score for different k')
    st.pyplot(fig)
    best_k = np.argmax(silhouette_scores) + range_n_clusters[0]
    st.markdown(f"**Best k in range:** {best_k} (Silhouette Score = {np.max(silhouette_scores):.3f})")

    # Show clusters with best_k
    kmeans_final = KMeans(n_clusters=best_k, random_state=42)
    cluster_labels_final = kmeans_final.fit_predict(X_scaled)
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels_final

    st.write("**Cluster Distribution:**")
    st.bar_chart(df_clustered['Cluster'].value_counts())

    if len(feature_cols) >= 2:
        fig2 = px.scatter(df_clustered, x=feature_cols[0], y=feature_cols[1],
                          color='Cluster', title="Cluster Scatterplot (first 2 features)")
        st.plotly_chart(fig2, use_container_width=True)

# ================== CLASSIFICATION TAB ==================
with tab2:
    st.header("🔸 Classification Models & Feature Importance")

    model_option = st.selectbox(
        "Select classifier for feature importance:",
        ("KNN", "Decision Tree", "Random Forest", "Gradient Boosting")
    )
    st.markdown("The chart below shows which features are most important for the selected model.")

    # Prepare models and feature importance
    feature_names = feature_cols

    if model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        importances = result.importances_mean
        title = "Feature Importance (Permutation) - KNN"
    elif model_option == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        title = "Feature Importance - Decision Tree"
    elif model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        title = "Feature Importance - Random Forest"
    elif model_option == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        title = "Feature Importance - Gradient Boosting"

    # Bar chart for feature importance
    fig_imp, ax_imp = plt.subplots()
    sorted_idx = np.argsort(importances)
    ax_imp.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx])
    ax_imp.set_xlabel("Importance")
    ax_imp.set_ylabel("Feature")
    ax_imp.set_title(title)
    st.pyplot(fig_imp)

    # -- Model performance metrics (optional) --
    st.subheader("Model Performance Metrics")
    y_pred = model.predict(X_test)
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.3f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted'):.3f}")
    st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted'):.3f}")
    st.write(f"**F1 Score:** {f1_score(y_test, y_pred, average='weighted'):.3f}")

# ================== REGRESSION TAB ==================
with tab3:
    st.header("🔹 Regression Models")
    st.write("Run and compare regression models for predicting numeric outcomes.")

    # Select a regression target (must be numeric and not the classification label)
    numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    regression_targets = [col for col in numeric_cols if col not in ['Attrition', target_col]]
    regression_target = st.selectbox("Select Regression Target:", regression_targets)

    X_reg = df_clean.drop(columns=[regression_target, target_col])
    y_reg = df_clean[regression_target]
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "Decision Tree Regression": DecisionTreeRegressor(random_state=42)
    }
    results = {}

    for name, model in models.items():
        model.fit(X_train_reg, y_train_reg)
        y_pred_reg = model.predict(X_test_reg)
        r2 = r2_score(y_test_reg, y_pred_reg)
        mse = mean_squared_error(y_test_reg, y_pred_reg)
        results[name] = {"R2": r2, "MSE": mse}

    st.dataframe(pd.DataFrame(results).T.style.format({"R2": "{:.3f}", "MSE": "{:.2f}"}))

    # Optional: Plot feature importance for tree regression
    reg_option = st.selectbox("Show Feature Importance for:", list(models.keys()))
    if reg_option == "Decision Tree Regression":
        tree_reg = DecisionTreeRegressor(random_state=42)
        tree_reg.fit(X_train_reg, y_train_reg)
        importances_reg = tree_reg.feature_importances_
        figr, axr = plt.subplots()
        sorted_idx = np.argsort(importances_reg)
        axr.barh(np.array(X_reg.columns)[sorted_idx], importances_reg[sorted_idx])
        axr.set_title("Feature Importance: Decision Tree Regression")
        st.pyplot(figr)

st.markdown("---")
st.caption("Enhanced HR Analytics Dashboard | Streamlit & scikit-learn")
