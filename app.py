#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard
Author  : Group-8 — aided by ChatGPT o4-mini
Version : 2025-07-08  (final)
"""

# ───────────────────────── Imports ──────────────────────────
import warnings
import base64
from typing import List, Tuple, Dict, Union
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, silhouette_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, LogisticRegression,
)
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from mlxtend.frequent_patterns import apriori, association_rules
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)
st.set_page_config(page_title="HR Analytics Dashboard",
                   page_icon=":bar_chart:", layout="wide")

# ────────────────────── Constants ───────────────────────────
# Replace <your-username> and <your-repo> with your actual GitHub details,
# and ensure spaces in filenames are percent-encoded (%20)
GITHUB_URL            = (
    "https://raw.githubusercontent.com/Subhayu25/BookMyShow/refs/heads/main/BookMyShow_Combined_Clean_v2.csv"
)
TARGET_CLASSIFICATION = "Attrition"
DEFAULT_REG_TARGET    = "MonthlyIncome"
RANDOM_STATE          = 42

# ────────────────────── Helpers ─────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load data directly from GitHub raw URL, with error handling."""
    try:
        df = pd.read_csv(GITHUB_URL)
    except HTTPError as e:
        st.error(f"❌ Failed to load data from GitHub (HTTP {e.code}): {e.reason}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading data:\n{e}")
        st.stop()

    if TARGET_CLASSIFICATION in df.columns:
        df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype("category")
    return df

def get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric     = df.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in df.columns if c not in numeric]
    return numeric, categorical

def universal_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Sidebar widgets that filter the DataFrame."""
    st.sidebar.markdown("### Universal Filters")
    numeric, categorical = get_column_types(df)
    df_filt = df.copy()

    with st.sidebar.expander("Numeric Ranges"):
        for col in numeric:
            lo, hi = float(df[col].min()), float(df[col].max())
            if lo == hi:
                st.number_input(col, value=lo, disabled=True)
                continue
            rng = st.slider(col, lo, hi, (lo, hi))
            df_filt = df_filt[(df_filt[col] >= rng[0]) & (df_filt[col] <= rng[1])]

    with st.sidebar.expander("Categorical Values"):
        for col in categorical:
            opts = df[col].dropna().unique().tolist()
            sel  = st.multiselect(col, opts, default=opts)
            df_filt = df_filt[df_filt[col].isin(sel)]

    return df_filt

def download_link(obj, filename: str, label: str) -> str:
    """Return a base-64 download link for a DataFrame."""
    text = obj.to_csv(index=False) if isinstance(obj, pd.DataFrame) else obj
    b64  = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'

# ─────────────────── Tab 1 – Visualization ──────────────────
def tab_visualisation(df: pd.DataFrame) -> None:
    st.header("📊 Data Visualization")
    numeric, categorical = get_column_types(df)

    # 1) Attrition % by Department
    if "Department" in df.columns and TARGET_CLASSIFICATION in df.columns:
        with st.expander("Attrition % by Department"):
            perc = (pd.crosstab(
                        df["Department"],
                        df[TARGET_CLASSIFICATION],
                        normalize="index"
                    ) * 100).reset_index()
            st.plotly_chart(
                px.bar(perc, x="Department", y="Yes", labels={"Yes":"Attrition %"}),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Attrition % by Department’ – columns not found.")

    # 2) Age Distribution
    if "Age" in df.columns and TARGET_CLASSIFICATION in df.columns:
        with st.expander("Age Distribution"):
            st.plotly_chart(
                px.histogram(df, x="Age", color=TARGET_CLASSIFICATION,
                             nbins=30, barmode="overlay"),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Age Distribution’ – columns not found.")

    # 3) Monthly Income vs JobLevel
    if "JobLevel" in df.columns and "MonthlyIncome" in df.columns:
        with st.expander("Monthly Income vs Job Level"):
            st.plotly_chart(
                px.violin(
                    df, x="JobLevel", y="MonthlyIncome",
                    color=TARGET_CLASSIFICATION, box=True, points="outliers"
                ),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Monthly Income vs Job Level’ – columns not found.")

    # 4) Correlation heatmap (only numeric cols)
    if len(numeric) >= 2:
        with st.expander("Correlation Heat-map"):
            st.plotly_chart(
                px.imshow(df[numeric].corr(), text_auto=".2f"),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Correlation Heat-map’ – not enough numeric columns.")

    # 5) Auto generated countplots for up to 8 categoricals
    for col in categorical[:8]:
        if col in df.columns and TARGET_CLASSIFICATION in df.columns:
            with st.expander(f"Countplot – {col}"):
                st.plotly_chart(
                    px.histogram(df, x=col, color=TARGET_CLASSIFICATION,
                                 barmode="group"),
                    use_container_width=True
                )
        else:
            st.warning(f"⚠️ Skipping Countplot – {col}")

    # 6) Auto generated boxplots for up to 8 numerics
    for col in numeric[:8]:
        if col in df.columns and TARGET_CLASSIFICATION in df.columns:
            with st.expander(f"Boxplot – {col} by {TARGET_CLASSIFICATION}"):
                st.plotly_chart(
                    px.box(df, y=col, color=TARGET_CLASSIFICATION),
                    use_container_width=True
                )
        else:
            st.warning(f"⚠️ Skipping Boxplot – {col}")

    st.success("✅ Finished rendering available visual insights.")

# ──────────────── Tab 2 – Classification ───────────────────
def preprocess(df: pd.DataFrame, target: str):
    num, cat = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre

def train_classifiers(X, y, pre) -> Dict[str, Dict]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300,
                                                random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    out = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", model)]).fit(X_tr, y_tr)
        p_tr, p_te = pipe.predict(X_tr), pipe.predict(X_te)
        out[name] = {
            "pipe": pipe,
            "train": {
                "accuracy": accuracy_score(y_tr, p_tr),
                "precision": precision_score(y_tr, p_tr, pos_label="Yes", zero_division=0),
                "recall":    recall_score(y_tr, p_tr, pos_label="Yes", zero_division=0),
                "f1":        f1_score(y_tr, p_tr, pos_label="Yes", zero_division=0),
            },
            "test": {
                "accuracy": accuracy_score(y_te, p_te),
                "precision": precision_score(y_te, p_te, pos_label="Yes", zero_division=0),
                "recall":    recall_score(y_te, p_te, pos_label="Yes", zero_division=0),
                "f1":        f1_score(y_te, p_te, pos_label="Yes", zero_division=0),
            },
            "proba": pipe.predict_proba(X_te)[:, 1],
            "y_test": y_te,
        }
    return out

def tab_classification(df: pd.DataFrame) -> None:
    st.header("🤖 Classification")
    X, y, pre = preprocess(df, TARGET_CLASSIFICATION)
    res = train_classifiers(X, y, pre)

    # Performance table
    rows = []
    for name, r in res.items():
        rows.append([
            name,
            r["train"]["accuracy"], r["train"]["precision"],
            r["train"]["recall"],   r["train"]["f1"],
            r["test"]["accuracy"],  r["test"]["precision"],
            r["test"]["recall"],    r["test"]["f1"],
        ])
    cols = pd.MultiIndex.from_product(
        [["Train", "Test"], ["Accuracy", "Precision", "Recall", "F1"]]
    )
    st.dataframe(
        pd.DataFrame(rows, columns=["Model"] + list(cols))
          .set_index("Model"),
        use_container_width=True
    )

    # Feature importance
    st.subheader("Feature Importance Comparison")
    feat_mdl = st.selectbox("Select classifier:", list(res.keys()))
    pipe = res[feat_mdl]["pipe"]
    feature_names = pipe.named_steps["prep"].get_feature_names_out()

    if feat_mdl == "KNN":
        X_te = pipe.named_steps["prep"].transform(X.iloc[res[feat_mdl]["y_test"].index])
        pi = permutation_importance(pipe.named_steps["mdl"],
                                    X_te, res[feat_mdl]["y_test"],
                                    n_repeats=10, random_state=RANDOM_STATE)
        importances = pi.importances_mean
    else:
        mdl = pipe.named_steps["mdl"]
        importances = getattr(mdl, "feature_importances_", np.zeros(len(feature_names)))

    idx = np.argsort(importances)
    imp_df = pd.DataFrame({
        "feature": feature_names[idx],
        "importance": importances[idx]
    })
    fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h",
                     title=f"Feature Importance: {feat_mdl}")
    st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion matrix & ROC
    sel = st.selectbox("Confusion-matrix model:", list(res.keys()))
    y_true = res[sel]["y_test"]
    y_pred = res[sel]["pipe"].predict(X.iloc[y_true.index])
    cm = confusion_matrix(y_true, y_pred, labels=["No", "Yes"])
    st.plotly_chart(px.imshow(cm, text_auto=True,
                              x=["No", "Yes"], y=["No", "Yes"]),
                    use_container_width=False)

    roc_fig = go.Figure()
    for name, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No":0, "Yes":1}), r["proba"])
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    roc_fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1,
                      line=dict(dash="dash"))
    roc_fig.update_layout(title="ROC Curves",
                          xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(roc_fig, use_container_width=True)

    # Batch prediction
    best = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
    df["PredictedAttrition"] = best.predict(df.drop(columns=[TARGET_CLASSIFICATION]))
    st.subheader("🔮 Batch Prediction (entire dataset)")
    st.dataframe(df.head(), use_container_width=True)
    st.download_button("Download predictions",
                       df.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")

# ──────────────── Tab 3 – Clustering ────────────────────────
def tab_clustering(df: pd.DataFrame) -> None:
    st.header("🕵️‍♀️ Clustering")
    numeric, _ = get_column_types(df)
    X_scaled  = StandardScaler().fit_transform(df[numeric].dropna())

    # Elbow method
    inertias = [
        KMeans(k, n_init="auto", random_state=RANDOM_STATE)
        .fit(X_scaled).inertia_
        for k in range(1, 11)
    ]
    st.plotly_chart(px.line(
        x=range(1, 11), y=inertias, markers=True,
        labels={"x":"k", "y":"Inertia"}, title="Elbow Method"
    ), use_container_width=True)

    # Silhouette analysis
    st.subheader("Silhouette Score for Optimal Clusters")
    sil_range = st.slider("Select k-range", 2, 10, (2, 6))
    silhouette_scores = []
    k_vals = range(sil_range[0], sil_range[1] + 1)
    for k in k_vals:
        lbls = KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, lbls))
    st.plotly_chart(px.line(
        x=list(k_vals), y=silhouette_scores, markers=True,
        labels={"x":"k", "y":"Silhouette Score"}, title="Silhouette vs k"
    ), use_container_width=True)
    best_k = k_vals[np.argmax(silhouette_scores)]
    st.markdown(f"**Best k:** {best_k} (Score={np.max(silhouette_scores):.3f})")

    # Final clustering
    k = st.slider("Choose final k", 2, 10, best_k)
    km = KMeans(k, n_init="auto", random_state=RANDOM_STATE).fit(X_scaled)
    df["cluster"] = km.labels_
    persona = df.groupby("cluster").agg(
        {c: ("mean" if c in numeric else "first") for c in df.columns}
    )
    st.dataframe(persona, use_container_width=True)
    st.markdown(download_link(df, "clustered_data.csv", "📥 Download clusters"),
                unsafe_allow_html=True)

# ──────────────── Tab 4 – Association Rules ────────────────
def tab_association(df: pd.DataFrame) -> None:
    st.header("🔗 Association Rule Mining")
    cols = st.multiselect("Pick 3 categorical columns",
                          df.columns.tolist(),
                          default=["JobRole", "MaritalStatus", "OverTime"])
    if len(cols) != 3:
        st.warning("Exactly three columns required.")
        return

    sup  = st.slider("min_support",    0.01, 0.5, 0.05, 0.01)
    conf = st.slider("min_confidence", 0.01, 1.0, 0.30, 0.01)
    lift = st.slider("min_lift",       0.50, 5.0, 1.00, 0.10)

    hot   = pd.get_dummies(df[cols].astype(str))
    rules = association_rules(
        apriori(hot, min_support=sup, use_colnames=True),
        metric="confidence", min_threshold=conf
    )
    rules = rules[rules["lift"] >= lift] \
                 .sort_values("confidence", ascending=False) \
                 .head(10)
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    st.plotly_chart(px.bar(
        rules, x=rules.index.astype(str), y="lift",
        title="Lift of Top-10 Rules"
    ), use_container_width=True)

# ──────────────── Tab 5 – Regression ───────────────────────
def tab_regression(df: pd.DataFrame) -> None:
    st.header("📈 Regression Insights")
    target = st.selectbox("Target variable",
                          df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET))
    y = df[target]; X = df.drop(columns=[target])
    num, cat = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)
    }
    scores = []
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", model)]).fit(X, y)
        scores.append((name, round(pipe.score(X, y), 3)))
        if name in ("Linear", "Ridge", "Lasso"):
            coefs = pd.DataFrame({
                "feature": pipe["prep"].get_feature_names_out(),
                "coef": model.coef_
            }).sort_values("coef", key=np.abs, ascending=False).head(10)
            with st.expander(f"{name} – Top coefficients"):
                st.plotly_chart(px.bar(coefs, x="coef", y="feature", orientation="h"),
                                use_container_width=True)
    st.table(pd.DataFrame(scores, columns=["Model","R²"]).set_index("Model"))

    dt_pipe = Pipeline([("prep", pre), ("mdl", models["Decision Tree"]) ]).fit(X, y)
    preds = dt_pipe.predict(X)
    st.plotly_chart(px.scatter(x=preds, y=y - preds,
                               labels={"x":"Predicted","y":"Residual"},
                               title="Residuals – Decision Tree"),
                    use_container_width=True)

# ──────────────── Tab 6 – Retention ────────────────────────
def tab_retention(df: pd.DataFrame) -> None:
    st.header("⏳ 12-Month Retention Forecast")
    alg = st.selectbox("Model", ["Logistic Regression", "Random Forest", "XGBoost"])
    horizon = st.slider("Horizon (months)", 6, 24, 12)

    if "YearsAtCompany" not in df:
        st.error("Column `YearsAtCompany` missing.")
        return

    tmp = df.copy()
    tmp["Stay"] = (tmp["YearsAtCompany"] * 12 >= horizon).astype(int)
    X = tmp.drop(columns=["Stay"])
    y = tmp["Stay"]
    num, cat = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    if alg == "Logistic Regression":
        mdl = LogisticRegression(max_iter=1000)
    elif alg == "Random Forest":
        mdl = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)
    else:
        mdl = xgb.XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss",
            learning_rate=0.05, n_estimators=500, max_depth=5
        )

    pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(X, y)
    tmp["RetentionProb"] = pipe.predict_proba(X)[:, 1]
    st.dataframe(tmp[["EmployeeNumber","RetentionProb"]].head(),
                 use_container_width=True)

    st.subheader("Feature importance")
    feat_names = pipe["prep"].get_feature_names_out()
    if alg == "Logistic Regression":
        importance = np.abs(mdl.coef_[0])
    elif hasattr(mdl, "feature_importances_"):
        importance = mdl.feature_importances_
    else:
        X_trans = pipe["prep"].transform(X)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        explainer = shap.Explainer(mdl, X_trans)
        shap_vals = explainer(X_trans[:200])
        importance = np.abs(shap_vals.values).mean(axis=0)

    if len(importance) != len(feat_names):
        min_len = min(len(importance), len(feat_names))
        importance = importance[:min_len]
        feat_names = feat_names[:min_len]

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importance
    }).sort_values("importance", ascending=False).head(15)
    st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h"),
                    use_container_width=True)

    st.markdown(
        download_link(tmp[["EmployeeNumber","RetentionProb"]],
                      "retention_predictions.csv",
                      "📥 Download predictions"),
        unsafe_allow_html=True
    )

# ─────────────────────────── Main ───────────────────────────
def main() -> None:
    st.sidebar.title("📂 Data Source")
    df = universal_filters(load_data())
    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv", "text/csv")

    tabs = st.tabs([
        "Visualization", "Classification", "Clustering",
        "Association Rules", "Regression", "12-Month Forecast"
    ])
    with tabs[0]: tab_visualisation(df)
    with tabs[1]: tab_classification(df)
    with tabs[2]: tab_clustering(df)
    with tabs[3]: tab_association(df)
    with tabs[4]: tab_regression(df)
    with tabs[5]: tab_retention(df)

if __name__ == "__main__":
    main()
