#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HR Analytics Streamlit Dashboard
Author  : Group-8 — aided by ChatGPT o4-mini
Version : 2025-07-09  (final)
"""

import warnings
import base64
from typing import List, Tuple, Dict
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from mlxtend.frequent_patterns import apriori, association_rules
import xgboost as xgb

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# ───────────────────── Constants ─────────────────────────
GITHUB_URL            = (
    "https://raw.githubusercontent.com/Subhayu25/HR-Analytics/refs/heads/main/Hr%20Analytics.csv"
)
TARGET_CLASSIFICATION = "Attrition"
DEFAULT_REG_TARGET    = "MonthlyIncome"
RANDOM_STATE          = 42

# ─────────────────── Data Loading ────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load data directly from GitHub raw URL, with error handling."""
    try:
        df = pd.read_csv(GITHUB_URL)
    except HTTPError as e:
        st.error(f"❌ Could not fetch data (HTTP {e.code}): {e.reason}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error loading data:\n{e}")
        st.stop()

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    if TARGET_CLASSIFICATION in df.columns:
        df[TARGET_CLASSIFICATION] = df[TARGET_CLASSIFICATION].astype("category")
    return df

# ───────────────── Helpers ────────────────────────────────
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
            else:
                rng = st.slider(col, lo, hi, (lo, hi))
                df_filt = df_filt[(df_filt[col] >= rng[0]) & (df_filt[col] <= rng[1])]

    with st.sidebar.expander("Categorical Values"):
        for col in categorical:
            opts = df[col].dropna().unique().tolist()
            sel  = st.multiselect(col, opts, default=opts)
            df_filt = df_filt[df_filt[col].isin(sel)]

    return df_filt

def download_link(df: pd.DataFrame, fname: str, label: str) -> str:
    """Return a base-64 download link for a DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{fname}">{label}</a>'

# ────────────── Tab 1: Visualization ─────────────────────
def tab_visualisation(df: pd.DataFrame):
    st.header("📊 Data Visualization")
    numeric, categorical = get_column_types(df)

    # 1) Attrition % by Department
    if "Department" in df.columns and TARGET_CLASSIFICATION in df.columns:
        with st.expander("Attrition % by Department"):
            perc = (pd.crosstab(df["Department"], df[TARGET_CLASSIFICATION], normalize="index") * 100).reset_index()
            st.plotly_chart(
                px.bar(perc, x="Department", y="Yes", labels={"Yes":"Attrition %"}),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Attrition % by Department’—missing column(s)")

    # 2) Age Distribution
    if "Age" in df.columns and TARGET_CLASSIFICATION in df.columns:
        with st.expander("Age Distribution"):
            st.plotly_chart(
                px.histogram(df, x="Age", color=TARGET_CLASSIFICATION, nbins=30, barmode="overlay"),
                use_container_width=True
            )
    else:
        st.warning("⚠️ Skipping ‘Age Distribution’—missing column(s)")

    # 3) Monthly Income vs Job Level
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
        st.warning("⚠️ Skipping ‘Monthly Income vs Job Level’—missing column(s)")

    # 4) Correlation heatmap
    if len(numeric) >= 2:
        with st.expander("Correlation Heat-map"):
            st.plotly_chart(px.imshow(df[numeric].corr(), text_auto=".2f"), use_container_width=True)
    else:
        st.warning("⚠️ Skipping ‘Correlation Heat-map’—not enough numeric columns")

    # 5) Countplots for categoricals (up to 8)
    for col in categorical[:8]:
        if col in df.columns and TARGET_CLASSIFICATION in df.columns:
            with st.expander(f"Countplot – {col}"):
                st.plotly_chart(
                    px.histogram(df, x=col, color=TARGET_CLASSIFICATION, barmode="group"),
                    use_container_width=True
                )
        else:
            st.warning(f"⚠️ Skipping Countplot – {col}")

    # 6) Boxplots for numerics (up to 8)
    for col in numeric[:8]:
        if col in df.columns and TARGET_CLASSIFICATION in df.columns:
            with st.expander(f"Boxplot – {col} by Attrition"):
                st.plotly_chart(px.box(df, y=col, color=TARGET_CLASSIFICATION), use_container_width=True)
        else:
            st.warning(f"⚠️ Skipping Boxplot – {col}")

    st.success("✅ Visualization done.")

# ────────────── Tab 2: Classification ────────────────────
def preprocess(df: pd.DataFrame, target: str):
    num, cat = get_column_types(df.drop(columns=[target]))
    X = df.drop(columns=[target])
    y = df[target]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return X, y, pre

def train_classifiers(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer) -> Dict[str, dict]:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
        "GBRT": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    out = {}
    for name, mdl in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        out[name] = {
            "pipe": pipe,
            "train": {
                "accuracy": accuracy_score(ytr, ptr),
                "precision": precision_score(ytr, ptr, pos_label="Yes", zero_division=0),
                "recall": recall_score(ytr, ptr, pos_label="Yes", zero_division=0),
                "f1": f1_score(ytr, ptr, pos_label="Yes", zero_division=0),
            },
            "test": {
                "accuracy": accuracy_score(yte, pte),
                "precision": precision_score(yte, pte, pos_label="Yes", zero_division=0),
                "recall": recall_score(yte, pte, pos_label="Yes", zero_division=0),
                "f1": f1_score(yte, pte, pos_label="Yes", zero_division=0),
            },
            "proba": pipe.predict_proba(Xte)[:, 1],
            "y_test": yte,
        }
    return out

def tab_classification(df: pd.DataFrame):
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
    cols = pd.MultiIndex.from_product([["Train","Test"], ["Acc","Prec","Rec","F1"]])
    st.dataframe(pd.DataFrame(rows, columns=["Model"]+list(cols)).set_index("Model"),
                 use_container_width=True)

    # Feature importance
    st.subheader("Feature Importance")
    choice = st.selectbox("Choose model:", list(res.keys()))
    pipe = res[choice]["pipe"]
    feats = pipe.named_steps["prep"].get_feature_names_out()

    if choice == "KNN":
        Xi = pipe.named_steps["prep"].transform(X.iloc[res[choice]["y_test"].index])
        perm = permutation_importance(pipe.named_steps["mdl"], Xi, res[choice]["y_test"],
                                      n_repeats=10, random_state=RANDOM_STATE)
        imps = perm.importances_mean
    else:
        mdl = pipe.named_steps["mdl"]
        imps = getattr(mdl, "feature_importances_", np.zeros(len(feats)))

    idx = np.argsort(imps)
    fig = px.bar(x=imps[idx], y=feats[idx], orientation="h", title=f"{choice} importances")
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix & ROC
    sel = st.selectbox("Confusion-matrix model:", list(res.keys()))
    y_t = res[sel]["y_test"]
    y_p = res[sel]["pipe"].predict(X.iloc[y_t.index])
    cm = confusion_matrix(y_t, y_p, labels=["No","Yes"])
    st.plotly_chart(px.imshow(cm, text_auto=True, x=["No","Yes"], y=["No","Yes"]), use_container_width=False)

    roc = go.Figure()
    for name, r in res.items():
        fpr, tpr, _ = roc_curve(r["y_test"].map({"No":0,"Yes":1}), r["proba"])
        roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    roc.add_shape(type="line", x0=0,x1=1,y0=0,y1=1, line=dict(dash="dash"))
    roc.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(roc, use_container_width=True)

    # Batch predict entire df
    best = max(res.items(), key=lambda kv: kv[1]["test"]["f1"])[1]["pipe"]
    df["PredictedAttrition"] = best.predict(df.drop(columns=[TARGET_CLASSIFICATION]))
    st.subheader("🔮 Batch Prediction (Full Data)")
    st.dataframe(df.head(), use_container_width=True)
    st.download_button("Download predictions", df.to_csv(index=False).encode(),
                       "predictions.csv", "text/csv")

# ──────────────── Tab 3: Clustering ───────────────────────
def tab_clustering(df: pd.DataFrame):
    st.header("🕵️ Clustering")
    num, _ = get_column_types(df)
    Xs = StandardScaler().fit_transform(df[num].dropna())

    # Elbow
    inertias = [KMeans(k, random_state=RANDOM_STATE, n_init="auto").fit(Xs).inertia_
                for k in range(1,11)]
    st.plotly_chart(px.line(x=range(1,11), y=inertias, markers=True, title="Elbow"), use_container_width=True)

    # Silhouette
    st.subheader("Silhouette Score")
    kr = st.slider("k-range", 2, 10, (2, 6))
    scores = []
    vals   = range(kr[0], kr[1] + 1)
    for k in vals:
        lbl = KMeans(k, random_state=RANDOM_STATE, n_init="auto").fit_predict(Xs)
        scores.append(silhouette_score(Xs, lbl))
    st.plotly_chart(px.line(x=list(vals), y=scores, markers=True, title="Silhouette vs k"),
                    use_container_width=True)
    best = vals[np.argmax(scores)]
    st.markdown(f"**Best k:** {best} (score={np.max(scores):.3f})")

    k = st.slider("Final k", 2, 10, best)
    km = KMeans(k, random_state=RANDOM_STATE, n_init="auto").fit(Xs)
    df["cluster"] = km.labels_
    persona = df.groupby("cluster").agg({c: ("mean" if c in num else "first") for c in df.columns})
    st.dataframe(persona, use_container_width=True)
    st.markdown(download_link(df, "clusters.csv", "📥 Download clusters"), unsafe_allow_html=True)

# ──────────────── Tab 4: Association Rules ────────────────
def tab_association(df: pd.DataFrame):
    st.header("🔗 Association Rules")
    cols = st.multiselect("Pick 3 categorical cols", df.columns.tolist(),
                          default=["JobRole","MaritalStatus","OverTime"])
    if len(cols) != 3:
        st.warning("Select exactly 3 columns.")
        return

    sup  = st.slider("min_support", 0.01, 0.5, 0.05, 0.01)
    conf = st.slider("min_confidence", 0.01, 1.0, 0.3, 0.01)
    lift = st.slider("min_lift", 0.5, 5.0, 1.0, 0.1)

    hot   = pd.get_dummies(df[cols].astype(str))
    rules = association_rules(apriori(hot, min_support=sup, use_colnames=True),
                              metric="confidence", min_threshold=conf)
    rules = rules[rules["lift"] >= lift].nlargest(10, "confidence")
    st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]])
    st.plotly_chart(px.bar(rules, x=rules.index.astype(str), y="lift", title="Top-10 Lift"),
                    use_container_width=True)

# ──────────────── Tab 5: Regression ──────────────────────
def tab_regression(df: pd.DataFrame):
    st.header("📈 Regression")
    target = st.selectbox("Target", df.select_dtypes("number").columns,
                          index=df.columns.get_loc(DEFAULT_REG_TARGET))
    y = df[target]
    X = df.drop(columns=[target])
    num, cat = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(alpha=0.01),
        "Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)
    }
    scores = []
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("mdl", model)]).fit(X, y)
        scores.append((name, round(pipe.score(X, y), 3)))
        if name in ("Linear", "Ridge", "Lasso"):
            coefs = pd.DataFrame({
                "feature": pipe["prep"].get_feature_names_out(),
                "coef": model.coef_
            })
            coefs = coefs.reindex(coefs["coef"].abs().sort_values(ascending=False).index)
            with st.expander(f"{name} Coefs"):
                st.plotly_chart(px.bar(coefs, x="coef", y="feature", orientation="h"),
                                use_container_width=True)
    st.table(pd.DataFrame(scores, columns=["Model","R²"]).set_index("Model"))

    dt_pipe = Pipeline([("prep", pre), ("mdl", models["Tree"]) ]).fit(X, y)
    preds = dt_pipe.predict(X)
    st.plotly_chart(px.scatter(x=preds, y=y - preds, labels={"x":"Pred","y":"Resid"}, title="Tree Residuals"),
                    use_container_width=True)

# ──────────────── Tab 6: Retention ───────────────────────
def tab_retention(df: pd.DataFrame):
    st.header("⏳ Retention Forecast")
    alg = st.selectbox("Model", ["Logistic", "Forest", "XGBoost"])
    hor = st.slider("Horizon (months)", 6, 24, 12)

    if "YearsAtCompany" not in df:
        st.error("Missing YearsAtCompany")
        return

    tmp = df.copy()
    tmp["Stay"] = (tmp["YearsAtCompany"] * 12 >= hor).astype(int)
    X = tmp.drop(columns=["Stay"])
    y = tmp["Stay"]
    num, cat = get_column_types(X)
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)
    ])

    if alg == "Logistic":
        mdl = LogisticRegression(max_iter=1000)
    elif alg == "Forest":
        mdl = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE)
    else:
        mdl = xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            learning_rate=0.05,
            n_estimators=500,
            max_depth=5
        )

    pipe = Pipeline([("prep", pre), ("mdl", mdl)]).fit(X, y)
    tmp["RetProb"] = pipe.predict_proba(X)[:, 1]
    st.dataframe(tmp[["EmployeeNumber","RetProb"]].head(), use_container_width=True)

    # Feature importance with SHAP fallback
    feat_names = pipe["prep"].get_feature_names_out()
    if alg == "Logistic":
        imp = np.abs(mdl.coef_[0])
    elif hasattr(mdl, "feature_importances_"):
        imp = mdl.feature_importances_
    else:
        Xt = pipe["prep"].transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        expl = shap.Explainer(mdl, Xt)
        sv = expl(Xt[:200])
        imp = np.abs(sv.values).mean(axis=0)

    L = min(len(imp), len(feat_names))
    imp_df = pd.DataFrame({
        "feature": feat_names[:L],
        "importance": imp[:L]
    }).nlargest(15, "importance")
    st.plotly_chart(px.bar(imp_df, x="importance", y="feature", orientation="h"),
                    use_container_width=True)

    st.markdown(download_link(tmp[["EmployeeNumber","RetProb"]], "retention.csv", "📥 Download"),
                unsafe_allow_html=True)

# ─────────────────────── Main ────────────────────────────
def main():
    st.sidebar.title("📂 Data Source")
    df = universal_filters(load_data())
    st.sidebar.download_button("Download filtered CSV",
                               df.to_csv(index=False).encode(),
                               "filtered_data.csv", "text/csv")

    tabs = st.tabs([
        "Visualization", "Classification", "Clustering",
        "Association Rules", "Regression", "Retention"
    ])
    with tabs[0]:
        tab_visualisation(df)
    with tabs[1]:
        tab_classification(df)
    with tabs[2]:
        tab_clustering(df)
    with tabs[3]:
        tab_association(df)
    with tabs[4]:
        tab_regression(df)
    with tabs[5]:
        tab_retention(df)

if __name__ == "__main__":
    main()
