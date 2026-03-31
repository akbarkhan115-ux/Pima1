import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Custom transformer needed for joblib deserialization
class LogTransformer:
    """Custom log1p transformer with fit/transform API."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(np.abs(X))
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

# --- Page Config ---
st.set_page_config(page_title="Pima Diabetes Predictor", page_icon="🩺", layout="wide")

# --- Load artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("feature_names.pkl")
    df = pd.read_csv("Pima  (1).csv")
    with open("model_results.json", "r") as f:
        data = json.load(f)
    return model, scaler, features, df, data

model, scaler, feature_names, df, results_data = load_artifacts()

all_results = results_data["all_results"]
scaler_best = results_data["scaler_best"]
sampler_best = results_data.get("sampler_best", {})
global_best = results_data["global_best"]

# --- Sidebar ---
st.sidebar.title("🩺 Pima Diabetes Predictor")
page = st.sidebar.radio("Navigate", [
    "Predict",
    "Sampling Comparison",
    "Preprocessing Comparison",
    "Algorithm Comparison",
    "Model Performance",
    "Dataset Explorer",
])

# ===================== PREDICT PAGE =====================
if page == "Predict":
    st.title("Diabetes Risk Prediction")
    st.markdown("Enter patient details below to predict diabetes risk.")

    col1, col2 = st.columns(2)

    with col1:
        preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Concentration", min_value=1.0, max_value=250.0, value=120.0, step=1.0)
        bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=1.0, max_value=150.0, value=70.0, step=1.0)
        skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=1.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=1.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.01, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)

    if st.button("🔍 Predict", use_container_width=True):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** — The model predicts the patient is **diabetic** (confidence: {proba[1]*100:.1f}%)")
        else:
            st.success(f"✅ **LOW RISK** — The model predicts the patient is **not diabetic** (confidence: {proba[0]*100:.1f}%)")

        st.markdown("#### Prediction Probabilities")
        prob_df = pd.DataFrame({"Class": ["Not Diabetic", "Diabetic"], "Probability (%)": [proba[0]*100, proba[1]*100]})
        st.bar_chart(prob_df.set_index("Class"))

# ===================== MODEL PERFORMANCE PAGE =====================
elif page == "Algorithm Comparison":
    st.title("📊 10 Algorithms — Accuracy Comparison (Best Scaler)")

    # Show results for the best preprocessing technique
    best_scaler_name = global_best["Preprocessing"]
    filtered = [r for r in all_results if r["Preprocessing"] == best_scaler_name]
    results_df = pd.DataFrame(filtered).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
    results_df.index += 1
    results_df.index.name = "Rank"

    st.info(f"Showing results using **{best_scaler_name}** (best overall preprocessing)")

    best_algo = results_df.iloc[0]["Algorithm"]
    best_acc = results_df.iloc[0]["Test Accuracy (%)"]
    st.success(f"🏆 **Best Model: {best_algo}** — Test Accuracy: **{best_acc}%**")

    def highlight_best(row):
        if row["Algorithm"] == best_algo:
            return ["background-color: #d4edda; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.markdown("### Ranked Accuracy Table")
    display_df = results_df.drop(columns=["Preprocessing"])
    st.dataframe(
        display_df.style.apply(highlight_best, axis=1).format({
            "CV Accuracy (%)": "{:.2f}",
            "Test Accuracy (%)": "{:.2f}",
        }),
        use_container_width=True,
        height=420,
    )

    st.markdown("### Test Accuracy Comparison")
    chart_df = display_df.set_index("Algorithm")["Test Accuracy (%)"]
    st.bar_chart(chart_df)

    st.markdown("### Cross-Validation vs Test Accuracy")
    compare_df = display_df.set_index("Algorithm")[["CV Accuracy (%)", "Test Accuracy (%)"]]
    st.bar_chart(compare_df)

# ===================== SAMPLING COMPARISON PAGE =====================
elif page == "Sampling Comparison":
    st.title("⚖️ Sampling Strategies: SMOTE vs Random OverSampler vs No Sampling")

    best_sampling = global_best.get("Sampling", "N/A")
    st.success(
        f"🏆 **Overall Best: {global_best['Algorithm']}** with "
        f"**{global_best['Preprocessing']}** + **{best_sampling}** — "
        f"Test Accuracy: **{global_best['Test Accuracy (%)']}%**"
    )

    # --- Full 150-combo table ---
    st.markdown("### Full Combination Table (3 Samplers × 5 Scalers × 10 Algorithms = 150 combos)")
    full_df = pd.DataFrame(all_results).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
    full_df.index += 1
    full_df.index.name = "Rank"

    def highlight_global(row):
        if (row["Algorithm"] == global_best["Algorithm"]
            and row["Preprocessing"] == global_best["Preprocessing"]
            and row.get("Sampling", "") == best_sampling):
            return ["background-color: #d4edda; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        full_df.style.apply(highlight_global, axis=1).format({
            "CV Accuracy (%)": "{:.2f}",
            "Test Accuracy (%)": "{:.2f}",
        }),
        use_container_width=True,
        height=600,
    )

    # --- Best per sampling strategy ---
    st.markdown("### Best Model per Sampling Strategy")
    if sampler_best:
        sampler_summary = pd.DataFrame([
            {"Sampling": s, "Best Algorithm": info["Algorithm"],
             "Preprocessing": info["Preprocessing"],
             "Test Accuracy (%)": info["Test Accuracy (%)"]}
            for s, info in sampler_best.items()
        ]).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
        sampler_summary.index += 1
        sampler_summary.index.name = "Rank"

        def highlight_best_sampler(row):
            if row["Sampling"] == best_sampling:
                return ["background-color: #d4edda; font-weight: bold"] * len(row)
            return [""] * len(row)

        st.dataframe(
            sampler_summary.style.apply(highlight_best_sampler, axis=1).format({
                "Test Accuracy (%)": "{:.2f}",
            }),
            use_container_width=True,
        )

    # --- Average accuracy per sampling strategy ---
    st.markdown("### Average Test Accuracy per Sampling Strategy")
    avg_by_sampler = full_df.groupby("Sampling")["Test Accuracy (%)"].mean().sort_values(ascending=False)
    st.bar_chart(avg_by_sampler)

    # --- Heatmap: Sampling x Algorithm (averaged across scalers) ---
    st.markdown("### Accuracy Heatmap: Sampling × Algorithm (avg across scalers)")
    pivot_sampler = full_df.pivot_table(
        index="Algorithm", columns="Sampling", values="Test Accuracy (%)", aggfunc="mean"
    )
    st.dataframe(
        pivot_sampler.style.background_gradient(cmap="Greens", axis=None).format("{:.2f}"),
        use_container_width=True,
        height=420,
    )

    # --- Heatmap: Sampling x Scaler (averaged across algorithms) ---
    st.markdown("### Accuracy Heatmap: Sampling × Scaler (avg across algorithms)")
    pivot_sampler_scaler = full_df.pivot_table(
        index="Preprocessing", columns="Sampling", values="Test Accuracy (%)", aggfunc="mean"
    )
    st.dataframe(
        pivot_sampler_scaler.style.background_gradient(cmap="Blues", axis=None).format("{:.2f}"),
        use_container_width=True,
    )

    # --- Class distribution before/after sampling ---
    st.markdown("### Class Distribution Impact")
    zero_cols = ["glucose_concentration", "diastolic_bp", "triceps_skin_fold_thickness", "two_hr_serum_insulin", "bmi"]
    df_clean = df.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)
    X_raw = df_clean.drop("diabetes_class", axis=1)
    y_raw = df_clean["diabetes_class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw)

    dist_data = {"Original Training Set": y_tr.value_counts().to_dict()}
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    smote = SMOTE(random_state=42)
    ros = RandomOverSampler(random_state=42)
    _, y_smote = smote.fit_resample(X_tr, y_tr)
    _, y_ros = ros.fit_resample(X_tr, y_tr)
    dist_data["After SMOTE"] = pd.Series(y_smote).value_counts().to_dict()
    dist_data["After Random OverSampler"] = pd.Series(y_ros).value_counts().to_dict()

    dist_df = pd.DataFrame(dist_data).T.rename(columns={0: "Class 0 (No Diabetes)", 1: "Class 1 (Diabetes)"})
    st.dataframe(dist_df, use_container_width=True)
    st.bar_chart(dist_df)

# ===================== PREPROCESSING COMPARISON PAGE =====================
elif page == "Preprocessing Comparison":
    st.title("🔬 5 Preprocessing Techniques × 10 Algorithms")

    best_sampling = global_best.get("Sampling", "N/A")
    st.success(
        f"🏆 **Overall Best: {global_best['Algorithm']}** with "
        f"**{global_best['Preprocessing']}** + **{best_sampling}** — "
        f"Test Accuracy: **{global_best['Test Accuracy (%)']}%**"
    )

    # --- Full 50-combo table ---
    st.markdown("### Full Combination Table (50 combinations)")
    full_df = pd.DataFrame(all_results).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
    full_df.index += 1
    full_df.index.name = "Rank"

    def highlight_global_best(row):
        if row["Algorithm"] == global_best["Algorithm"] and row["Preprocessing"] == global_best["Preprocessing"]:
            return ["background-color: #d4edda; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        full_df.style.apply(highlight_global_best, axis=1).format({
            "CV Accuracy (%)": "{:.2f}",
            "Test Accuracy (%)": "{:.2f}",
        }),
        use_container_width=True,
        height=600,
    )

    # --- Best model per scaler ---
    st.markdown("### Best Algorithm per Preprocessing Technique")
    scaler_summary = pd.DataFrame([
        {"Preprocessing": s, "Best Algorithm": info["Algorithm"], "Test Accuracy (%)": info["Test Accuracy (%)"]}
        for s, info in scaler_best.items()
    ]).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
    scaler_summary.index += 1
    scaler_summary.index.name = "Rank"

    st.dataframe(
        scaler_summary.style.format({"Test Accuracy (%)": "{:.2f}"}),
        use_container_width=True,
    )

    # --- Pivot heatmap-style table ---
    st.markdown("### Accuracy Heatmap: Scaler × Algorithm")
    pivot_df = full_df.pivot_table(
        index="Algorithm", columns="Preprocessing", values="Test Accuracy (%)"
    )
    st.dataframe(
        pivot_df.style.background_gradient(cmap="Greens", axis=None).format("{:.2f}"),
        use_container_width=True,
        height=420,
    )

    # --- Bar chart per scaler ---
    st.markdown("### Best Accuracy per Preprocessing Technique")
    scaler_chart = scaler_summary.set_index("Preprocessing")["Test Accuracy (%)"]
    st.bar_chart(scaler_chart)

elif page == "Model Performance":
    st.title("Model Performance")

    zero_cols = ["glucose_concentration", "diastolic_bp", "triceps_skin_fold_thickness", "two_hr_serum_insulin", "bmi"]
    df_clean = df.copy()
    df_clean[zero_cols] = df_clean[zero_cols].replace(0, np.nan)
    df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

    X = df_clean.drop("diabetes_class", axis=1)
    y = df_clean["diabetes_class"]
    X_scaled = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%")
    col2.metric("Test Samples", len(y_test))
    col3.metric("Model Type", type(model).__name__)

    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual: No", "Actual: Yes"], columns=["Pred: No", "Pred: Yes"])
    st.dataframe(cm_df)

    if hasattr(model, "feature_importances_"):
        st.markdown("### Feature Importance")
        imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        st.bar_chart(imp)

# ===================== DATASET EXPLORER PAGE =====================
elif page == "Dataset Explorer":
    st.title("Dataset Explorer")

    st.markdown(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

    st.markdown("### Data Preview")
    st.dataframe(df.head(20))

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe().style.format("{:.2f}"))

    st.markdown("### Class Distribution")
    class_counts = df["diabetes_class"].value_counts().rename({0: "Not Diabetic", 1: "Diabetic"})
    st.bar_chart(class_counts)
