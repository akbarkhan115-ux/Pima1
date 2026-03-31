import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, RandomOverSampler
import joblib
import json

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("Pima  (1).csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:\n", df["diabetes_class"].value_counts())

# --- Data Cleaning ---
zero_invalid_cols = [
    "glucose_concentration",
    "diastolic_bp",
    "triceps_skin_fold_thickness",
    "two_hr_serum_insulin",
    "bmi",
]
df[zero_invalid_cols] = df[zero_invalid_cols].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

# --- Features & Target ---
X = df.drop("diabetes_class", axis=1)
y = df["diabetes_class"]

# --- 5 Preprocessing Techniques ---
class LogTransformer:
    """Custom log1p transformer with fit/transform API."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(np.abs(X))
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

scalers = {
    "MinMax Scaler": MinMaxScaler(),
    "Standard Scaler": StandardScaler(),
    "Robust Scaler": RobustScaler(),
    "MaxAbs Scaler": MaxAbsScaler(),
    "Log Transformation": LogTransformer(),
}

# --- 10 Classification Algorithms ---
def get_models():
    return {
        "Logistic Regression": LogisticRegression(C=1, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42),
        "SVM (RBF)": SVC(C=10, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=10, random_state=42),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 3 Sampling Strategies ---
sampling_strategies = {
    "No Sampling": None,
    "SMOTE": SMOTE(random_state=42),
    "Random OverSampler": RandomOverSampler(random_state=42),
}

# --- Train all combinations: 3 samplers x 5 scalers x 10 models = 150 combos ---
all_results = []
scaler_best = {}
sampler_best = {}
global_best_acc = 0
global_best_model = None
global_best_scaler = None
global_best_scaler_name = None
global_best_model_name = None
global_best_sampler_name = None

for sampler_name, sampler_obj in sampling_strategies.items():
    for scaler_name, scaler_obj in scalers.items():
        print(f"\n{'='*60}")
        print(f"  Sampling: {sampler_name} | Preprocessing: {scaler_name}")
        print(f"{'='*60}")

        X_scaled = scaler_obj.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply sampling to training set only
        if sampler_obj is not None:
            X_train_res, y_train_res = sampler_obj.fit_resample(X_train, y_train)
            print(f"  Resampled training set: {len(y_train)} -> {len(y_train_res)}")
        else:
            X_train_res, y_train_res = X_train, y_train

        models = get_models()

        for model_name, clf in models.items():
            cv_scores = cross_val_score(clf, X_train_res, y_train_res, cv=cv, scoring="accuracy")
            cv_mean = cv_scores.mean()

            clf.fit(X_train_res, y_train_res)
            y_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            all_results.append({
                "Sampling": sampler_name,
                "Preprocessing": scaler_name,
                "Algorithm": model_name,
                "CV Accuracy (%)": round(cv_mean * 100, 2),
                "Test Accuracy (%)": round(test_acc * 100, 2),
            })

            print(f"  {model_name:25s} | CV: {cv_mean*100:.2f}% | Test: {test_acc*100:.2f}%")

            if test_acc > global_best_acc:
                global_best_acc = test_acc
                global_best_model = clf
                global_best_scaler = scaler_obj
                global_best_scaler_name = scaler_name
                global_best_model_name = model_name
                global_best_sampler_name = sampler_name

        # Track best per scaler
        scaler_key = f"{sampler_name} + {scaler_name}"
        scaler_results = [r for r in all_results if r["Sampling"] == sampler_name and r["Preprocessing"] == scaler_name]
        best_r = max(scaler_results, key=lambda x: x["Test Accuracy (%)"])
        scaler_best[scaler_key] = {"Algorithm": best_r["Algorithm"], "Test Accuracy (%)": best_r["Test Accuracy (%)"]}

    # Track best per sampler
    sampler_results = [r for r in all_results if r["Sampling"] == sampler_name]
    best_s = max(sampler_results, key=lambda x: x["Test Accuracy (%)"])
    sampler_best[sampler_name] = {
        "Algorithm": best_s["Algorithm"],
        "Preprocessing": best_s["Preprocessing"],
        "Test Accuracy (%)": best_s["Test Accuracy (%)"],
    }

# --- Summary table ---
results_df = pd.DataFrame(all_results).sort_values("Test Accuracy (%)", ascending=False).reset_index(drop=True)
results_df.index += 1
results_df.index.name = "Rank"

print(f"\n{'='*70}")
print(f"  FULL COMBINATION TABLE (3 Samplers x 5 Scalers x 10 Algorithms = {len(all_results)} combos)")
print(f"{'='*70}\n")
print(results_df.head(20).to_string())

print(f"\n{'='*70}")
print("  BEST MODEL PER SAMPLING STRATEGY")
print(f"{'='*70}\n")
for s, info in sampler_best.items():
    print(f"  {s:20s} => {info['Algorithm']:25s} + {info['Preprocessing']:20s} | Acc: {info['Test Accuracy (%)']:.2f}%")

print(f"\n*** OVERALL BEST: {global_best_model_name} + {global_best_scaler_name} + {global_best_sampler_name} => {global_best_acc*100:.2f}% ***\n")

# Re-train best model with best scaler for final save
X_best = global_best_scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_best, y, test_size=0.2, random_state=42, stratify=y
)
global_best_model.fit(X_train, y_train)
print(classification_report(y_test, global_best_model.predict(X_test)))

# --- Save artifacts ---
joblib.dump(global_best_model, "best_model.pkl")
joblib.dump(global_best_scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

# Save all results for the Streamlit app
save_data = {
    "all_results": all_results,
    "scaler_best": scaler_best,
    "sampler_best": sampler_best,
    "global_best": {
        "Algorithm": global_best_model_name,
        "Preprocessing": global_best_scaler_name,
        "Sampling": global_best_sampler_name,
        "Test Accuracy (%)": round(global_best_acc * 100, 2),
    },
}
with open("model_results.json", "w") as f:
    json.dump(save_data, f)

print(f"Saved: best_model.pkl, scaler.pkl, feature_names.pkl, model_results.json")
