"""
train_models.py
---------------
Full machine learning pipeline for the student performance prediction study.

Implements every stage described in Chapter Three:
  1. Load the dataset
  2. Preprocess (encode categoricals, scale numerics)
  3. Train/test split
  4. Handle class imbalance with SMOTE (training partition only)
  5. Feature selection with Recursive Feature Elimination (RFE)
  6. Train and tune five classifiers: Decision Tree, Random Forest,
     Logistic Regression, SVM, K-Nearest Neighbour
  7. Evaluate with 10-fold cross-validation and on the held-out test set
     (accuracy, precision, recall, F1, AUC-ROC)
  8. Save the best model, the fitted preprocessor, results tables and charts

Outputs:
  models/best_model.joblib
  models/preprocessor.joblib
  models/metadata.joblib
  results/*.csv, results/*.png
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data", "student_performance.csv")
MODELS = os.path.join(BASE, "models")
RESULTS = os.path.join(BASE, "results")
RANDOM_STATE = 42

NUMERIC = ["prior_cgpa", "attendance_rate", "study_hours", "age"]
CATEGORICAL = ["gender", "parental_education", "extracurricular"]
TARGET = "academic_outcome"


# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA)
    df = df.drop(columns=["student_id"])
    return df


def build_preprocessor():
    """ColumnTransformer: scale numerics, one-hot encode categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def get_models():
    """Five classifiers with a small tuning grid each."""
    return {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 8, None], "min_samples_split": [2, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {"n_estimators": [100, 200], "max_depth": [6, 10, None]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 10.0]},
        ),
        "Support Vector Machine": (
            SVC(probability=True, random_state=RANDOM_STATE),
            {"C": [0.5, 1.0, 10.0], "kernel": ["rbf", "linear"]},
        ),
        "K-Nearest Neighbour": (
            KNeighborsClassifier(),
            {"n_neighbors": [5, 7, 9, 11], "weights": ["uniform", "distance"]},
        ),
    }


# ----------------------------------------------------------------------
def main():
    os.makedirs(MODELS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION - MODEL TRAINING")
    print("=" * 60)

    # ---- 1. Load ----
    df = load_data()
    X = df[NUMERIC + CATEGORICAL]
    y_raw = df[TARGET]

    # Encode target: Fail = 1 (positive/at-risk class), Pass = 0
    # We treat the at-risk (Fail) student as the positive class because that is
    # the group the early-warning system exists to detect.
    le = LabelEncoder()
    le.fit(["Pass", "Fail"])                 # Fail -> 0? ensure mapping explicit
    y = (y_raw == "Fail").astype(int)        # Fail = 1, Pass = 0
    print(f"\nRecords: {len(df)} | Features: {X.shape[1]}")
    print(f"Positive class = 'Fail' (at-risk). Count: {int(y.sum())} "
          f"({y.mean()*100:.1f}%)")

    # ---- 2/3. Preprocess + split ----
    preprocessor = build_preprocessor()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    feat_names = (
        NUMERIC
        + list(preprocessor.named_transformers_["cat"]
               .get_feature_names_out(CATEGORICAL))
    )
    print(f"After encoding: {X_train.shape[1]} model features")

    # ---- 4. SMOTE on training only ----
    print("\nApplying SMOTE to training partition...")
    print(f"  Before: {np.bincount(y_train)} (Pass, Fail)")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  After : {np.bincount(y_train_bal)} (Pass, Fail)")

    # ---- 5. Feature selection (RFE with Random Forest estimator) ----
    print("\nRunning Recursive Feature Elimination (RFE)...")
    rfe = RFE(
        estimator=RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        n_features_to_select=max(4, X_train_bal.shape[1] // 2),
    )
    rfe.fit(X_train_bal, y_train_bal)
    selected = [f for f, keep in zip(feat_names, rfe.support_) if keep]
    print(f"  Selected {len(selected)} features: {selected}")

    X_train_sel = rfe.transform(X_train_bal)
    X_test_sel = rfe.transform(X_test)

    # ---- 6/7. Train, tune, cross-validate, evaluate ----
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    trained = {}
    roc_data = {}

    for name, (estimator, grid) in get_models().items():
        print(f"\n>>> {name}")
        gs = GridSearchCV(estimator, grid, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_train_sel, y_train_bal)
        best = gs.best_estimator_
        trained[name] = best
        print(f"    Best params: {gs.best_params_}")

        # 10-fold CV on the balanced training data
        cv_acc = cross_val_score(best, X_train_sel, y_train_bal, cv=cv,
                                 scoring="accuracy").mean()

        # Held-out test evaluation
        y_pred = best.predict(X_test_sel)
        y_proba = best.predict_proba(X_test_sel)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        rows.append({
            "Model": name,
            "CV Accuracy (10-fold)": round(cv_acc, 4),
            "Test Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "AUC-ROC": round(auc, 4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, auc)

        print(f"    Test Acc {acc:.3f} | Prec {prec:.3f} | Rec {rec:.3f} "
              f"| F1 {f1:.3f} | AUC {auc:.3f}")

    results_df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    results_df.to_csv(os.path.join(RESULTS, "model_comparison.csv"), index=False)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (sorted by F1-Score)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # ---- Best model ----
    best_name = results_df.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"\nBEST MODEL: {best_name}")

    # ---- 8. Persist artifacts ----
    joblib.dump(best_model, os.path.join(MODELS, "best_model.joblib"))
    joblib.dump(preprocessor, os.path.join(MODELS, "preprocessor.joblib"))
    joblib.dump(rfe, os.path.join(MODELS, "rfe.joblib"))
    joblib.dump({
        "best_model_name": best_name,
        "selected_features": selected,
        "all_features": feat_names,
        "numeric": NUMERIC,
        "categorical": CATEGORICAL,
        "parental_education_levels": ["None", "Primary", "Secondary", "Tertiary", "Postgraduate"],
        "gender_levels": ["Male", "Female"],
        "extracurricular_levels": ["Yes", "No"],
    }, os.path.join(MODELS, "metadata.joblib"))

    # ---- Charts ----
    _plot_comparison(results_df)
    _plot_confusion(best_model, X_test_sel, y_test, best_name)
    _plot_roc(roc_data)
    _plot_feature_importance(best_model, selected, best_name)
    _save_classification_report(best_model, X_test_sel, y_test, best_name)

    print(f"\nAll artifacts saved to '{MODELS}' and '{RESULTS}'.")
    print("Done.")


# ----------------------------------------------------------------------
def _plot_comparison(df):
    metrics = ["Test Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    ax = df.set_index("Model")[metrics].plot(kind="bar", figsize=(11, 6),
                                             width=0.8, colormap="viridis")
    ax.set_title("Comparison of Machine Learning Classifiers", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "model_comparison.png"), dpi=150)
    plt.close()


def _plot_confusion(model, X_test, y_test, name):
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pass", "Fail"], yticklabels=["Pass", "Fail"])
    plt.title(f"Confusion Matrix - {name}", fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
    plt.close()


def _plot_roc(roc_data):
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.title("ROC Curves - All Classifiers", fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "roc_curves.png"), dpi=150)
    plt.close()


def _plot_feature_importance(model, features, name):
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    if importances is None:
        return
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances[order], y=np.array(features)[order], palette="rocket")
    plt.title(f"Feature Importance - {name}", fontweight="bold")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "feature_importance.png"), dpi=150)
    plt.close()


def _save_classification_report(model, X_test, y_test, name):
    rep = classification_report(y_test, model.predict(X_test),
                                target_names=["Pass", "Fail"])
    with open(os.path.join(RESULTS, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {name}\n\n{rep}")


if __name__ == "__main__":
    main()
