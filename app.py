"""
app.py
------
Flask web application for the Student Academic Performance Early-Warning System,
University of Benin.

An academic advisor enters a student's details; the trained best model returns a
Pass / At-Risk classification together with a risk probability, so that timely
intervention can be arranged for at-risk students.

Run:
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify

BASE = os.path.dirname(__file__)
MODELS = os.path.join(BASE, "models")

app = Flask(__name__)

# ---- Load artifacts once at startup ----
best_model = joblib.load(os.path.join(MODELS, "best_model.joblib"))
preprocessor = joblib.load(os.path.join(MODELS, "preprocessor.joblib"))
rfe = joblib.load(os.path.join(MODELS, "rfe.joblib"))
meta = joblib.load(os.path.join(MODELS, "metadata.joblib"))

NUMERIC = meta["numeric"]
CATEGORICAL = meta["categorical"]
MODEL_NAME = meta["best_model_name"]


def predict_student(payload):
    """Take a dict of raw student features, return prediction dict."""
    row = pd.DataFrame([{
        "prior_cgpa": float(payload["prior_cgpa"]),
        "attendance_rate": float(payload["attendance_rate"]),
        "study_hours": float(payload["study_hours"]),
        "age": int(payload["age"]),
        "gender": payload["gender"],
        "parental_education": payload["parental_education"],
        "extracurricular": payload["extracurricular"],
    }])

    X = preprocessor.transform(row)
    X = rfe.transform(X)

    # Positive class (index 1) = Fail / At-Risk
    proba_at_risk = float(best_model.predict_proba(X)[0][1])
    pred = int(best_model.predict(X)[0])   # 1 = Fail, 0 = Pass

    label = "At-Risk" if pred == 1 else "On Track"
    outcome = "Fail" if pred == 1 else "Pass"

    # Risk band for advisor guidance
    if proba_at_risk >= 0.66:
        band = "High"
    elif proba_at_risk >= 0.40:
        band = "Moderate"
    else:
        band = "Low"

    return {
        "outcome": outcome,
        "label": label,
        "risk_probability": round(proba_at_risk * 100, 1),
        "success_probability": round((1 - proba_at_risk) * 100, 1),
        "risk_band": band,
        "model": MODEL_NAME,
    }


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        parental_levels=meta["parental_education_levels"],
        gender_levels=meta["gender_levels"],
        extracurricular_levels=meta["extracurricular_levels"],
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        result = predict_student(data)
        return jsonify({"ok": True, "result": result})
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"ok": False, "error": f"Invalid input: {e}"}), 400


@app.route("/about")
def about():
    return render_template("about.html", model_name=MODEL_NAME)


if __name__ == "__main__":
    # For local use, run: python app.py  (opens on http://127.0.0.1:5000)
    # When hosted (e.g. on Render), the platform sets the PORT variable and
    # runs the app through gunicorn instead of this block.
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
