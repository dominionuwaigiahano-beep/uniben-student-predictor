# Student Academic Performance Early-Warning System
### Department of Computer Science, University of Benin

A machine learning system that predicts whether an undergraduate student is on
track to pass or is at risk of failing, so that academic advisors can arrange
early intervention. The project trains and compares five supervised classifiers,
selects the best one, and serves it through a Flask web application.

---

## What the project contains

| File / folder | Purpose |
|---|---|
| `generate_dataset.py` | Creates the synthetic student dataset (`data/student_performance.csv`). |
| `train_models.py` | Full ML pipeline: preprocessing, SMOTE, feature selection, training and evaluating all five models, saving the best one and all result charts. |
| `app.py` | Flask web application that serves the prediction interface. |
| `templates/` | HTML pages (`index.html`, `about.html`). |
| `static/` | Stylesheet and front-end JavaScript. |
| `models/` | Saved best model, preprocessor, feature selector and metadata (created by `train_models.py`). |
| `results/` | Comparison table, charts and classification report (created by `train_models.py`). |
| `requirements.txt` | Python dependencies. |

---

## How to run it

You need Python 3.10 or newer.

**1. Install the dependencies**

```bash
pip install -r requirements.txt
```

**2. Generate the dataset**

```bash
python generate_dataset.py
```

**3. Train the models** (produces the saved model and all result charts)

```bash
python train_models.py
```

**4. Start the web application**

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## Using the application

1. On the **Assess** page, enter the student's details: prior CGPA, attendance
   rate, study hours per day, age, gender, parental education level and
   extracurricular participation.
2. Click **Assess student**.
3. The system shows the predicted outcome (On Track or At-Risk), the estimated
   risk of failing, a risk band (Low, Moderate or High) and a recommended action.

---

## The five models compared

Decision Tree, Random Forest, Logistic Regression, Support Vector Machine and
K-Nearest Neighbour are each trained and evaluated with ten-fold cross-validation
using accuracy, precision, recall, F1-score and AUC-ROC. Class imbalance is
handled with SMOTE and the most informative features are chosen using Recursive
Feature Elimination. The best-performing model (by F1-score) is saved and used by
the web application. See `results/model_comparison.csv` for the full table.

---

## Note on the dataset

The dataset used here is **synthetic**: it was generated to mirror the realistic
statistical relationships between student attributes and academic outcomes
reported in the literature (prior performance, attendance and study hours being
the strongest predictors). It contains no real student records. To use real
institutional data instead, replace `data/student_performance.csv` with a file
that has the same columns and re-run `train_models.py`.
