"""
generate_dataset.py
-------------------
Generates a clean, realistic synthetic dataset of undergraduate student records
for the student performance prediction study (University of Benin).

The relationships between features and the Pass/Fail outcome are modelled on the
findings reviewed in Chapter Two: prior academic performance, attendance and
study hours are the strongest predictors, with parental education and
extracurricular participation contributing smaller effects. Noise is added so the
task is non-trivial and the models have to genuinely learn (rather than memorise a
perfect rule).

Output: data/student_performance.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_STATE = 42
N_STUDENTS = 1200          # realistic size for a departmental study
OUTPUT = os.path.join(os.path.dirname(__file__), "data", "student_performance.csv")


def generate(n=N_STUDENTS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    # ---- Raw feature generation (realistic distributions) ----
    # Prior CGPA on a 5.0 scale, centred slightly above the pass mark
    prior_cgpa = np.clip(rng.normal(3.05, 0.75, n), 0.5, 5.0)

    # Attendance rate (%) - most students attend fairly well, left-skewed
    attendance = np.clip(rng.normal(78, 14, n), 25, 100)

    # Study hours per day - right-skewed, most study 1-4 hours
    study_hours = np.clip(rng.gamma(shape=2.2, scale=1.25, size=n), 0.0, 10.0)

    # Age (years) - typical undergraduate range
    age = np.clip(rng.normal(21, 2.2, n).round(), 16, 32).astype(int)

    # Gender - roughly balanced
    gender = rng.choice(["Male", "Female"], size=n, p=[0.52, 0.48])

    # Parental education level (ordinal categories)
    parental_edu = rng.choice(
        ["None", "Primary", "Secondary", "Tertiary", "Postgraduate"],
        size=n, p=[0.06, 0.14, 0.34, 0.34, 0.12]
    )
    parental_edu_score = pd.Series(parental_edu).map({
        "None": 0, "Primary": 1, "Secondary": 2, "Tertiary": 3, "Postgraduate": 4
    }).to_numpy()

    # Extracurricular participation (Yes/No)
    extracurricular = rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58])
    extra_score = (extracurricular == "Yes").astype(float)

    # ---- Latent "academic strength" score that drives the outcome ----
    # Standardise the numeric drivers so weights are interpretable
    def z(x):
        return (x - np.mean(x)) / np.std(x)

    latent = (
        1.55 * z(prior_cgpa) +      # strongest predictor
        1.05 * z(attendance) +      # strong behavioural predictor
        0.95 * z(study_hours) +     # strong behavioural predictor
        0.35 * z(parental_edu_score) +  # moderate contextual effect
        0.20 * extra_score -        # small positive effect
        0.05 * z(age)               # very weak effect
    )

    # Add noise so the boundary is not perfectly separable (realistic)
    latent = latent + rng.normal(0, 1.1, n)

    # Convert latent score to a Pass probability via logistic function
    prob_pass = 1.0 / (1.0 + np.exp(-latent))

    # Draw the outcome. A positive bias is added to prob_pass so that roughly
    # two-thirds of students pass, leaving a realistic minority "Fail/At-Risk"
    # class (~33%) for SMOTE to balance during training.
    prob_pass = np.clip(prob_pass + 0.12, 0.0, 1.0)
    outcome = np.where(rng.random(n) < prob_pass, "Pass", "Fail")

    df = pd.DataFrame({
        "student_id": [f"UNIBEN{str(i).zfill(4)}" for i in range(1, n + 1)],
        "prior_cgpa": prior_cgpa.round(2),
        "attendance_rate": attendance.round(1),
        "study_hours": study_hours.round(1),
        "age": age,
        "gender": gender,
        "parental_education": parental_edu,
        "extracurricular": extracurricular,
        "academic_outcome": outcome,
    })

    return df


def main():
    df = generate()
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    # Quick summary for sanity
    print(f"Dataset written to: {OUTPUT}")
    print(f"Total records      : {len(df)}")
    print("\nClass distribution :")
    print(df["academic_outcome"].value_counts())
    print(f"\nPass rate          : {(df['academic_outcome'].eq('Pass').mean()*100):.1f}%")
    print("\nSample rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
