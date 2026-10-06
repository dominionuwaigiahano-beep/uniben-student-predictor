/* Front-end logic: submit student details, render the risk assessment. */

const form = document.getElementById("assess-form");
const empty = document.getElementById("result-empty");
const body = document.getElementById("result-body");
const errorBox = document.getElementById("result-error");
const panel = document.getElementById("result-panel");
const btn = form.querySelector(".assess-btn");

const ACTIONS = {
  Low: "No action needed. Continue routine monitoring.",
  Moderate: "Schedule a check-in with the student's academic advisor.",
  High: "Refer for immediate intervention: tutoring, counselling or course-load review.",
};

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    prior_cgpa: document.getElementById("prior_cgpa").value,
    attendance_rate: document.getElementById("attendance_rate").value,
    study_hours: document.getElementById("study_hours").value,
    age: document.getElementById("age").value,
    gender: document.getElementById("gender").value,
    parental_education: document.getElementById("parental_education").value,
    extracurricular: document.getElementById("extracurricular").value,
  };

  btn.disabled = true;
  btn.textContent = "Assessing…";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error || "Prediction failed.");
    render(data.result);
  } catch (err) {
    showError(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Assess student";
  }
});

function render(r) {
  empty.hidden = true;
  errorBox.hidden = true;
  body.hidden = false;

  // state class drives all the signal colours
  const state =
    r.risk_band === "High" ? "state-high" :
    r.risk_band === "Moderate" ? "state-moderate" : "state-low";
  panel.className = "panel result-panel " + state;

  document.getElementById("verdict-label").textContent = r.label;
  document.getElementById("verdict-band").textContent = r.risk_band + " risk";
  document.getElementById("risk-value").textContent = r.risk_probability + "%";
  document.getElementById("pass-value").textContent = r.success_probability + "%";
  document.getElementById("band-value").textContent = r.risk_band;
  document.getElementById("action-value").textContent = ACTIONS[r.risk_band];

  // animate gauge after paint
  const fill = document.getElementById("gauge-fill");
  fill.style.width = "0%";
  requestAnimationFrame(() =>
    requestAnimationFrame(() => { fill.style.width = r.risk_probability + "%"; })
  );
}

function showError(msg) {
  body.hidden = true;
  empty.hidden = true;
  errorBox.hidden = false;
  errorBox.textContent = msg;
  panel.className = "panel result-panel";
}
