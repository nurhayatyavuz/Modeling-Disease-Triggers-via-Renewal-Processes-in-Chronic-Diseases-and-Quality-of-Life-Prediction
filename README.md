# Modeling Disease Triggers via Renewal Processes in Chronic Diseases and Quality of Life Prediction

> A statistical modeling framework for epileptic seizure intervals using Weibull-based non-homogeneous renewal processes, EM algorithm for censored data, and multi-model comparison.

---

## 📌 Overview

This project models the time intervals between epileptic seizures using survival analysis and renewal process theory. Unlike classical approaches that assume constant risk (e.g. Poisson), this framework captures the **time-varying nature of seizure risk** through a Weibull distribution fitted via the Expectation-Maximization (EM) algorithm.

The key finding: **k = 1.39 > 1**, meaning seizure risk increases over time — the longer a patient goes without a seizure, the closer the next one becomes. This directly refutes the memoryless assumption used in standard models.

---

## 🚀 Features

- **Weibull renewal process** with shape (k) and scale (λ) parameter estimation
- **EM Algorithm** to handle right-censored observations (missing seizure times)
- **Multi-model comparison** via log-likelihood: Weibull (EM) vs. Poisson vs. CoxPH
- **MSE evaluation** on censored data to assess imputation accuracy
- **Interactive web application** — adjust seed and iterations, run live and compare models in real time
- Convergence tracking with parameter evolution plot (k and λ per iteration)

---

## 📊 Results Summary

| Model | Log-Likelihood |
|---|---|
| Weibull (EM) | −4,871.04 ✅ Best fit |
| Poisson | −2,680.50 ⚠️ Structurally mismatched |
| CoxPH | −12,025.77 ❌ Poorest fit |

**Final Weibull Parameters:**
```json
{
  "k": 1.3917,
  "lambda": 5.8120
}
```

**Censored Data MSE:** `13.18`

**EM Convergence:** Met at iteration 3 out of 12 maximum — indicating strong model-data alignment.

---

## 🗂️ Dataset

- **Total intervals:** 2,299
- **Fully observed:** 1,812
- **Simulated censored (20%):** 487
- Censoring applied randomly via a seed-controlled mechanism to simulate real-world incomplete follow-up

---

## 🛠️ How to Run

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install streamlit pandas numpy scipy statsmodels lifelines matplotlib
```

### 3. Run the web application

```bash
streamlit app.py
```

Then open your browser and go to `http://localhost:8501`

### 4. Using the interface

| Control | Description |
|---|---|
| **Random seed** | Determines which 20% of observations are treated as censored |
| **Max EM iterations** | Upper limit for EM loop (algorithm stops early if convergence is met) |
| **Run EM and Compare** | Fits Weibull via EM, compares against Poisson and CoxPH, outputs MSE |

---

## 📐 Methodology

### Weibull Distribution
Models the waiting time until the next seizure. Parameterized by:
- **k (shape):** Controls how risk changes over time. k > 1 → increasing risk.
- **λ (scale):** Determines the time scale of events.

### EM Algorithm
Handles right-censored observations in two alternating steps:
- **E-step:** Computes expected values for missing seizure times given current parameters.
- **M-step:** Updates k and λ by maximizing the expected log-likelihood.

Steps repeat until the change between iterations falls below a convergence threshold.

### Censoring
Right-censored observations occur when a patient's follow-up ends before a seizure is recorded. Rather than discarding these, the EM algorithm incorporates them as partial information — improving parameter estimates without introducing bias.

### Model Comparison
Log-likelihood values are used to assess model fit. Note: Poisson and Weibull log-likelihoods are not directly comparable due to different data type assumptions (count vs. continuous duration data). Weibull is selected on both structural grounds and fit quality.

---

## 📚 Background & References

This project builds on:
- **Renewal Theory** — each seizure resets the process; the next waiting time begins fresh.
- **Survival Analysis** — adapted for repeated events with censored observations.
- Cox, D.R. (1972). *Regression Models and Life-Tables.* Journal of the Royal Statistical Society.
- Dempster, A.P., Laird, N.M., Rubin, D.B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm.*

---

## 👩‍💻 Author

**Nur Hayat Yavuz**  
Statistics — Chronic Disease Modeling  
[LinkedIn](https://linkedin.com/in/nurhayatyavuz) ·

---

## 📄 License

MIT License — free to use with attribution.
