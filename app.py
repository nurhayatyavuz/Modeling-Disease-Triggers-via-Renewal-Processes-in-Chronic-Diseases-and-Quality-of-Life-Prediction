import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from lifelines import CoxPHFitter
import time

# Import helper functions from modeling_pipeline
from modeling_pipeline import (
    weibull_conditional_expectation,
    m_step_weibull,
    weibull_loglik,
)

st.set_page_config(page_title="Epileptic Seizure Modeling", layout="wide")
st.title("Epileptic Seizure: Weibull EM vs Poisson & Cox")

@st.cache_data
def load_data(path='data.csv/data.csv'):
    return pd.read_csv(path)

# Load data
df = load_data()
# Compute seizure intervals (keep identical logic to modeling_pipeline.py)
seizure_indices = df.index[df['y'] == 1].tolist()
intervals = np.diff(seizure_indices)
intervals = intervals[intervals > 0]

# Sidebar controls
st.sidebar.header('Controls')
seed = st.sidebar.number_input('Random seed for censoring', value=42, step=1)
max_iter = st.sidebar.number_input('Max EM iterations', value=12, step=1)
run_button = st.sidebar.button('Run EM and Compare')

# Show distribution
st.subheader('Distribution of Time Intervals Between Seizures')
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(intervals, bins=50, color='skyblue', edgecolor='black')
ax.set_xlabel('Time Interval Elapsed')
ax.set_ylabel('Frequency Count')
st.pyplot(fig)

# Prepare censoring simulation
np.random.seed(int(seed))
n = len(intervals)
censored_flags = np.random.rand(n) < 0.20
observed_intervals = np.copy(intervals)
for i in range(n):
    if censored_flags[i] and intervals[i] > 1:
        observed_intervals[i] = np.random.randint(1, intervals[i])

st.write(f"Total intervals: {n}")
st.write(f"Fully observed: {np.sum(~censored_flags)} | Simulated censored: {np.sum(censored_flags)}")

# Placeholders for live progress
progress_placeholder = st.empty()
chart_placeholder = st.empty()

if run_button:
    # Warm start using observed-only
    obs_only = observed_intervals[~censored_flags]
    k_est, lam_est = m_step_weibull(obs_only)

    ks = [k_est]
    lams = [lam_est]

    progress_placeholder.write('Starting custom EM Algorithm Optimization...')

    for iteration in range(int(max_iter)):
        # E-step
        complete_data = np.copy(observed_intervals)
        for i in range(n):
            if censored_flags[i]:
                complete_data[i] = weibull_conditional_expectation(complete_data[i], k_est, lam_est)

        # M-step
        k_new, lam_new = m_step_weibull(complete_data)
        ks.append(k_new)
        lams.append(lam_new)

        # Update live progress
        progress_placeholder.markdown(f"**Iteration {iteration+1}** — Shape (k): **{k_new:.4f}**, Scale (λ): **{lam_new:.4f}**")
        history_df = pd.DataFrame({'k': ks, 'lambda': lams})
        chart_placeholder.line_chart(history_df)

        # Small sleep to make updates visible
        time.sleep(0.2)

        if abs(k_new - k_est) < 1e-4 and abs(lam_new - lam_est) < 1e-4:
            progress_placeholder.markdown("**Convergence criteria met.**")
            k_est, lam_est = k_new, lam_new
            break

        k_est, lam_est = k_new, lam_new

    weibull_k, weibull_lam = k_est, lam_est

    st.subheader('Final Weibull Parameters')
    st.write({'k': float(weibull_k), 'lambda': float(weibull_lam)})

    # Build df_model consistent with modeling_pipeline
    df_model = pd.DataFrame({
        'interval': observed_intervals[1:],
        'prev_interval': observed_intervals[:-1],
        'censored': censored_flags[1:].astype(int)
    })
    df_model['event'] = 1 - df_model['censored']

    # Fit Poisson
    poi_model = sm.GLM(df_model['event'],
                       sm.add_constant(df_model['prev_interval']),
                       family=sm.families.Poisson(),
                       offset=np.log(df_model['interval'])).fit()

    # Fit Cox
    cph = CoxPHFitter()
    cph.fit(df_model[['interval', 'event', 'prev_interval']], duration_col='interval', event_col='event')

    ll_weibull = weibull_loglik(df_model['interval'], df_model['event'], weibull_k, weibull_lam)
    ll_poisson = poi_model.llf
    ll_cox = cph.log_likelihood_

    comp_df = pd.DataFrame({
        'model': ['Weibull (EM)', 'Poisson', 'CoxPH'],
        'log_likelihood': [ll_weibull, ll_poisson, ll_cox]
    })

    st.subheader('Log-Likelihood Comparison')
    st.table(comp_df)

    # MSE for censored predictions
    true_censored = intervals[1:][df_model['censored'] == 1]
    pred_weibull = [weibull_conditional_expectation(c, weibull_k, weibull_lam)
                    for c in df_model['interval'][df_model['censored'] == 1]]
    if len(true_censored) > 0:
        mse_weibull = np.mean((true_censored - pred_weibull)**2)
    else:
        mse_weibull = float('nan')

    st.subheader('Censored Data MSE')
    st.write({'mse_weibull': float(mse_weibull)})

    st.success('Done')
