"""Modeling pipeline: helper functions + run_pipeline().

This cleaned module ensures imports are at top-level and no model-fitting
code runs on import. Use `from modeling_pipeline import run_pipeline` to
execute the pipeline from other scripts (e.g., `app.py`).
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import statsmodels.api as sm
from lifelines import CoxPHFitter
from scipy.integrate import quad


def weibull_conditional_expectation(c, k, lam):
    """E[T | T > c] for a Weibull(k, lam) distribution."""
    try:
        Sc = np.exp(-(c/lam)**k)
    except OverflowError:
        Sc = 0
    if Sc < 1e-12:
        return c

    def integrand(t):
        return t * (k/lam) * (t/lam)**(k-1) * np.exp(-(t/lam)**k)

    integral, _ = quad(integrand, c, np.inf)
    return integral / Sc


def m_step_weibull(data):
    """Perform M-step by maximizing Weibull log-likelihood (via minimize).

    Returns (k, lam).
    """
    def neg_log_likelihood(params):
        k, lam = params[0], params[1]
        if k <= 0 or lam <= 0:
            return 1e10
        term1 = len(data) * np.log(k) - len(data) * k * np.log(lam)
        term2 = (k - 1) * np.sum(np.log(data))
        term3 = -np.sum((data / lam)**k)
        return -(term1 + term2 + term3)

    k_init, lam_init = 1.0, np.mean(data)
    res = minimize(neg_log_likelihood, [k_init, lam_init], bounds=[(1e-2, None), (1e-2, None)])
    return res.x[0], res.x[1]


def weibull_loglik(data, event, k, lam):
    ll = 0
    for t, e in zip(data, event):
        if e == 1:
            ll += np.log(k/lam) + (k-1)*np.log(t/lam) - (t/lam)**k
        else:
            ll += - (t/lam)**k
    return ll


def run_pipeline(plot=False, random_seed=42, max_iter=12, tolerance=1e-4):
    """Execute the full modeling pipeline and return results.

    Returns a dict with keys:
      - intervals, observed_intervals, censored_flags
      - weibull_k, weibull_lam
      - ll_weibull, ll_poisson, ll_cox
      - mse_weibull
      - df_model, poi_model, cph
    """
    df = pd.read_csv('data.csv/data.csv')

    seizure_indices = df.index[df['y'] == 1].tolist()
    intervals = np.diff(seizure_indices)
    intervals = intervals[intervals > 0]

    np.random.seed(random_seed)
    n = len(intervals)
    censored_flags = np.random.rand(n) < 0.20

    observed_intervals = np.copy(intervals)
    for i in range(n):
        if censored_flags[i] and intervals[i] > 1:
            observed_intervals[i] = np.random.randint(1, intervals[i])

    # EM
    obs_only = observed_intervals[~censored_flags]
    k_est, lam_est = m_step_weibull(obs_only)

    for iteration in range(max_iter):
        complete_data = np.copy(observed_intervals)
        for i in range(n):
            if censored_flags[i]:
                complete_data[i] = weibull_conditional_expectation(complete_data[i], k_est, lam_est)

        k_new, lam_new = m_step_weibull(complete_data)
        if abs(k_new - k_est) < tolerance and abs(lam_new - lam_est) < tolerance:
            k_est, lam_est = k_new, lam_new
            break
        k_est, lam_est = k_new, lam_new

    weibull_k, weibull_lam = k_est, lam_est

    # Build model dataframe
    df_model = pd.DataFrame({
        'interval': observed_intervals[1:],
        'prev_interval': observed_intervals[:-1],
        'censored': censored_flags[1:].astype(int)
    })
    df_model['event'] = 1 - df_model['censored']

    poi_model = sm.GLM(df_model['event'],
                       sm.add_constant(df_model['prev_interval']),
                       family=sm.families.Poisson(),
                       offset=np.log(df_model['interval'])).fit()

    cph = CoxPHFitter()
    cph.fit(df_model[['interval', 'event', 'prev_interval']], duration_col='interval', event_col='event')

    ll_weibull = weibull_loglik(df_model['interval'], df_model['event'], weibull_k, weibull_lam)
    ll_poisson = poi_model.llf
    ll_cox = cph.log_likelihood_

    true_censored = intervals[1:][df_model['censored'] == 1]
    pred_weibull = [weibull_conditional_expectation(c, weibull_k, weibull_lam)
                    for c in df_model['interval'][df_model['censored'] == 1]]
    mse_weibull = np.mean((true_censored - pred_weibull)**2) if len(true_censored) > 0 else float('nan')

    results = {
        'df': df,
        'intervals': intervals,
        'observed_intervals': observed_intervals,
        'censored_flags': censored_flags,
        'weibull_k': weibull_k,
        'weibull_lam': weibull_lam,
        'll_weibull': ll_weibull,
        'll_poisson': ll_poisson,
        'll_cox': ll_cox,
        'mse_weibull': mse_weibull,
        'df_model': df_model,
        'poi_model': poi_model,
        'cph': cph,
    }

    return results

