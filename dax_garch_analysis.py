"""
DAX Volatility Modeling with GARCH
===================================
End-to-end pipeline: download -> diagnose -> fit GARCH(1,1) -> forecast -> VaR
-> compare to asymmetric GJR-GARCH.

Dependencies:
    pip install yfinance arch statsmodels matplotlib pandas numpy scipy

Run:
    python dax_garch_analysis.py

Outputs:
    - Printed diagnostics, coefficients, VaR estimate
    - PNG plots saved to ./plots/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# 1. DATA: download DAX and compute log returns
# ============================================================
TICKER = "^GDAXI"          # DAX index on Yahoo Finance
START  = "2015-01-01"
END    = "2025-12-31"

print(f"\n[1] Downloading {TICKER} from {START} to {END}...")
raw = yf.download(TICKER, start=START, end=END, auto_adjust=True, progress=False)
prices = raw["Close"]
if isinstance(prices, pd.DataFrame):       # yfinance multi-col shim
    prices = prices.iloc[:, 0]
prices = prices.dropna()

# Log returns in percent (arch_model is happier with returns scaled this way)
returns = 100 * np.log(prices / prices.shift(1)).dropna()
returns.name = "DAX log returns (%)"

print(f"    {len(returns)} observations | "
      f"{returns.index[0].date()} → {returns.index[-1].date()}")
print(f"    mean = {returns.mean():.4f}%   std = {returns.std():.4f}%   "
      f"skew = {returns.skew():.2f}   kurt = {returns.kurt():.2f}")


# ============================================================
# 2. VISUALIZE: prices vs returns (clustering is the hook)
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(prices, color="#1f4e79", lw=0.8)
axes[0].set_title("DAX — Price Level")
axes[0].set_ylabel("Index level")

axes[1].plot(returns, color="#c0392b", lw=0.5)
axes[1].axhline(0, color="black", lw=0.4)
axes[1].set_title("DAX — Daily Log Returns (%)  ← note how calm/wild periods cluster")
axes[1].set_ylabel("Return (%)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_price_and_returns.png", bbox_inches="tight")
plt.close()
print(f"[2] Saved: {OUT_DIR}/01_price_and_returns.png")


# ============================================================
# 3. STATIONARITY: ADF test on returns
# ============================================================
print("\n[3] Augmented Dickey-Fuller test on returns")
adf_stat, adf_p, _, _, adf_crit, _ = adfuller(returns, autolag="AIC")
print(f"    ADF statistic = {adf_stat:.4f}")
print(f"    p-value       = {adf_p:.6f}")
print(f"    crit (5%)     = {adf_crit['5%']:.4f}")
print(f"    => {'reject H0: stationary' if adf_p < 0.05 else 'fail to reject H0'}")
# Returns should be stationary even though prices aren't — that's why we model returns.


# ============================================================
# 4. AR(1) + DETECT ARCH EFFECTS
# ============================================================
print("\n[4] Fitting AR(1) on returns, then testing squared residuals for ARCH effects")
ar_fit = ARIMA(returns, order=(1, 0, 0)).fit()
resid = ar_fit.resid

# Ljung-Box on SQUARED residuals = test for autocorrelation in variance = ARCH effect
lb = acorr_ljungbox(resid ** 2, lags=[5, 10, 20], return_df=True)
print(lb.round(6))
print("    Tiny p-values => squared residuals are serially correlated => ARCH effects present")
print("    => standard regression assumptions are violated => GARCH is justified")


# ============================================================
# 5. FIT GARCH(1,1)
# ============================================================
print("\n[5] Fitting GARCH(1,1) with constant mean, Student-t innovations")
# Student-t handles the fat tails equity returns are famous for; try dist='normal' to compare.
garch = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
garch_fit = garch.fit(disp="off")
print(garch_fit.summary())

# Pull key params
mu     = garch_fit.params["mu"]
omega  = garch_fit.params["omega"]
alpha  = garch_fit.params["alpha[1]"]
beta   = garch_fit.params["beta[1]"]
persistence = alpha + beta
# Unconditional (long-run) variance implied by the model:
uncond_var = omega / (1 - persistence) if persistence < 1 else np.nan
uncond_vol_daily = np.sqrt(uncond_var)
uncond_vol_annual = uncond_vol_daily * np.sqrt(252)

print(f"\n    α (shock reaction)   = {alpha:.4f}")
print(f"    β (vol persistence)  = {beta:.4f}")
print(f"    α + β                = {persistence:.4f}   "
      f"(closer to 1 = more persistent)")
print(f"    implied long-run daily vol  = {uncond_vol_daily:.3f}%")
print(f"    implied long-run annual vol = {uncond_vol_annual:.2f}%")


# ============================================================
# 6. CONDITIONAL VOLATILITY vs REALIZED VOLATILITY
# ============================================================
cond_vol = garch_fit.conditional_volatility            # daily, in %
realized_vol = returns.rolling(20).std()               # 20-day rolling std as proxy

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(realized_vol, label="Realized vol (20-day rolling σ)",
        color="#7f8c8d", lw=1.0)
ax.plot(cond_vol, label="GARCH(1,1) conditional vol",
        color="#c0392b", lw=1.0)
ax.set_title("DAX Volatility — GARCH conditional vs realized")
ax.set_ylabel("Daily vol (%)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_garch_vs_realized.png", bbox_inches="tight")
plt.close()
print(f"\n[6] Saved: {OUT_DIR}/02_garch_vs_realized.png")


# ============================================================
# 7. OUT-OF-SAMPLE FORECAST (20 trading days ahead)
# ============================================================
HORIZON = 20
fc = garch_fit.forecast(horizon=HORIZON, reindex=False)
fc_var = fc.variance.values[-1, :]
fc_vol = np.sqrt(fc_var)              # daily vol path, in %

# Mean-revert toward the unconditional vol — that's the signature of GARCH forecasts
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, HORIZON + 1), fc_vol, marker="o",
        color="#c0392b", label=f"{HORIZON}-day vol forecast")
ax.axhline(uncond_vol_daily, color="gray", ls="--",
           label=f"Long-run daily vol ({uncond_vol_daily:.2f}%)")
ax.set_title("GARCH(1,1) volatility forecast — mean reversion to long-run level")
ax.set_xlabel("Days ahead")
ax.set_ylabel("Forecast daily vol (%)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_vol_forecast.png", bbox_inches="tight")
plt.close()
print(f"[7] Saved: {OUT_DIR}/03_vol_forecast.png")
print(f"    Day 1 forecast vol:  {fc_vol[0]:.3f}%")
print(f"    Day 20 forecast vol: {fc_vol[-1]:.3f}%   (reverting toward {uncond_vol_daily:.3f}%)")


# ============================================================
# 8. VALUE-AT-RISK from the 1-day forecast
# ============================================================
# 95% 1-day VaR under Student-t innovations:
#   VaR_95 = -(mu + t_quantile_5% * sigma_next)
# We use the fitted distribution's quantile, not a normal z-score.
nu = garch_fit.params.get("nu", None)   # degrees of freedom for Student-t
if nu is not None:
    from scipy.stats import t
    q05 = t.ppf(0.05, df=nu)            # left-tail 5% quantile, negative
else:
    from scipy.stats import norm
    q05 = norm.ppf(0.05)

sigma_next = fc_vol[0]                  # 1-day-ahead forecast vol in %
var_95_pct = -(mu + q05 * sigma_next)   # positive number = loss magnitude

print(f"\n[8] 1-day 95% VaR (parametric, from GARCH forecast)")
print(f"    σ_next = {sigma_next:.3f}%   q(0.05) = {q05:.3f}")
print(f"    => VaR_95 ≈ {var_95_pct:.2f}%  (i.e., 1-day loss exceeds this ~5% of days)")


# ============================================================
# 9. BONUS — GJR-GARCH for the leverage effect
# ============================================================
# Equity vol typically rises MORE after negative shocks than after positive ones
# of equal size. Standard GARCH treats them symmetrically; GJR adds an asymmetric term.
print("\n[9] Fitting GJR-GARCH(1,1,1) to capture asymmetry")
gjr = arch_model(returns, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
gjr_fit = gjr.fit(disp="off")
print(gjr_fit.summary().tables[1])      # parameter table only

gamma = gjr_fit.params["gamma[1]"]
print(f"\n    γ (asymmetry term) = {gamma:.4f}")
print(f"    γ > 0 means negative shocks add MORE to next-day variance than positive ones")
print(f"    => classic leverage effect; expected for equity indices like DAX")

# Compare information criteria
print(f"\n    Model comparison (lower = better):")
print(f"      GARCH(1,1)   AIC = {garch_fit.aic:.2f}   BIC = {garch_fit.bic:.2f}")
print(f"      GJR-GARCH    AIC = {gjr_fit.aic:.2f}   BIC = {gjr_fit.bic:.2f}")

# Overlay the two conditional vol series
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(garch_fit.conditional_volatility, label="GARCH(1,1)",
        color="#c0392b", lw=0.9, alpha=0.85)
ax.plot(gjr_fit.conditional_volatility, label="GJR-GARCH (asymmetric)",
        color="#1f4e79", lw=0.9, alpha=0.85)
ax.set_title("Conditional volatility — symmetric vs asymmetric model")
ax.set_ylabel("Daily vol (%)")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_garch_vs_gjr.png", bbox_inches="tight")
plt.close()
print(f"\n    Saved: {OUT_DIR}/04_garch_vs_gjr.png")

print("\nDone. All plots in ./plots/")
