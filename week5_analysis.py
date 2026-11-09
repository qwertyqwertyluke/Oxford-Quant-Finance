# Week 5: Time-series computational exercise
# Data: Monthly U.S. CPI (seasonally adjusted), January 2020–September 2025.
# Source: FRED CPIAUCSL (https://fred.stlouisfed.org/series/CPIAUCSL)
# Values here are rounded to 2 decimal places from the previous workbook.
#
# Install if needed: pip install numpy statsmodels matplotlib
# Put week5_cpi_data.csv in the same folder as this script.

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.api import OLS, add_constant
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

folder = Path(__file__).parent
with (folder / "week5_cpi_data.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

months = [r["month"] for r in rows]
cpi = np.array([float(r["cpi_seasonally_adjusted"]) for r in rows])

# Inflation as monthly percent log change, NOT the CPI price level
inflation = 100 * np.diff(np.log(cpi))
inflation_months = months[1:]

# Compare AR(1) through AR(6) on the SAME sample, for fair AIC/BIC comparison.
# All dependent observations run from Aug 2020 to Sep 2025 (62 months).
max_p = 6
y = inflation[max_p:]
n = len(y)

print("AR(p) comparison using monthly log inflation")
print(f"Common estimation sample: {inflation_months[max_p]} to {inflation_months[-1]} (n={n})")
print("Model   AIC       BIC       Ljung-Box p (lag 12)")
results = []

for p in range(1, max_p + 1):
    X = np.column_stack([inflation[max_p-j:len(inflation)-j] for j in range(1, p + 1)])
    fit = OLS(y, add_constant(X)).fit()
    rss = np.sum(fit.resid**2)
    k = p + 1  # Intercept plus p lag coefficients
    # Comparable Gaussian OLS criteria, omitting terms constant across models.
    aic = n * np.log(rss / n) + 2*k
    bic = n * np.log(rss / n) + k*np.log(n)
    # model_df=p adjusts Ljung-Box degrees of freedom for the estimated AR lags.
    lb_p = acorr_ljungbox(fit.resid, lags=[12], model_df=p)["lb_pvalue"].iloc[0]
    results.append((p, aic, bic, lb_p, fit))
    print(f"AR({p})  {aic:8.2f}  {bic:8.2f}     {lb_p:.3f}")

best = min(results, key=lambda x: x[2])  # Select by BIC
p, aic, bic, lb_p, fit = best
print(f"\nSelected by BIC: AR({p}); AIC agrees: AR({min(results, key=lambda x:x[1])[0]})")
print("Fitted coefficients:", np.round(fit.params, 4))

# ADF tests: H0 = a unit root (nonstationarity).
level_adf = adfuller(inflation, regression="c", autolag="AIC")
diff_adf = adfuller(np.diff(inflation), regression="c", autolag="AIC")
print("\nADF test (constant, lag length chosen by AIC)")
print(f"Inflation levels:      statistic={level_adf[0]:.3f}, p={level_adf[1]:.4g}, lags={level_adf[2]}")
print(f"First differences:     statistic={diff_adf[0]:.3f}, p={diff_adf[1]:.4g}, lags={diff_adf[2]}")

# Two uncomplicated plots
plt.figure(figsize=(9, 3.3))
plt.plot(inflation_months, inflation, linewidth=1.3)
plt.axhline(np.mean(inflation), linestyle="--", linewidth=1, color="gray")
plt.title("U.S. monthly inflation: 100 × log(CPI_t / CPI_(t−1))")
plt.ylabel("Percent per month")
plt.xticks(range(0, len(inflation_months), 12),
           [inflation_months[i] for i in range(0, len(inflation_months), 12)],
           rotation=30)
plt.tight_layout()
plt.savefig(folder / "week5_inflation.png", dpi=170)
plt.close()

fig, ax = plt.subplots(figsize=(8, 3.3))
plot_acf(fit.resid, lags=20, alpha=0.05, ax=ax, zero=False)
ax.set_title(f"Residual ACF for selected AR({p})")
fig.tight_layout()
fig.savefig(folder / "week5_residual_acf.png", dpi=170)
plt.close(fig)
