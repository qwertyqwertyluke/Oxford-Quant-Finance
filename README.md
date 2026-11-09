# Week 5 — Time-Series Analysis of U.S. Inflation

Oxford Quantitative Finance tutorial: stationary and non-stationary processes.

## Question
Fit AR(p) models to a monthly inflation series, select p using AIC/BIC, check residual autocorrelation, and run ADF tests on levels and first differences.

## Data
- Seasonally adjusted monthly U.S. Consumer Price Index (CPIAUCSL), January 2020–September 2025.
- Data source: [FRED, Federal Reserve Bank of St. Louis](https://fred.stlouisfed.org/series/CPIAUCSL), originally from the U.S. Bureau of Labor Statistics.
- The CSV uses CPI values rounded to two decimal places.
- Inflation is defined as `100 * (log(CPI_t) - log(CPI_(t-1)))`, i.e. the monthly percentage log change in CPI.

## Run the analysis
With Python installed, from this directory:

```bash
pip install -r requirements.txt
python week5_analysis.py
```

The script prints the model comparison, coefficients, and ADF test results, and saves the inflation and residual-ACF charts.

## Findings
- AR(1)–AR(6) were compared using a common dependent-variable sample (August 2020–September 2025, 62 observations).
- **AR(3)** was preferred by both AIC (−179.03) and BIC (−170.52).
- The selected model's Ljung–Box test at lag 12 had **p = 0.194**, so the null of no residual autocorrelation was not rejected at the 5% level.
- ADF on monthly inflation in levels: statistic **−4.031**, **p = 0.0013**.
- ADF on the first difference of inflation: statistic **−6.295**, **p < 0.001**.

The unit-root tests support treating the **monthly inflation rate** as I(0) over this sample; they do **not** establish that the CPI *price level* is stationary. A relatively short sample including pandemic-era inflation is a limitation.

The AIC/BIC values in the script use comparable Gaussian OLS criteria that omit terms common to all specifications; they are appropriate for ranking the tested AR models, rather than comparison with unadjusted information criteria from other software.

## Files
- `week5_analysis.py` — all calculations and charts
- `week5_cpi_data.csv` — monthly CPI data
- `week5_inflation.png` — monthly inflation chart
- `week5_residual_acf.png` — residual autocorrelation plot
- `week5_computational_tutor_summary.pdf` — one-page tutor summary
- `requirements.txt` — Python package dependencies
