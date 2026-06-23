# Commodity Risk Monitor

Streamlit dashboard for the QF637 commodity-risk project. The research write-up and notebook results live in `notebooks/README.md`; this file is only for installing and running the app.

## First-Time Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

The app uses checked-in processed data under `data/processed`. If those files are missing, the app can download market data when enabled in Settings, or you can run the notebooks in order from `notebooks/01_DataExtraction.ipynb`.

## Dashboard

The `Monitor` tab is the daily risk view: risk state, action note, historical VaR, exposure, drawdown, NAV, signal checks, and today's historical VaR distribution.

The `Stress Test` tab applies Brent shocks to the current proxy book. The loss limit defaults to 10% of current exposure and remains editable. Margin stress is optional: enter initial margin per contract to enable the margin multiplier, margin call, and cash-need impact.

## Smoke Test

```powershell
python -m helpers._smoke_test
```
