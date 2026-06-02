# Income Portfolio Dashboard

## Setup (one time)

```
pip install -r requirements.txt
```

## Run

```
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## How it works

- **Data** refreshes automatically every 7 days when you open the app
- **Scores** recalculate on each refresh using 9 factors (yield, stability, total return, NAV trend, etc.)
- **Recommendations** update based on your holdings vs the target model
- **Backtest** correctly tracks shares × price + reinvested dividends — not just price return

## Files created automatically

| File | Purpose |
|------|---------|
| `my_holdings.json` | Your positions (edit in sidebar) |
| `score_history.csv` | Weekly score log for trend charts |
| `data_cache.json`  | Tracks last refresh date |
| `config.json`      | Your settings |

## Entering holdings

Use the sidebar to enter shares owned and your average cost per share.
Click **Save Holdings** — the trade recommendations update instantly.

## Forcing a refresh

Click **Force Refresh Now** in the sidebar to pull fresh data immediately,
regardless of when the last refresh was.
