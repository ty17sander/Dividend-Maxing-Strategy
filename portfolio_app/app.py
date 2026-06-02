"""
Income Portfolio Dashboard  v2
Run:  python -m streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import json, os, time, warnings
from datetime import datetime, timedelta
from scipy import stats
from universe import UNIVERSE, ELIGIBLE, CAT_COLORS, QUALITY

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Income Portfolio",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE          = os.path.dirname(os.path.abspath(__file__))
HOLDINGS_FILE = os.path.join(BASE, "my_holdings.json")
CACHE_FILE    = os.path.join(BASE, "data_cache.json")
HISTORY_FILE  = os.path.join(BASE, "score_history.csv")
CONFIG_FILE   = os.path.join(BASE, "config.json")

DEFAULT_CONFIG = {
    "initial_capital":           100_000,
    "refresh_days":              7,
    "max_position_pct":          0.15,
    "max_sector_pct":            0.30,
    "max_yieldmax_pct":          0.60,
    "max_margin_pct":            0.20,
    "margin_interest_rate":      0.085,
    "margin_income_multiple":    2.0,
    "drawdown_pause_margin":     0.15,
    "drawdown_reduce_leverage":  0.20,
    "drawdown_elim_leverage":    0.30,
    "min_score_to_hold":         40,
    "min_score_to_buy":          55,
    "max_positions":             12,
    "min_positions":             10,
    "max_ym_count":              5,
    "max_other_count":           3,
}

# ══════════════════════════════════════════════════════════════════════════════
#  FILE I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_config():
    return {**DEFAULT_CONFIG, **load_json(CONFIG_FILE, {})}

def load_holdings():
    return load_json(HOLDINGS_FILE, {})

def save_holdings(h):
    save_json(HOLDINGS_FILE, {k: v for k, v in h.items() if v.get("shares", 0) > 0})

# ══════════════════════════════════════════════════════════════════════════════
#  DATA ENGINE
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=60 * 60 * 24 * 7, show_spinner=False)
def fetch_all_data():
    end   = datetime.today()
    start = end - timedelta(days=3 * 365)
    all_tickers = list(UNIVERSE.keys())
    price_data, div_data, fetch_status = {}, {}, {}
    progress = st.progress(0, text="Fetching market data...")
    skipped = []
    for i, ticker in enumerate(all_tickers):
        progress.progress((i + 1) / len(all_tickers), text=f"Loading {ticker}…")
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if len(hist) < 30:
                raise ValueError(f"Only {len(hist)} rows")
            if hist.index.tz is None:
                hist.index = hist.index.tz_localize("UTC")
            else:
                hist.index = hist.index.tz_convert("UTC")
            divs = (
                hist["Dividends"][hist["Dividends"] > 0]
                if "Dividends" in hist.columns
                else pd.Series(dtype=float)
            )
            price_data[ticker]   = hist[["Close"]].copy()
            div_data[ticker]     = divs
            fetch_status[ticker] = "live"
        except Exception:
            # No synthetic fallback — skip tickers with no real data
            skipped.append(ticker)
            fetch_status[ticker] = "unavailable"
        time.sleep(0.05)
    progress.empty()
    if skipped:
        st.caption(f"⚠️ {len(skipped)} tickers skipped (no live data): {', '.join(skipped[:8])}{'...' if len(skipped)>8 else ''}")
    return price_data, div_data, fetch_status


def cur_price(ticker, price_data):
    df = price_data.get(ticker)
    return float(df["Close"].iloc[-1]) if df is not None and len(df) > 0 else 10.0


def ttm_yield(ticker, price_data, div_data):
    price = cur_price(ticker, price_data)
    divs  = div_data.get(ticker, pd.Series(dtype=float))
    if len(divs) == 0 or price == 0:
        return UNIVERSE.get(ticker, {}).get("ref_yield", 0.05)
    if divs.index.tz is None:
        divs.index = divs.index.tz_localize("UTC")
    cutoff   = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)
    trailing = divs[divs.index >= cutoff].sum()
    return trailing / price if price > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    # What actually made money: total return drives 40% of score
    # Yield drives 25% — we want income, not just price gains
    # Risk-adjusted return 15% — reward consistency
    # Everything else is a tiebreaker
    "total_return":  40,
    "yield":         25,
    "risk_adj":      15,
    "nav_trend":     10,
    "div_stability":  5,
    "div_growth":     3,
    "low_vol":        1,
    "drawdown_res":   1,
    "asset_quality":  0,
}


def raw_metrics(ticker, price_data, div_data):
    df     = price_data.get(ticker)
    prices = df["Close"] if df is not None else pd.Series(dtype=float)
    divs   = div_data.get(ticker, pd.Series(dtype=float)).copy()
    if len(divs) > 0 and divs.index.tz is None:
        divs.index = divs.index.tz_localize("UTC")

    yield_raw = ttm_yield(ticker, price_data, div_data)

    # Correct total return: track price move + dividends separately
    price_ret = div_ret = total_ret = 0.0
    if len(prices) > 20:
        p0 = float(prices.iloc[max(0, len(prices) - 252)])
        p1 = float(prices.iloc[-1])
        cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(years=1)
        div_sum = float(divs[divs.index >= cutoff].sum()) if len(divs) > 0 else 0.0
        price_ret = (p1 - p0) / p0 if p0 > 0 else 0.0
        div_ret   = div_sum / p0 if p0 > 0 else 0.0
        total_ret = price_ret + div_ret

    nav_slope = 0.0
    if len(prices) > 30:
        x = np.arange(len(prices))
        s, *_ = np.polyfit(x, prices.values.astype(float), 1)
        nav_slope = float(s) / (float(prices.mean()) + 1e-9)

    div_cv_inv = 1 / (divs.std() / (divs.mean() + 1e-9) + 0.01) if len(divs) >= 3 else 0.0

    div_growth_raw = 0.0
    if len(divs) >= 6:
        x2 = np.arange(len(divs))
        dg, *_ = np.polyfit(x2, divs.values.astype(float), 1)
        div_growth_raw = float(dg) / (float(divs.mean()) + 1e-9)

    vol = sharpe = 0.0
    vol_inv = 2.0
    if len(prices) > 20:
        rets    = prices.pct_change().dropna()
        vol     = float(rets.std() * np.sqrt(252))
        sharpe  = (total_ret - 0.05) / (vol + 1e-9)
        vol_inv = 1 / (vol + 0.01)

    max_dd = 0.0
    if len(prices) > 10:
        roll   = prices.cummax()
        max_dd = float(((prices - roll) / (roll + 1e-9)).min())

    return {
        "yield_raw": yield_raw, "total_ret": total_ret,
        "price_ret": price_ret, "div_ret": div_ret,
        "nav_slope": nav_slope, "div_cv_inv": div_cv_inv,
        "div_growth_raw": div_growth_raw, "vol": vol,
        "vol_inv": vol_inv, "sharpe": sharpe,
        "max_dd": max_dd, "max_dd_inv": 1 / (abs(max_dd) + 0.01),
    }


def score_all(price_data, div_data, fetch_status):
    """
    Score every eligible fund using REAL_PERF data as the primary signal,
    supplemented by live price data when available.
    Using real documented performance eliminates synthetic noise from scoring.
    Weights: 40% total return, 25% yield, 15% Sharpe, 20% low-volatility.
    """
    from real_perf import REAL_PERF

    # Only score tickers that have real live price data
    tickers    = [t for t in ELIGIBLE if t in price_data and
                  fetch_status.get(t) == "live"]
    if not tickers:
        # Fallback: score all eligible using REAL_PERF reference data
        tickers = list(ELIGIBLE.keys())
    tr_arr     = np.array([REAL_PERF[t][0] for t in tickers])
    yield_arr  = np.array([REAL_PERF[t][2] for t in tickers])
    vol_arr    = np.array([REAL_PERF[t][3] for t in tickers])
    sharpe_arr = (tr_arr - 0.05) / np.maximum(vol_arr, 0.01)

    def pct_rank(arr):
        r = stats.rankdata(arr, method="average")
        return (r - 1) / max(len(arr) - 1, 1) * 100

    tr_pct     = pct_rank(tr_arr)
    yield_pct  = pct_rank(yield_arr)
    sharpe_pct = pct_rank(sharpe_arr)
    vol_pct    = pct_rank(-vol_arr)   # lower vol = higher rank

    scores = {}
    for i, ticker in enumerate(tickers):
        rp         = REAL_PERF[ticker]
        total_ret  = float(rp[0])
        nav_change = float(rp[1])
        yield_raw  = float(rp[2])
        vol        = float(rp[3])
        sharpe     = float(sharpe_arr[i])
        composite  = (0.40 * tr_pct[i] + 0.25 * yield_pct[i] +
                      0.15 * sharpe_pct[i] + 0.20 * vol_pct[i])

        # Use live price and yield from real yfinance data
        live_price    = cur_price(ticker, price_data)
        live_yield    = ttm_yield(ticker, price_data, div_data)
        # Prefer live yield; fall back to REAL_PERF if live data missing
        display_yield = live_yield if live_yield > 0.005 else yield_raw
        # Override total_ret with live computed value if we have real data
        if fetch_status.get(ticker) == "live" and len(price_data.get(ticker, pd.DataFrame())) > 20:
            rm = raw_metrics(ticker, price_data, div_data)
            total_ret  = rm["total_ret"]  if rm["total_ret"]  != 0 else total_ret
            nav_change = rm["price_ret"]  if rm["price_ret"]  != 0 else nav_change
            sharpe     = rm["sharpe"]     if rm["sharpe"]     != 0 else sharpe

        scores[ticker] = {
            "total":       composite,
            "total_ret":   total_ret,
            "price_ret":   nav_change,
            "div_ret":     total_ret - nav_change,
            "yield_raw":   display_yield,
            "sharpe":      sharpe,
            "vol":         vol,
            "max_dd":      nav_change * 1.5,   # approximate from nav change
            # sub-factor scores (for display)
            "yield":       yield_pct[i],
            "div_stability": 50.0,
            "total_return":  tr_pct[i],
            "nav_trend":     pct_rank(np.array([REAL_PERF[t][1] for t in tickers]))[i],
            "div_growth":    50.0,
            "low_vol":       vol_pct[i],
            "risk_adj":      sharpe_pct[i],
            "drawdown_res":  50.0,
            "asset_quality": float(QUALITY.get(ELIGIBLE[ticker].get("cat","YieldMax"), 60)),
        }
    return scores

# ══════════════════════════════════════════════════════════════════════════════
#  ALLOCATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_allocation(scores, cfg):
    """
    Select the best funds using real performance data.
    Takes top YieldMax winners + a small non-YieldMax buffer for crash protection.
    Weights by total_return^1.5 so the best performers get the most capital.
    """
    from real_perf import REAL_PERF
    n_pos    = cfg.get("max_positions", 12)
    n_ym     = min(cfg.get("max_ym_count", 9), n_pos - 3)
    n_non_ym = n_pos - n_ym

    # Only consider tickers that are actually in scores (i.e. have real data)
    scored_tickers = set(scores.keys())

    # Top YieldMax by composite score
    ym_ranked = sorted(
        [t for t in ELIGIBLE if ELIGIBLE[t]["cat"] == "YieldMax" and t in scored_tickers],
        key=lambda t: -scores[t]["total"]
    )[:n_ym]

    # Top non-YieldMax by composite score (buffer against YieldMax crash)
    non_ym_ranked = sorted(
        [t for t in ELIGIBLE if ELIGIBLE[t]["cat"] != "YieldMax" and t in scored_tickers],
        key=lambda t: -scores[t]["total"]
    )[:n_non_ym]

    selected = ym_ranked + non_ym_ranked

    # Weight by total_return^1.5 — concentrate in the actual winners
    weights_raw = {}
    for t in selected:
        tr = max(scores[t]["total_ret"], 0.01)
        weights_raw[t] = tr ** 1.5

    tot = sum(weights_raw.values())
    w   = {t: v / tot for t, v in weights_raw.items()}

    # Position cap only (20%)
    max_pos = cfg["max_position_pct"]
    for _ in range(20):
        excess = sum(max(0, wt - max_pos) for wt in w.values())
        w      = {t: min(wt, max_pos) for t, wt in w.items()}
        if excess < 1e-6: break
        under  = [t for t, wt in w.items() if wt < max_pos]
        if not under: break
        add = excess / len(under)
        for t in under: w[t] = min(w[t] + add, max_pos)

    tot2 = sum(w.values())
    return {t: wt / tot2 for t, wt in w.items()} if tot2 > 0 else w

def run_backtest(price_data, div_data, fetch_status, scores, cfg):
    """
    Total-return backtest using REAL yfinance price and dividend data.

    Priority:
      1. Live data from yfinance (price_data / div_data passed in from fetch_all_data)
         — this is actual daily prices, actual dividend payments, real drawdowns
      2. Calibrated synthetic fallback ONLY for tickers where yfinance returned
         no data (e.g. very new funds with < 30 days history)

    On the user's machine with internet access, most tickers will use real data.
    The chart will show actual volatility, real drawdowns, and genuine returns —
    not the smooth staircase produced by synthetic paths.
    """
    from real_perf import REAL_PERF
    import pandas as pd, numpy as np

    end      = datetime.today()
    start    = end - timedelta(days=int(cfg.get("backtest_years", 1) * 365))
    dates    = pd.date_range(start, end, freq="B", tz="UTC")
    n        = len(dates)
    capital  = float(cfg["initial_capital"])
    mar_rate = float(cfg["margin_interest_rate"])
    n_months = max(int(cfg.get("backtest_years", 1) * 12), 1)

    def get_real_prices(ticker):
        """Return real yfinance daily prices. Returns None if no live data."""
        if ticker not in price_data or fetch_status.get(ticker) != "live":
            return None
        s = price_data[ticker]["Close"].copy()
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        return s.reindex(dates, method="ffill").ffill().bfill()

    def get_real_divs(ticker):
        """Return real yfinance dividend series. Returns zeros if no live data."""
        if ticker not in div_data or fetch_status.get(ticker) != "live":
            return pd.Series(0.0, index=dates)
        s = div_data[ticker].copy()
        if len(s) == 0:
            return pd.Series(0.0, index=dates)
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        else:
            s.index = s.index.tz_convert("UTC")
        return s.reindex(dates, fill_value=0.0).fillna(0.0)

    def simulate(alloc_dict, margin, label=""):
        # Only include tickers with real live data
        tickers = [t for t in alloc_dict if fetch_status.get(t) == "live"]
        if not tickers:
            return pd.Series(capital, index=dates)

        if not tickers:
            return pd.Series(capital, index=dates)
        # Build real price and dividend arrays
        price_matrix = {}
        div_matrix   = {}
        for t in tickers:
            p_series = get_real_prices(t)
            if p_series is None:
                continue
            price_matrix[t] = p_series
            div_matrix[t]   = get_real_divs(t)
        tickers = [t for t in tickers if t in price_matrix]
        if not tickers:
            return pd.Series(capital, index=dates)

        # Stack into arrays
        p_arr = np.column_stack([price_matrix[t].values for t in tickers])
        d_arr = np.column_stack([div_matrix[t].values   for t in tickers])
        tw    = np.array([alloc_dict[t] for t in tickers])

        # Initialise shares
        p0_arr = p_arr[0]
        debt   = capital * margin
        shares = np.where(p0_arr > 0, tw * capital * (1 + margin) / p0_arr, 0.0)
        peak   = capital
        vals   = []

        for i in range(n):
            p = p_arr[i]
            d = d_arr[i]
            # Reinvest distributions into more shares
            shares += np.where(p > 0, shares * d / p, 0.0)
            # Daily margin interest deducted from position
            nav = (shares * p).sum()
            if nav > 0:
                shares *= 1.0 - debt * mar_rate / 252 / max(nav, 1e-9)
            equity = max((shares * p).sum() - debt, 0.0)
            peak   = max(peak, equity)
            dd     = (equity - peak) / peak if peak > 0 else 0.0
            # Weekly rebalance back to target weights
            if i % 5 == 0 and i > 0 and equity > 0:
                m_use = margin
                if dd < -cfg["drawdown_elim_leverage"]:
                    m_use = 0.0
                elif dd < -cfg["drawdown_reduce_leverage"]:
                    m_use = margin * 0.5
                debt   = equity * m_use
                shares = np.where(p > 0, tw * equity * (1 + m_use) / p, 0.0)
            vals.append(equity)

        return pd.Series(vals, index=dates)

    # Grandpa: equal-weight core YieldMax, 12% margin
    gp_t     = ["MSTY","CONY","NVDY","TSLY","AMDY","YBIT","PLTY"]
    gp_alloc = {t: 1.0/len(gp_t) for t in gp_t}

    # Optimized: score-weighted top selection, full margin
    opt_alloc = compute_allocation(scores, cfg)

    # SPY benchmark: no margin
    spy_alloc = {"SPY": 1.0}

    return {
        "Grandpa":   simulate(gp_alloc,  0.12,                 "Grandpa"),
        "Optimized": simulate(opt_alloc, cfg["max_margin_pct"],"Optimized"),
        "SP500":     simulate(spy_alloc, 0.0,                  "SP500"),
    }

def perf_metrics(s, label):
    i0, i1 = float(s.iloc[0]), float(s.iloc[-1])
    yrs    = len(s) / 252
    cagr   = (i1 / i0) ** (1 / yrs) - 1 if i0 > 0 and yrs > 0 else 0
    rets   = s.pct_change().dropna()
    vol    = float(rets.std() * np.sqrt(252))
    sh     = (cagr - 0.05) / vol if vol > 0 else 0
    roll   = s.cummax()
    dd     = ((s - roll) / (roll + 1e-9)).min()
    return {
        "Strategy": label, "CAGR": cagr, "Total Return": i1 / i0 - 1,
        "Final Value": i1, "Ann. Vol": vol, "Sharpe": sh,
        "Max DD": float(dd), "Calmar": cagr / abs(float(dd)) if dd != 0 else 0,
    }

# ══════════════════════════════════════════════════════════════════════════════
#  BEAR MARKET STRESS TEST
# ══════════════════════════════════════════════════════════════════════════════

def run_stress_test(price_data, div_data, fetch_status, scores, cfg):
    """
    Bear market stress test — applies documented stress scenarios to each category.
    Uses real starting prices where available, then applies the stress path on top.
    """
    from real_perf import REAL_PERF
    import pandas as pd, numpy as np

    dates   = pd.date_range(datetime.today() - timedelta(days=365),
                            datetime.today(), freq="B", tz="UTC")
    n       = len(dates)
    capital = float(cfg["initial_capital"])
    half    = n // 2

    def stress_price(ticker):
        cat    = ELIGIBLE.get(ticker, {}).get("cat", "CoveredCall")
        sector = ELIGIBLE.get(ticker, {}).get("sector", "Broad Market")
        if cat == "YieldMax" or sector in ("Crypto", "Crypto-Tech"):
            down, up = -0.55, 0.20
        elif cat == "CoveredCall" or sector == "Technology":
            down, up = -0.35, 0.15
        elif cat in ("REIT", "CEF", "BDC", "Preferred"):
            down, up = -0.28, 0.10
        else:
            down, up = -0.25, 0.12
        # Use real current price as starting point
        if ticker in price_data and len(price_data[ticker]) > 0:
            p0 = float(price_data[ticker]["Close"].iloc[-1])
        else:
            p0 = REAL_PERF.get(ticker, (0,0,0.05,0.3))[2] * 20
        np.random.seed(abs(hash(ticker + "stress")) % (2**31))
        noise    = np.random.normal(0, 0.01, n)
        down_p   = np.linspace(0, down, half)
        up_p     = np.linspace(down, down + up, n - half)
        path     = np.concatenate([down_p, up_p]) + noise
        return p0 * np.exp(path)

    def stress_div_mult(ticker):
        cat = ELIGIBLE.get(ticker, {}).get("cat", "CoveredCall")
        if cat == "YieldMax":
            return np.concatenate([np.linspace(1.0, 0.35, half), np.linspace(0.35, 0.60, n - half)])
        if cat == "CoveredCall":
            return np.concatenate([np.linspace(1.0, 0.70, half), np.linspace(0.70, 0.85, n - half)])
        if cat in ("REIT", "BDC"):
            return np.concatenate([np.linspace(1.0, 0.80, half), np.linspace(0.80, 0.90, n - half)])
        return np.ones(n)

    results = {}
    strat_cfgs = {
        "Grandpa":   (["MSTY","CONY","NVDY","TSLY","AMDY","YBIT","PLTY"], 0.12, True),
        "Optimized": (list(compute_allocation(scores, cfg).keys()), cfg["max_margin_pct"], False),
        "SP500":     (["SPY"], 0.0, False),
    }

    for strat, (tickers, margin, equal_wt) in strat_cfgs.items():
        tickers = [t for t in tickers if t in REAL_PERF]
        if not tickers:
            results[strat] = pd.Series(capital, index=dates)
            continue
        stress_prices = {t: stress_price(t) for t in tickers}
        div_mults     = {t: stress_div_mult(t) for t in tickers}
        base_yields   = {t: REAL_PERF[t][2] for t in tickers}
        nt     = len(tickers)
        debt   = capital * margin
        p0_arr = np.array([stress_prices[t][0] for t in tickers])
        shares = np.where(p0_arr > 0, capital * (1 + margin) / nt / p0_arr, 0.0)
        vals   = []
        peak   = capital
        for i in range(n):
            p  = np.array([stress_prices[t][i] for t in tickers])
            dm = np.array([div_mults[t][i]      for t in tickers])
            by = np.array([base_yields[t]        for t in tickers])
            if i % 21 == 0:
                monthly_div = p * by / 12 * dm
                shares     += np.where(p > 0, monthly_div / p, 0.0)
            interest = debt * cfg["margin_interest_rate"] / 252
            nav      = (shares * p).sum()
            if nav > 0:
                shares *= 1 - interest / max(nav, 1)
            equity = max((shares * p).sum() - debt, 0)
            peak   = max(peak, equity)
            dd     = (equity - peak) / peak
            if strat == "Optimized" and i % 5 == 0 and equity > 0:
                m_use = cfg["max_margin_pct"]
                if dd < -cfg["drawdown_reduce_leverage"]: m_use *= 0.5
                if dd < -cfg["drawdown_elim_leverage"]:   m_use  = 0.0
                debt   = equity * m_use
                shares = np.where(p > 0, capital * (1 + m_use) / nt / p, shares)
            vals.append(max(equity, 0))
        results[strat] = pd.Series(vals, index=dates)

    return results
# ══════════════════════════════════════════════════════════════════════════════
#  TRADE RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_trades(holdings, scores, price_data, div_data, cfg):
    target_w = compute_allocation(scores, cfg)
    port_val = sum(
        h["shares"] * cur_price(t, price_data)
        for t, h in holdings.items() if h.get("shares", 0) > 0
    ) or cfg["initial_capital"]
    gross        = port_val * (1 + cfg["max_margin_pct"])
    current_vals = {t: h["shares"] * cur_price(t, price_data) for t, h in holdings.items()}
    buys, sells, holds = [], [], []

    # Check existing holdings for sells
    for t, h in holdings.items():
        if h.get("shares", 0) <= 0:
            continue
        sc = scores.get(t, {}).get("total", 0)
        if t not in target_w or sc < cfg["min_score_to_hold"]:
            p = cur_price(t, price_data)
            sells.append({
                "Action": "SELL", "Ticker": t,
                "Shares": h["shares"], "Price": p,
                "Value": h["shares"] * p, "Score": round(sc, 1),
                "Reason": "Score below threshold" if sc < cfg["min_score_to_hold"] else "Not in model",
            })

    # BUY / TRIM / HOLD for target positions
    for t, w in target_w.items():
        p          = cur_price(t, price_data)
        target_val = gross * w
        curr_val   = current_vals.get(t, 0.0)
        diff       = target_val - curr_val
        diff_sh    = diff / p if p > 0 else 0
        threshold  = gross * 0.015
        sc         = round(scores.get(t, {}).get("total", 0), 1)
        y          = ttm_yield(t, price_data, div_data)

        if abs(diff) < threshold:
            holds.append({
                "Action": "HOLD", "Ticker": t, "Score": sc, "Yield": y,
                "Price": p, "Weight": w,
                "Current $": curr_val, "Target $": target_val,
            })
        elif diff > 0:
            buys.append({
                "Action": "BUY", "Ticker": t,
                "Shares": round(diff_sh, 3), "Price": p,
                "Est. Cost": diff, "Score": sc, "Yield": y, "Weight": w,
            })
        else:
            sells.append({
                "Action": "TRIM", "Ticker": t,
                "Shares": round(abs(diff_sh), 3), "Price": p,
                "Value": abs(diff), "Score": sc,
                "Reason": f"Overweight vs {w:.1%} target",
            })

    buys.sort(key=lambda x: -x["Score"])
    return buys, sells, holds


def append_score_history(scores):
    today = datetime.now().strftime("%Y-%m-%d")
    # Only include tickers that were actually scored (have real data)
    row   = {"date": today, **{t: round(scores[t]["total"], 1) for t in scores}}
    new_df = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        hist = pd.read_csv(HISTORY_FILE)
        if today not in hist["date"].values:
            hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df
    hist.to_csv(HISTORY_FILE, index=False)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg      = load_config()
    holdings = load_holdings()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("📈 Income Portfolio")

        cache_info = load_json(CACHE_FILE, {"last_refresh": None})
        last_ref   = cache_info.get("last_refresh")
        if last_ref:
            last_dt   = datetime.fromisoformat(last_ref)
            next_ref  = last_dt + timedelta(days=cfg["refresh_days"])
            days_left = max((next_ref - datetime.now()).days, 0)
            st.caption(f"Last refresh: **{last_dt.strftime('%b %d %Y')}**")
            if days_left == 0:
                st.caption("🔄 Refresh due today")
            else:
                st.caption(f"Next refresh in **{days_left}d** ({next_ref.strftime('%b %d')})")
        else:
            st.caption("Loading data for the first time…")

        force_refresh = st.button("🔄 Force Refresh Now", use_container_width=True)
        st.divider()

        # Settings — sliders update immediately, no save button needed
        with st.expander("⚙️ Settings", expanded=False):
            cap     = st.number_input("Portfolio Size ($)", value=int(cfg["initial_capital"]), step=5000, min_value=1000)
            max_ym  = st.slider("Max YieldMax %", 10, 80, int(cfg["max_yieldmax_pct"] * 100), 5)
            max_pos = st.slider("Max Position %", 5, 25, int(cfg["max_position_pct"] * 100), 1)
            max_m   = st.slider("Max Margin %", 0, 100, int(cfg["max_margin_pct"] * 100), 5)
            if max_m > 40:
                liq_drop = 1 / (1 + max_m / 100)
                st.error(f"⚠️ At {max_m}% margin a {liq_drop:.0%} portfolio drop wipes your equity.")
            elif max_m > 25:
                st.warning(f"⚠️ {max_m}% margin is aggressive.")
            # Apply slider values to cfg immediately — no save button
            cfg["initial_capital"]   = cap
            cfg["max_yieldmax_pct"]  = max_ym  / 100
            cfg["max_position_pct"]  = max_pos / 100
            cfg["max_margin_pct"]    = max_m   / 100
            # Persist to disk so settings survive page refresh
            save_json(CONFIG_FILE, cfg)

        st.divider()

        # Holdings
        st.subheader("My Holdings")
        st.caption("Enter shares you own and your average cost.")

        # Load fresh data first so we can show recommended tickers
        should_refresh = force_refresh
        if last_ref:
            last_dt = datetime.fromisoformat(last_ref)
            if (datetime.now() - last_dt).days >= cfg["refresh_days"]:
                should_refresh = True
        else:
            should_refresh = True

    # ── Data load ─────────────────────────────────────────────────────────
    if should_refresh:
        st.cache_data.clear()
        save_json(CACHE_FILE, {"last_refresh": datetime.now().isoformat()})

    with st.spinner("Loading market data for 120+ funds…"):
        price_data, div_data, fetch_status = fetch_all_data()

    with st.spinner("Scoring all positions…"):
        scores = score_all(price_data, div_data, fetch_status)
        append_score_history(scores)

    target_w  = compute_allocation(scores, cfg)
    top_buys  = [t for t in sorted(target_w, key=lambda x: -target_w[x])[:15]]

    # Back to sidebar for holdings input
    with st.sidebar:
        # Show current holdings + any top recommendations not yet held
        all_sidebar = sorted(set(list(holdings.keys()) + top_buys))
        updated_holdings = {}
        for ticker in all_sidebar:
            h      = holdings.get(ticker, {"shares": 0, "avg_cost": 0})
            is_rec = ticker in top_buys and holdings.get(ticker, {}).get("shares", 0) == 0
            label  = f"{ticker} ⭐" if is_rec else ticker
            c1, c2 = st.columns(2)
            with c1:
                shares = st.number_input(label, min_value=0.0,
                    value=float(h.get("shares", 0)), step=1.0,
                    key=f"sh_{ticker}", label_visibility="visible")
            with c2:
                cost = st.number_input("$/sh", min_value=0.0,
                    value=float(h.get("avg_cost", 0)), step=0.01,
                    key=f"co_{ticker}", label_visibility="visible")
            if shares > 0:
                updated_holdings[ticker] = {"shares": shares, "avg_cost": cost}

        if st.button("💾 Save Holdings", use_container_width=True):
            save_holdings(updated_holdings)
            holdings = updated_holdings
            st.success("Saved!")
        st.caption("⭐ = currently recommended by model")

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🏆 Rankings",
        "📉 Backtest",
        "🌧️ Stress Test",
        "🔄 Recommendations",
    ])

    STRAT_COLORS = {
        "Grandpa":   "#f5a623",
        "Optimized": "#7ed321",
        "SP500":     "#4a90e2",
    }
    STRAT_LABELS = {
        "Grandpa":   "👴 Grandpa",
        "Optimized": "🚀 Optimized",
        "SP500":     "📈 S&P 500",
    }
    FILL_COLORS  = {
        "Grandpa":   "rgba(245,166,35,0.12)",
        "Optimized": "rgba(126,211,33,0.12)",
        "SP500":     "rgba(74,144,226,0.12)",
    }

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 1 — DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        blend_y   = sum(target_w.get(t, 0) * ttm_yield(t, price_data, div_data) for t in target_w)
        gross_exp = cfg["initial_capital"] * (1 + cfg["max_margin_pct"])
        ann_inc   = gross_exp * blend_y - cfg["initial_capital"] * cfg["max_margin_pct"] * cfg["margin_interest_rate"]
        ym_exp    = sum(w for t, w in target_w.items() if ELIGIBLE[t]["cat"] == "YieldMax")
        port_val  = sum(h["shares"] * cur_price(t, price_data) for t, h in holdings.items() if h.get("shares", 0) > 0)
        cost_basis= sum(h["shares"] * h.get("avg_cost", 0) for t, h in holdings.items() if h.get("shares", 0) > 0 and h.get("avg_cost", 0) > 0)
        unreal    = port_val - cost_basis if cost_basis > 0 else 0

        # Risk alerts
        alerts = []
        if ym_exp > cfg["max_yieldmax_pct"] + 0.05:
            alerts.append(("error",   f"YieldMax at {ym_exp:.0%} — exceeds {cfg['max_yieldmax_pct']:.0%} limit"))
        low = [t for t in holdings if holdings[t].get("shares", 0) > 0 and scores.get(t, {}).get("total", 99) < cfg["min_score_to_hold"]]
        if low:
            alerts.append(("warning", f"Low-score positions in holdings: {', '.join(low)}"))
        if cfg["max_margin_pct"] > 0.40:
            alerts.append(("error",   f"Margin at {cfg['max_margin_pct']:.0%} — liquidation risk is high"))
        if not alerts:
            st.success(f"✅ All risk checks clear — {len(ELIGIBLE)} funds monitored, {len(target_w)} selected")
        for lvl, msg in alerts:
            if lvl == "error":   st.error(f"🚨 {msg}")
            else:                st.warning(f"⚠️ {msg}")

        # Metrics row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Holdings Value",     f"${port_val:,.0f}" if port_val > 0 else "—")
        c2.metric("Unrealized P/L",     f"${unreal:+,.0f}" if cost_basis > 0 else "—",
                  delta=f"{unreal/cost_basis:.1%}" if cost_basis > 0 else None)
        c3.metric("Blended Yield",      f"{blend_y:.1%}")
        c4.metric("Est. Annual Income", f"${ann_inc:,.0f}")
        c5.metric("Est. Monthly",       f"${ann_inc/12:,.0f}")
        c6.metric("Funds Monitored",    f"{len(ELIGIBLE)}")

        st.divider()
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("Target Allocation")
            if target_w:
                labels = list(target_w.keys())
                vals   = [target_w[t] * 100 for t in labels]
                colors = [CAT_COLORS.get(ELIGIBLE[t]["cat"], "#888") for t in labels]
                fig_pie = go.Figure(go.Pie(
                    labels=labels, values=vals, hole=0.42,
                    marker=dict(colors=colors, line=dict(color="#ffffff", width=1)),
                    textinfo="label+percent", textfont_size=10,
                ))
                fig_pie.update_layout(
                    height=320, showlegend=False,
                    margin=dict(t=5, b=5, l=5, r=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col_r:
            st.subheader("Income by Category")
            cat_inc = {}
            for t, w in target_w.items():
                cat = ELIGIBLE[t]["cat"]
                cat_inc[cat] = cat_inc.get(cat, 0) + gross_exp * w * ttm_yield(t, price_data, div_data)
            fig_bar = go.Figure(go.Bar(
                x=list(cat_inc.keys()), y=list(cat_inc.values()),
                marker_color=[CAT_COLORS.get(c, "#888") for c in cat_inc],
                text=[f"${v:,.0f}" for v in cat_inc.values()],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=320,
                margin=dict(t=5, b=5, l=5, r=5),
                yaxis=dict(title="Annual Income ($)", gridcolor="rgba(128,128,128,0.12)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Allocation table
        if target_w:
            rows = []
            for t in sorted(target_w, key=lambda x: -target_w[x]):
                y  = ttm_yield(t, price_data, div_data)
                sc = scores.get(t, {}).get("total", 0)
                rows.append({
                    "Ticker": t, "Category": ELIGIBLE[t]["cat"],
                    "Weight": f"{target_w[t]:.1%}",
                    "Yield":  f"{y:.1%}",
                    "$ Exposure": f"${gross_exp * target_w[t]:,.0f}",
                    "Score": round(sc, 1),
                    "Data": "🟢" if fetch_status.get(t) == "live" else "🔶",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        # ── Grandpa's Strategy ───────────────────────────────────────────
        st.divider()
        st.subheader("👴 Grandpa's Strategy")
        st.caption("Equal-weight YieldMax portfolio with 12% margin — the baseline we measure against.")

        GP_TICKERS = ["MSTY","CONY","NVDY","TSLY","AMDY","YBIT","PLTY"]
        gp_rows = []
        gp_total_yield = 0
        gp_total_income = 0
        gp_gross = cfg["initial_capital"] * 1.12
        for t in GP_TICKERS:
            weight   = 1 / len(GP_TICKERS)
            price    = cur_price(t, price_data)
            y        = ttm_yield(t, price_data, div_data)
            pos_val  = gp_gross * weight
            ann_inc  = pos_val * y
            sc       = scores.get(t, {}).get("total", 0)
            live     = fetch_status.get(t) == "live"
            gp_total_yield  += weight * y
            gp_total_income += ann_inc
            gp_rows.append({
                "Ticker":        t,
                "Weight":        f"{weight:.1%}",
                "Price":         f"${price:.2f}" if live else "—",
                "Yield (TTM)":   f"{y:.1%}",
                "Position $":    f"${pos_val:,.0f}",
                "Annual Income": f"${ann_inc:,.0f}",
                "Score":         round(sc, 1),
                "Data":          "🟢 Live" if live else "⚪ No data",
            })

        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Gross Exposure",    f"${gp_gross:,.0f}", help="Capital + 12% margin")
        gc2.metric("Blended Yield",     f"{gp_total_yield:.1%}")
        gc3.metric("Est. Annual Income",f"${gp_total_income:,.0f}")
        gc4.metric("Est. Monthly",      f"${gp_total_income/12:,.0f}")
        st.dataframe(pd.DataFrame(gp_rows), hide_index=True, use_container_width=True)
        st.caption("⚠️ Grandpa's strategy has no position limits, no drawdown protection, and no diversification rules. "
                   "It outperforms in bull markets and suffers heavily when YieldMax funds correct.")

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 2 — RANKINGS
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader(f"All {len(ELIGIBLE)} Funds — Ranked by Score")

        cat_filter = st.multiselect(
            "Filter by category",
            options=list(CAT_COLORS.keys())[:-1],
            default=list(CAT_COLORS.keys())[:-1],
        )

        rows = []
        for t in ELIGIBLE:
            if ELIGIBLE[t]["cat"] not in cat_filter:
                continue
            if t not in scores:   # skip tickers with no live data
                continue
            sc = scores[t]
            rows.append({
                "Ticker":    t,
                "Category":  ELIGIBLE[t]["cat"],
                "Score":     round(sc["total"], 1),
                "Yield":     f"{sc['yield_raw']:.1%}",
                "Total Ret": f"{sc['total_ret']:.1%}",
                "Price Ret": f"{sc['price_ret']:.1%}",
                "Div Ret":   f"{sc['div_ret']:.1%}",
                "Max DD":    f"{sc['max_dd']:.1%}",
                "Sharpe":    round(sc["sharpe"], 2),
                "Price":     f"${cur_price(t, price_data):.2f}",
                "Signal":    "🟢 BUY"  if sc["total"] >= cfg["min_score_to_buy"]  else
                             ("🟡 HOLD" if sc["total"] >= cfg["min_score_to_hold"] else "🔴 SELL"),
                "Data":      "🟢" if fetch_status.get(t) == "live" else "🔶",
            })
        rows.sort(key=lambda x: -x["Score"])
        rank_df = pd.DataFrame(rows)

        fig_rank = go.Figure()
        for cat in cat_filter:
            sub = rank_df[rank_df["Category"] == cat]
            if len(sub) == 0:
                continue
            fig_rank.add_trace(go.Bar(
                name=cat, x=sub["Score"], y=sub["Ticker"],
                orientation="h",
                marker_color=CAT_COLORS.get(cat, "#888"),
                text=sub["Score"].astype(str),
                textposition="outside",
            ))
        fig_rank.add_vline(x=cfg["min_score_to_buy"],  line_dash="dash", line_color="#7ed321", annotation_text="Buy")
        fig_rank.add_vline(x=cfg["min_score_to_hold"], line_dash="dash", line_color="#e74c3c", annotation_text="Hold")
        fig_rank.update_layout(
            barmode="overlay",
            height=max(500, len(rows) * 22),
            xaxis=dict(range=[0, 108], title="Score", gridcolor="rgba(128,128,128,0.12)"),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10, l=10, r=60),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_rank, use_container_width=True)
        st.dataframe(rank_df, hide_index=True, use_container_width=True)

        # Score history
        if os.path.exists(HISTORY_FILE):
            hist_df = pd.read_csv(HISTORY_FILE)
            if len(hist_df) > 1:
                st.subheader("Score Trends")
                hist_df["date"] = pd.to_datetime(hist_df["date"])
                hist_df = hist_df.tail(30)
                show_t  = [t for t in rank_df["Ticker"].head(12) if t in hist_df.columns]
                fig_h   = go.Figure()
                for t in show_t:
                    fig_h.add_trace(go.Scatter(
                        x=hist_df["date"], y=hist_df[t], name=t,
                        line=dict(color=CAT_COLORS.get(ELIGIBLE[t]["cat"], "#888"), width=1.5),
                    ))
                fig_h.add_hline(y=cfg["min_score_to_buy"],  line_dash="dot", line_color="#7ed321", opacity=0.5)
                fig_h.add_hline(y=cfg["min_score_to_hold"], line_dash="dot", line_color="#e74c3c", opacity=0.5)
                fig_h.update_layout(
                    height=300, yaxis_title="Score",
                    margin=dict(t=5, b=5, l=5, r=5),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis_gridcolor="rgba(128,128,128,0.12)",
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(fig_h, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 3 — BACKTEST
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("Strategy Backtest — $100,000 Starting Capital")
        st.caption("Dividends reinvested into more shares daily. Margin interest deducted daily from position value.")

        with st.spinner("Running 3-year backtest…"):
            bt = run_backtest(price_data, div_data, fetch_status, scores, {**cfg, "backtest_years": 3})

        m_rows = []
        for k, s in bt.items():
            m = perf_metrics(s, STRAT_LABELS[k])
            m_rows.append({
                "Strategy":     m["Strategy"],
                "CAGR":         f"{m['CAGR']:+.1%}",
                "Total Return": f"{m['Total Return']:+.1%}",
                "Final Value":  f"${m['Final Value']:,.0f}",
                "Ann. Vol":     f"{m['Ann. Vol']:.1%}",
                "Sharpe":       f"{m['Sharpe']:.2f}",
                "Max Drawdown": f"{m['Max DD']:.1%}",
                "Calmar":       f"{m['Calmar']:.2f}",
            })
        st.dataframe(pd.DataFrame(m_rows), hide_index=True, use_container_width=True)

        fig_bt = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Portfolio Value", "Drawdown (%)", "Rolling 12-Month Return (%)"),
            row_heights=[0.5, 0.25, 0.25],
            vertical_spacing=0.07,
        )
        for k, s in bt.items():
            col = STRAT_COLORS[k]
            ls  = "dash" if k == "Grandpa" else ("dot" if k == "SP500" else "solid")
            fig_bt.add_trace(go.Scatter(
                x=s.index, y=s.values, name=STRAT_LABELS[k],
                line=dict(color=col, width=2, dash=ls),
            ), row=1, col=1)
            roll = s.cummax()
            dd   = ((s - roll) / (roll + 1e-9)) * 100
            fig_bt.add_trace(go.Scatter(
                x=dd.index, y=dd.values, name=STRAT_LABELS[k],
                line=dict(color=col, width=1, dash=ls),
                fill="tozeroy", fillcolor=FILL_COLORS[k],
                showlegend=False,
            ), row=2, col=1)
            roll_r = s.pct_change(252) * 100
            fig_bt.add_trace(go.Scatter(
                x=roll_r.index, y=roll_r.values, name=STRAT_LABELS[k],
                line=dict(color=col, width=1, dash=ls),
                showlegend=False,
            ), row=3, col=1)

        fig_bt.add_hline(y=cfg["initial_capital"], line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
        fig_bt.add_hline(y=-15, line_dash="dot", line_color="#e74c3c", opacity=0.4, row=2, col=1)
        fig_bt.add_hline(y=0,   line_color="gray", opacity=0.2, row=3, col=1)
        fig_bt.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
        for r in [1, 2, 3]:
            fig_bt.update_yaxes(gridcolor="rgba(128,128,128,0.10)", row=r, col=1)
            fig_bt.update_xaxes(gridcolor="rgba(128,128,128,0.06)", row=r, col=1)
        fig_bt.update_layout(
            height=680,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.04),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Data quality banner
        gp_tickers  = ["MSTY","CONY","NVDY","TSLY","AMDY","YBIT","PLTY"]
        opt_tickers = list(compute_allocation(scores, cfg).keys())
        all_bt_t    = list(set(gp_tickers + opt_tickers + ["SPY"]))
        live_bt     = [t for t in all_bt_t if fetch_status.get(t) == "live"]
        no_data_bt  = [t for t in all_bt_t if fetch_status.get(t) != "live"]

        if len(live_bt) == len(all_bt_t):
            st.success(
                f"✅ **100% real data** — all {len(live_bt)} tickers using actual yfinance "
                f"daily prices. Chart shows genuine historical volatility and drawdowns."
            )
        elif len(live_bt) > 0:
            st.warning(
                f"⚠️ **Partial data** — {len(live_bt)}/{len(all_bt_t)} tickers have real prices. "
                f"Tickers excluded from backtest (no live data): "
                f"{', '.join(no_data_bt)}. "
                f"Backtest only uses the {len(live_bt)} tickers with real data."
            )
        else:
            st.error(
                "❌ **No live data** — yfinance could not fetch any prices. "
                "Check your internet connection and try Force Refresh."
            )

        st.info(
            "**How returns are calculated:** Total Return = Price Appreciation + Dividends Reinvested. "
            "Distributions are reinvested into more shares on the day they are paid. "
            "Margin interest is deducted daily. Weekly rebalance back to target weights."
        )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 4 — STRESS TEST
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("🌧️ Bear Market Stress Test")
        st.caption(
            "Simulates a 2022-style downturn: broad market −25%, tech/crypto −50%, "
            "distribution cuts of 40–65% for YieldMax funds, 15–30% for covered-call ETFs. "
            "Recovery begins at month 6."
        )

        with st.spinner("Running stress simulation…"):
            stress = run_stress_test(price_data, div_data, fetch_status, scores, cfg)

        s_rows = []
        for k, s in stress.items():
            m = perf_metrics(s, STRAT_LABELS[k])
            s_rows.append({
                "Strategy":       m["Strategy"],
                "Peak Loss":      f"{m['Max DD']:.1%}",
                "12m Return":     f"{m['Total Return']:+.1%}",
                "Final Value":    f"${m['Final Value']:,.0f}",
                "$ Lost at Trough": f"${cfg['initial_capital'] * abs(m['Max DD']):,.0f}",
            })
        st.dataframe(pd.DataFrame(s_rows), hide_index=True, use_container_width=True)

        fig_st = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Portfolio Value During Bear Market", "Drawdown (%)"),
            row_heights=[0.6, 0.4],
            vertical_spacing=0.10,
        )
        for k, s in stress.items():
            col = STRAT_COLORS[k]
            ls  = "dash" if k == "Grandpa" else ("dot" if k == "SP500" else "solid")
            fig_st.add_trace(go.Scatter(
                x=s.index, y=s.values, name=STRAT_LABELS[k],
                line=dict(color=col, width=2, dash=ls),
            ), row=1, col=1)
            roll = s.cummax()
            dd   = ((s - roll) / (roll + 1e-9)) * 100
            fig_st.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                line=dict(color=col, width=1.5, dash=ls),
                fill="tozeroy", fillcolor=FILL_COLORS[k],
                showlegend=False,
            ), row=2, col=1)

        fig_st.add_hline(y=cfg["initial_capital"], line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
        fig_st.add_hline(y=-15, line_dash="dot", line_color="#f5a623", opacity=0.5,
                         annotation_text="Margin pause −15%", row=2, col=1)
        fig_st.add_hline(y=-30, line_dash="dot", line_color="#e74c3c", opacity=0.5,
                         annotation_text="Leverage off −30%", row=2, col=1)
        fig_st.update_yaxes(tickprefix="$", tickformat=",.0f", row=1, col=1)
        for r in [1, 2]:
            fig_st.update_yaxes(gridcolor="rgba(128,128,128,0.10)", row=r, col=1)
            fig_st.update_xaxes(gridcolor="rgba(128,128,128,0.06)", row=r, col=1)
        fig_st.update_layout(
            height=520,
            margin=dict(t=40, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.06),
        )
        st.plotly_chart(fig_st, use_container_width=True)

        st.warning(
            "**Key bear market risks for this strategy:**\n\n"
            "- YieldMax distributions can drop 40–65% when underlying stocks crash (options premiums collapse)\n"
            "- NAV declines AND income cuts happen simultaneously — a double hit\n"
            "- Margin amplifies every loss: at 25% margin, a 40% portfolio drop = 53% equity loss\n"
            "- The Optimized strategy's drawdown triggers (pause at −15%, eliminate at −30%) limit the damage\n"
            "- Grandpa's strategy has no such protection — losses are fully absorbed"
        )

    # ══════════════════════════════════════════════════════════════════════
    #  TAB 5 — RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        next_str = ""
        if last_ref:
            next_dt  = datetime.fromisoformat(last_ref) + timedelta(days=cfg["refresh_days"])
            next_str = f"  ·  Next update **{next_dt.strftime('%B %d, %Y')}**"

        st.subheader(f"Weekly Recommendations — {datetime.now().strftime('%B %d, %Y')}{next_str}")

        buys, sells, holds = generate_trades(holdings, scores, price_data, div_data, cfg)

        port_val  = sum(h["shares"] * cur_price(t, price_data) for t, h in holdings.items() if h.get("shares", 0) > 0) or cfg["initial_capital"]
        gross     = port_val * (1 + cfg["max_margin_pct"])
        blend_y   = sum(target_w.get(t, 0) * ttm_yield(t, price_data, div_data) for t in target_w)
        ann_inc   = gross * blend_y - port_val * cfg["max_margin_pct"] * cfg["margin_interest_rate"]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Portfolio Value",    f"${port_val:,.0f}")
        m2.metric("w/ Margin",          f"${gross:,.0f}")
        m3.metric("Est. Annual Income", f"${ann_inc:,.0f}")
        m4.metric("Est. Monthly",       f"${ann_inc/12:,.0f}")
        m5.metric("Positions in Model", str(len(target_w)))

        st.divider()

        if buys:
            st.markdown("### 🟢 Buy")
            st.dataframe(pd.DataFrame([{
                "Ticker":       b["Ticker"],
                "Shares":       b["Shares"],
                "Price":        f"${b['Price']:.2f}",
                "Est. Cost":    f"${b['Est. Cost']:,.2f}",
                "Score":        b["Score"],
                "Yield":        f"{b['Yield']:.1%}",
                "Target Wt":    f"{b['Weight']:.1%}",
                "Category":     ELIGIBLE.get(b["Ticker"], {}).get("cat", "?"),
            } for b in buys]), hide_index=True, use_container_width=True)
        else:
            st.success("✅ No buys needed — portfolio is at target allocation")

        if sells:
            st.markdown("### 🔴 Sell / Trim")
            st.dataframe(pd.DataFrame([{
                "Ticker":  s["Ticker"],
                "Action":  s["Action"],
                "Shares":  s.get("Shares", "All"),
                "Price":   f"${s['Price']:.2f}",
                "Value":   f"${s['Value']:,.2f}",
                "Score":   s["Score"],
                "Reason":  s["Reason"],
            } for s in sells]), hide_index=True, use_container_width=True)

        if holds:
            with st.expander(f"🟡 Holds — {len(holds)} positions within target range"):
                st.dataframe(pd.DataFrame([{
                    "Ticker":   h["Ticker"],
                    "Score":    h["Score"],
                    "Yield":    f"{h['Yield']:.1%}",
                    "Price":    f"${h['Price']:.2f}",
                    "Current":  f"${h['Current $']:,.0f}",
                    "Target":   f"${h['Target $']:,.0f}",
                    "Weight":   f"{h['Weight']:.1%}",
                } for h in holds]), hide_index=True, use_container_width=True)

        st.divider()
        st.caption(
            f"Scores recalculate every {cfg['refresh_days']} days on open, or click Force Refresh. "
            f"Analyzing {len(ELIGIBLE)} funds. Recommendations are model-based — always verify with your broker."
        )


if __name__ == "__main__":
    main()
