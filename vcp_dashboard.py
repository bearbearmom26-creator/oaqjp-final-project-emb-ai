"""
Swing Trade S&P 500 VCP Dashboard — Mark Minervini Methodology
Volatility Contraction Pattern screener + interactive chart analysis

Run:  python vcp_dashboard.py
Then: open http://localhost:8050
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# ── Try live data first, fall back to synthetic demo data ───────────────────
try:
    import yfinance as yf
    _LIVE_DATA = True
except ImportError:
    _LIVE_DATA = False

# ── S&P 500 representative universe ─────────────────────────────────────────
SP500_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","JPM","UNH","XOM",
    "V","LLY","JNJ","AVGO","MA","PG","HD","MRK","COST","ABBV",
    "CVX","CRM","BAC","NFLX","AMD","PEP","KO","TMO","ORCL","WMT",
    "ACN","LIN","ADBE","CSCO","MCD","NKE","DIS","ABT","DHR","TXN",
    "CMCSA","VZ","NEE","AMGN","PM","RTX","HON","IBM","QCOM","INTU",
    "CAT","SPGI","UPS","GE","ELV","AMAT","SYK","ISRG","BKNG","NOW",
    "REGN","KLAC","LRCX","ADI","MRVL","PANW","CRWD","SNPS","CDNS","FTNT",
]

# ── Synthetic data generator (demo / offline mode) ──────────────────────────

_RNG = np.random.default_rng(42)


def _generate_ticker_df(ticker: str, n_days: int = 504) -> pd.DataFrame:
    """
    Create a realistic OHLCV DataFrame for demo/offline mode.

    Architecture:
    • Phases 0-3: strong Stage-2 uptrend (trend ≈ 0.0008/day) → MAs stack properly
    • Phase 4: VCP zone (last ~75 bars) with 3 explicit, shrinking triangular swings
    • Volume dry-up is monotonically enforced in the VCP zone
    """
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(ticker))
    rng  = np.random.default_rng(seed)

    base_price = float(rng.uniform(40, 600))
    trend      = float(rng.uniform(0.0006, 0.0012))   # strong Stage-2
    base_vol   = int(rng.integers(2_000_000, 40_000_000))
    vcp_bars   = int(rng.integers(60, 80))             # VCP always in detection window
    vcp_start  = n_days - vcp_bars
    n_contract = int(rng.integers(3, 5))               # 3-4 contractions

    # ── Uptrend phase ─────────────────────────────────────────────────────
    close = np.zeros(n_days)
    close[0] = base_price
    for i in range(1, vcp_start):
        noise = rng.normal(0, 0.010)
        close[i] = close[i - 1] * (1 + trend + noise)

    # ── VCP zone: explicit triangular contractions ──────────────────────
    swing_len  = vcp_bars // (n_contract + 1)
    pivot      = close[vcp_start - 1]
    first_amp  = pivot * 0.075
    pos        = vcp_start

    for c in range(n_contract):
        amp = first_amp * (0.52 ** c)      # each ~48% smaller than prior
        seg = min(swing_len, n_days - pos)
        half = seg // 2
        for j in range(half):
            close[pos + j] = pivot + amp * (j / max(1, half)) + rng.normal(0, pivot * 0.0015)
        for j in range(half, seg):
            t = (j - half) / max(1, seg - half)
            close[pos + j] = pivot + amp * (1 - t) + rng.normal(0, pivot * 0.0015)
        pos += seg

    # Tight box for remaining bars (pivot ± 1%)
    for i in range(pos, n_days):
        close[i] = close[i - 1] * (1 + rng.normal(0, 0.004))

    # ── OHLC spread ───────────────────────────────────────────────────────
    h_noise = np.abs(rng.normal(0, 0.006, n_days))
    l_noise = np.abs(rng.normal(0, 0.006, n_days))
    high    = close * (1 + h_noise)
    low     = close * (1 - l_noise)
    open_   = np.roll(close, 1)
    open_[0] = close[0]

    # ── Volume: random in uptrend, monotonically shrinking in VCP ────────
    vol = np.zeros(n_days, dtype=np.int64)
    for i in range(vcp_start):
        vol[i] = max(1_000, int(base_vol * rng.lognormal(0, 0.35)))
    # Enforce strict per-swing average shrinkage
    swing_vols = []
    for c in range(n_contract):
        frac = 1.0 - c * 0.22           # 100%, 78%, 56%, 34% …
        swing_vols.append(frac)
    # Remaining tail
    swing_vols.append(swing_vols[-1] * 0.6)
    sv_idx = 0
    current_swing_start = vcp_start
    for c, frac in enumerate(swing_vols):
        seg_end = vcp_start + (c + 1) * swing_len if c < n_contract else n_days
        seg_end = min(seg_end, n_days)
        for i in range(current_swing_start, seg_end):
            noise = rng.lognormal(0, 0.18)
            vol[i] = max(1_000, int(base_vol * frac * noise))
        current_swing_start = seg_end

    dates = pd.bdate_range(end=datetime.today(), periods=n_days)
    return pd.DataFrame({
        "Open":   open_,
        "High":   high,
        "Low":    low,
        "Close":  close,
        "Volume": vol,
    }, index=dates)


def _fetch_df(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Return OHLCV DataFrame: live from yfinance if available, else synthetic.
    period: '3mo','6mo','1y','2y'
    """
    period_days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}
    n_days = period_days.get(period, 504)

    if _LIVE_DATA:
        try:
            df = yf.download(ticker, period="2y", interval="1d",
                             progress=False, auto_adjust=True)
            if not df.empty and len(df) >= 200:
                return df
        except Exception:
            pass
    return _generate_ticker_df(ticker, n_days=504)


# ── Technical Indicators ─────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add MAs, ATR-14, volume MA-20, and Bollinger Bands."""

    def _squeeze(s):
        return s.squeeze() if hasattr(s, "squeeze") else s

    close = _squeeze(df["Close"])
    high  = _squeeze(df["High"])
    low   = _squeeze(df["Low"])
    vol   = _squeeze(df["Volume"])

    df = df.copy()
    df["MA50"]  = close.rolling(50).mean()
    df["MA150"] = close.rolling(150).mean()
    df["MA200"] = close.rolling(200).mean()

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR14"]   = tr.rolling(14).mean()
    df["VolMA20"] = vol.rolling(20).mean()

    ma20  = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_upper"] = ma20 + 2 * std20
    df["BB_lower"] = ma20 - 2 * std20
    df["BB_width"]  = (df["BB_upper"] - df["BB_lower"]) / ma20

    return df


# ── Minervini 8-Point Trend Template ────────────────────────────────────────

def minervini_trend_score(df: pd.DataFrame) -> dict:
    """
    Return score dict.  Valid Stage 2 requires score >= 6.
    """
    if len(df) < 200:
        return {"score": 0, "criteria": {}, "valid": False, "close": 0,
                "ma50": 0, "ma150": 0, "ma200": 0,
                "high52": 0, "low52": 0, "pct_off_high": 0}

    def _f(x):
        v = x.iloc[-1]
        return float(v.squeeze()) if hasattr(v, "squeeze") else float(v)

    close = _f(df["Close"])
    ma50  = _f(df["MA50"])
    ma150 = _f(df["MA150"])
    ma200 = _f(df["MA200"])

    high_col = df["High"].squeeze() if hasattr(df["High"], "squeeze") else df["High"]
    low_col  = df["Low"].squeeze()  if hasattr(df["Low"],  "squeeze") else df["Low"]
    high52 = float(high_col.tail(252).max())
    low52  = float(low_col.tail(252).min())

    ma200_1m = float(df["MA200"].iloc[-21])
    ma200_trend = ma200 > ma200_1m

    criteria = {
        "Price > MA150 & MA200":    close > ma150 and close > ma200,
        "MA150 > MA200":            ma150 > ma200,
        "MA200 trending up":        ma200_trend,
        "MA50 > MA150 & MA200":     ma50 > ma150 and ma50 > ma200,
        "Price > MA50":             close > ma50,
        "Price ≥ 30% above 52w L":  close >= low52 * 1.30,
        "Price within 25% of 52w H": close >= high52 * 0.75,
        "52w H/L ratio ≥ 1.3":     (high52 / low52 >= 1.30) if low52 > 0 else False,
    }

    score = sum(criteria.values())
    return {
        "score":       score,
        "criteria":    criteria,
        "valid":       score >= 6,
        "close":       close,
        "ma50":        ma50,
        "ma150":       ma150,
        "ma200":       ma200,
        "high52":      high52,
        "low52":       low52,
        "pct_off_high": round((close / high52 - 1) * 100, 1),
    }


# ── VCP Detection ────────────────────────────────────────────────────────────

def detect_vcp(df: pd.DataFrame, min_contractions: int = 2) -> dict:
    """
    Detect Volatility Contraction Pattern in the most recent 90 bars.
    Returns vcp_score (0-100), list of contractions, pivot price.
    """
    empty = {"vcp_score": 0, "contractions": [], "pivot": None, "tightness": 0,
             "shrinking_amp": False, "shrinking_vol": False}

    if len(df) < 60:
        return empty

    window = df.tail(90).copy()

    def _s(col):
        v = window[col]
        return v.squeeze() if hasattr(v, "squeeze") else v

    close  = _s("Close")
    high   = _s("High")
    low    = _s("Low")
    vol    = _s("Volume")
    n      = len(window)

    # Identify pivot highs and lows (8-bar look-around to filter micro-pivots)
    LK = 8
    pivots = []
    for i in range(LK, n - LK):
        h_slice = high.iloc[i - LK: i + LK + 1]
        l_slice = low.iloc[i - LK: i + LK + 1]
        if float(high.iloc[i]) == float(h_slice.max()):
            pivots.append(("H", i, float(high.iloc[i]), window.index[i]))
        elif float(low.iloc[i]) == float(l_slice.min()):
            pivots.append(("L", i, float(low.iloc[i]), window.index[i]))

    # Keep strictly alternating H/L, prefer the more extreme value
    clean = []
    for p in pivots:
        if not clean or clean[-1][0] != p[0]:
            clean.append(list(p))
        else:
            if p[0] == "H" and p[2] > clean[-1][2]:
                clean[-1] = list(p)
            elif p[0] == "L" and p[2] < clean[-1][2]:
                clean[-1] = list(p)

    if len(clean) < 4:
        return empty

    contractions = []
    for i in range(len(clean) - 1):
        a, b = clean[i], clean[i + 1]
        swing_high = max(a[2], b[2])
        swing_low  = min(a[2], b[2])
        amp_pct    = (swing_high - swing_low) / swing_low * 100
        bar_s, bar_e = min(a[1], b[1]), max(a[1], b[1])
        avg_vol = float(vol.iloc[bar_s: bar_e + 1].mean()) if bar_e > bar_s else 0
        contractions.append({
            "from_date":  str(a[3])[:10],
            "to_date":    str(b[3])[:10],
            "amplitude":  round(amp_pct, 2),
            "avg_vol":    int(avg_vol),
            "swing_high": swing_high,
            "swing_low":  swing_low,
        })

    # Drop micro-swings (< 1.2%) that are just noise
    contractions = [c for c in contractions if c["amplitude"] >= 1.2]

    if len(contractions) < min_contractions:
        return {**empty, "contractions": contractions}

    amplitudes = [c["amplitude"] for c in contractions]
    volumes    = [c["avg_vol"]   for c in contractions]

    # Allow one non-shrinking pair (real markets aren't perfectly monotone)
    n_pairs = len(amplitudes) - 1
    amp_pairs_ok = sum(amplitudes[i] > amplitudes[i + 1] for i in range(n_pairs))
    vol_pairs_ok = sum(volumes[i]    >= volumes[i + 1]   for i in range(n_pairs))
    shrinking_amp = amp_pairs_ok >= max(1, n_pairs - 1) and amplitudes[0] > amplitudes[-1]
    shrinking_vol = vol_pairs_ok >= max(1, n_pairs - 1)

    first_amp, last_amp = amplitudes[0], amplitudes[-1]
    contraction_ratio = (first_amp - last_amp) / first_amp if first_amp > 0 else 0
    tightness = max(0.0, 100 - last_amp * 10)

    vcp_score = 0
    if shrinking_amp:                       vcp_score += 40
    if shrinking_vol:                       vcp_score += 20
    vcp_score += min(20, len(contractions) * 7)
    vcp_score += min(20, int(contraction_ratio * 20))

    pivot = None
    for p in reversed(clean):
        if p[0] == "H":
            pivot = {"price": p[2], "date": str(p[3])[:10]}
            break

    return {
        "vcp_score":    min(100, vcp_score),
        "contractions": contractions,
        "pivot":        pivot,
        "tightness":    round(tightness, 1),
        "shrinking_amp": shrinking_amp,
        "shrinking_vol": shrinking_vol,
    }


# ── Screener ─────────────────────────────────────────────────────────────────

def screen_ticker(ticker: str) -> dict | None:
    """Screen a single ticker. Returns dict or None if it fails the template."""
    try:
        df = _fetch_df(ticker, "2y")
        if df.empty or len(df) < 200:
            return None
        df = compute_indicators(df)
        trend = minervini_trend_score(df)
        if not trend["valid"]:
            return None
        vcp = detect_vcp(df)

        def _s(col):
            v = df[col]
            return v.squeeze() if hasattr(v, "squeeze") else v

        vol_series = _s("Volume")
        vol_ma     = _s("VolMA20")
        last_vol   = float(vol_series.iloc[-1])
        last_volma = float(vol_ma.iloc[-1])
        bb_width   = float((_s("BB_width")).iloc[-1])

        return {
            "ticker":       ticker,
            "price":        round(trend["close"], 2),
            "trend_score":  trend["score"],
            "vcp_score":    vcp["vcp_score"],
            "tightness":    vcp["tightness"],
            "pct_off_high": trend["pct_off_high"],
            "pivot":        vcp["pivot"]["price"] if vcp["pivot"] else None,
            "shrink_amp":   vcp.get("shrinking_amp", False),
            "shrink_vol":   vcp.get("shrinking_vol", False),
            "n_contract":   len(vcp["contractions"]),
            "vol_ratio":    round(last_vol / last_volma, 2) if last_volma > 0 else 1.0,
            "bb_width":     round(bb_width * 100, 2),
            "ma50":         round(trend["ma50"], 2),
            "ma150":        round(trend["ma150"], 2),
            "ma200":        round(trend["ma200"], 2),
            "high52":       round(trend["high52"], 2),
            "low52":        round(trend["low52"], 2),
        }
    except Exception:
        return None


# ── Dashboard colours ────────────────────────────────────────────────────────

DARK   = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
ACCENT = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
YELLOW = "#e3b341"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"

BADGE = {
    "display": "inline-block", "padding": "2px 10px",
    "borderRadius": "12px", "fontSize": "12px",
    "fontWeight": "600", "marginLeft": "8px",
}

# ── Layout helper ────────────────────────────────────────────────────────────

def _filter_block(label: str, control, width: str = "200px"):
    return html.Div(
        [html.Label(label, style={"fontSize": "11px", "color": MUTED,
                                  "display": "block", "marginBottom": "4px"}), control],
        style={"width": width},
    )


# ── App Layout ───────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="VCP Swing Trade Dashboard — Minervini",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = html.Div(
    style={"backgroundColor": DARK, "minHeight": "100vh",
           "fontFamily": "system-ui, -apple-system, sans-serif", "color": TEXT},
    children=[

        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "background": f"linear-gradient(135deg, {PANEL} 0%, #1c2128 100%)",
                "borderBottom": f"1px solid {BORDER}",
                "padding": "14px 24px",
                "display": "flex", "alignItems": "center", "justifyContent": "space-between",
            },
            children=[
                html.Div([
                    html.H1("VCP Swing Trade Dashboard",
                            style={"margin": 0, "fontSize": "20px", "fontWeight": 700, "color": ACCENT}),
                    html.Span("Mark Minervini · Volatility Contraction Pattern · S&P 500",
                              style={"fontSize": "11px", "color": MUTED}),
                ]),
                html.Div([
                    html.Div(
                        "DEMO MODE — synthetic data (yfinance blocked in this env)" if not _LIVE_DATA else "LIVE DATA via yfinance",
                        style={"fontSize": "11px", "color": YELLOW if not _LIVE_DATA else GREEN,
                               "marginBottom": "6px", "textAlign": "right"},
                    ),
                    html.Button("Run Screen", id="run-screen-btn", n_clicks=0,
                        style={
                            "backgroundColor": ACCENT, "color": "#0d1117", "border": "none",
                            "borderRadius": "6px", "padding": "8px 20px",
                            "fontWeight": 700, "cursor": "pointer", "fontSize": "14px",
                        }),
                    html.Div(id="screen-status",
                             style={"color": MUTED, "fontSize": "11px", "marginTop": "4px", "textAlign": "right"}),
                ]),
            ],
        ),

        # ── Filter Bar ──────────────────────────────────────────────────────
        html.Div(
            style={"padding": "10px 24px", "display": "flex", "gap": "24px",
                   "flexWrap": "wrap", "borderBottom": f"1px solid {BORDER}",
                   "backgroundColor": "#10161e"},
            children=[
                _filter_block("Min Trend Score (0-8)", dcc.Slider(
                    0, 8, 1, value=6, id="filter-trend",
                    marks={i: str(i) for i in range(9)},
                    tooltip={"placement": "bottom"}, updatemode="drag",
                ), width="200px"),
                _filter_block("Min VCP Score (0-100)", dcc.Slider(
                    0, 100, 5, value=40, id="filter-vcp",
                    marks={i: str(i) for i in range(0, 101, 20)},
                    tooltip={"placement": "bottom"}, updatemode="drag",
                ), width="220px"),
                _filter_block("Max % Off 52w High", dcc.Slider(
                    5, 40, 5, value=25, id="filter-pct-high",
                    marks={i: f"{i}%" for i in range(5, 41, 5)},
                    tooltip={"placement": "bottom"}, updatemode="drag",
                ), width="200px"),
                html.Div([
                    dcc.Checklist(
                        ["Require shrinking volume"],
                        id="filter-vol",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"fontSize": "13px"},
                    ),
                ], style={"alignSelf": "center", "paddingTop": "18px"}),
            ],
        ),

        # ── Body ────────────────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "height": "calc(100vh - 175px)"},
            children=[

                # Left panel: screener table + criteria
                html.Div(
                    style={"width": "460px", "minWidth": "380px",
                           "borderRight": f"1px solid {BORDER}",
                           "overflowY": "auto", "padding": "12px"},
                    children=[
                        html.Div(id="results-count",
                                 style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px"}),
                        dash_table.DataTable(
                            id="screener-table",
                            columns=[
                                {"name": "Ticker",     "id": "ticker"},
                                {"name": "Price",      "id": "price"},
                                {"name": "Trend /8",   "id": "trend_score"},
                                {"name": "VCP",        "id": "vcp_score"},
                                {"name": "% Off Hi",   "id": "pct_off_high"},
                                {"name": "Tight%",     "id": "tightness"},
                                {"name": "#VCP",       "id": "n_contract"},
                                {"name": "Vol Ratio",  "id": "vol_ratio"},
                            ],
                            data=[],
                            row_selectable="single",
                            selected_rows=[],
                            sort_action="native",
                            sort_by=[{"column_id": "vcp_score", "direction": "desc"}],
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "backgroundColor": PANEL, "color": TEXT,
                                "border": f"1px solid {BORDER}",
                                "fontFamily": "monospace", "fontSize": "13px",
                                "padding": "5px 8px", "textAlign": "right",
                            },
                            style_cell_conditional=[
                                {"if": {"column_id": "ticker"},
                                 "textAlign": "left", "fontWeight": "700", "color": ACCENT},
                            ],
                            style_header={
                                "backgroundColor": "#1c2128", "color": MUTED,
                                "fontWeight": "600", "fontSize": "11px",
                                "border": f"1px solid {BORDER}",
                            },
                            style_data_conditional=[
                                {"if": {"filter_query": "{vcp_score} >= 80"},
                                 "backgroundColor": "#0d2818", "color": GREEN},
                                {"if": {"filter_query": "{vcp_score} >= 60 && {vcp_score} < 80"},
                                 "backgroundColor": "#1a1400", "color": YELLOW},
                                {"if": {"state": "selected"},
                                 "backgroundColor": "#1c2d44",
                                 "border": f"1px solid {ACCENT}"},
                            ],
                        ),
                        html.Div(id="criteria-panel", style={"marginTop": "14px"}),
                    ],
                ),

                # Right panel: chart + stats
                html.Div(
                    style={"flex": 1, "padding": "12px", "overflowY": "auto"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "marginBottom": "8px", "gap": "10px"},
                            children=[
                                dcc.Input(
                                    id="manual-ticker", type="text",
                                    placeholder="Enter ticker…", debounce=True,
                                    style={
                                        "backgroundColor": PANEL, "color": TEXT,
                                        "border": f"1px solid {BORDER}", "borderRadius": "6px",
                                        "padding": "6px 12px", "fontSize": "14px", "width": "130px",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="period-dropdown",
                                    options=[
                                        {"label": "3 Months", "value": "3mo"},
                                        {"label": "6 Months", "value": "6mo"},
                                        {"label": "1 Year",   "value": "1y"},
                                        {"label": "2 Years",  "value": "2y"},
                                    ],
                                    value="1y", clearable=False,
                                    style={"width": "130px", "fontSize": "13px"},
                                ),
                                html.Div(id="chart-ticker-label",
                                         style={"fontWeight": 700, "fontSize": "16px", "color": ACCENT}),
                            ],
                        ),
                        dcc.Loading(
                            dcc.Graph(id="price-chart",
                                      config={"displayModeBar": True, "scrollZoom": True},
                                      style={"height": "500px"}),
                            color=ACCENT,
                        ),
                        html.Div(id="stats-row",   style={"marginTop": "8px"}),
                        html.Div(id="vcp-detail",  style={"marginTop": "8px"}),
                    ],
                ),
            ],
        ),

        # Stores
        dcc.Store(id="screener-store"),
        dcc.Store(id="selected-ticker-store"),
    ],
)


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("screener-store", "data"),
    Output("screen-status", "children"),
    Input("run-screen-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_screener(_):
    results = [r for t in SP500_TICKERS if (r := screen_ticker(t))]
    return results, f"Screened {len(SP500_TICKERS)} tickers → {len(results)} passed Minervini template"


@app.callback(
    Output("screener-table", "data"),
    Output("results-count", "children"),
    Input("screener-store", "data"),
    Input("filter-trend", "value"),
    Input("filter-vcp", "value"),
    Input("filter-pct-high", "value"),
    Input("filter-vol", "value"),
)
def filter_table(data, min_trend, min_vcp, max_pct_high, req_vol):
    if not data:
        return [], "Click 'Run Screen' to load results."
    df = pd.DataFrame(data)
    df = df[df["trend_score"] >= (min_trend or 0)]
    df = df[df["vcp_score"]   >= (min_vcp or 0)]
    df = df[df["pct_off_high"] >= -abs(max_pct_high or 25)]
    if req_vol and "Require shrinking volume" in req_vol:
        df = df[df["shrink_vol"] == True]
    df = df.sort_values("vcp_score", ascending=False)
    return df.to_dict("records"), f"{len(df)} stocks match filters"


@app.callback(
    Output("selected-ticker-store", "data"),
    Input("screener-table", "selected_rows"),
    Input("screener-table", "data"),
    Input("manual-ticker", "value"),
)
def set_ticker(selected_rows, table_data, manual):
    ctx = callback_context.triggered_id
    if ctx == "manual-ticker" and manual:
        return manual.strip().upper()
    if selected_rows and table_data:
        return table_data[selected_rows[0]]["ticker"]
    return None


@app.callback(
    Output("price-chart",        "figure"),
    Output("chart-ticker-label", "children"),
    Output("stats-row",          "children"),
    Output("criteria-panel",     "children"),
    Output("vcp-detail",         "children"),
    Input("selected-ticker-store", "data"),
    Input("period-dropdown",       "value"),
)
def update_chart(ticker, period):
    empty_fig = go.Figure(layout=go.Layout(
        paper_bgcolor=DARK, plot_bgcolor=PANEL, font=dict(color=TEXT),
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        annotations=[dict(text="Select a ticker from the screen or type one above",
                          showarrow=False, font=dict(size=15, color=MUTED),
                          xref="paper", yref="paper", x=0.5, y=0.5)],
        margin=dict(l=0, r=0, t=10, b=0),
    ))
    if not ticker:
        return empty_fig, "", [], [], []

    try:
        df_raw = _fetch_df(ticker, period)
    except Exception as e:
        err_fig = go.Figure(layout=go.Layout(
            paper_bgcolor=DARK, plot_bgcolor=PANEL,
            annotations=[dict(text=f"Error: {e}", showarrow=False,
                              font=dict(size=14, color=RED),
                              xref="paper", yref="paper", x=0.5, y=0.5)],
        ))
        return err_fig, ticker, [], [], []

    df = compute_indicators(df_raw)
    trend = minervini_trend_score(df)
    vcp   = detect_vcp(df)

    period_days = {"3mo": 63, "6mo": 126, "1y": 252, "2y": 504}
    df_plot = df.tail(period_days.get(period, 252))

    def _s(col):
        v = df_plot[col]
        return v.squeeze() if hasattr(v, "squeeze") else v

    close  = _s("Close")
    open_  = _s("Open")
    high   = _s("High")
    low    = _s("Low")
    vol    = _s("Volume")
    dates  = df_plot.index

    # ── Figure ──────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20], vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=dates, open=open_, high=high, low=low, close=close,
        name="Price",
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor=GREEN, decreasing_fillcolor=RED,
    ), row=1, col=1)

    for col, color, name in [
        ("MA50", "#f0a500", "MA50"),
        ("MA150", "#a78bfa", "MA150"),
        ("MA200", "#f85149", "MA200"),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=_s(col), name=name,
            line=dict(color=color, width=1.5), opacity=0.9,
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=_s("BB_upper"), name="BB Upper",
        line=dict(color=MUTED, width=1, dash="dot"), opacity=0.4,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=_s("BB_lower"), name="BB Lower",
        line=dict(color=MUTED, width=1, dash="dot"), opacity=0.4,
        fill="tonexty", fillcolor="rgba(139,148,158,0.05)",
    ), row=1, col=1)

    if vcp["pivot"]:
        fig.add_hline(
            y=vcp["pivot"]["price"],
            line_color=GREEN, line_width=2, line_dash="dash",
            annotation_text=f"  Pivot {vcp['pivot']['price']:.2f}",
            annotation_font_color=GREEN,
            annotation_position="top right",
            row=1, col=1,
        )

    # VCP shading
    for c in vcp["contractions"]:
        fig.add_vrect(
            x0=c["from_date"], x1=c["to_date"],
            fillcolor=ACCENT, opacity=0.06,
            layer="below", line_width=0,
        )

    bar_colors = [GREEN if float(close.iloc[i]) >= float(open_.iloc[i]) else RED
                  for i in range(len(df_plot))]
    fig.add_trace(go.Bar(
        x=dates, y=vol, name="Volume",
        marker_color=bar_colors, opacity=0.7,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=_s("VolMA20"), name="VolMA20",
        line=dict(color=YELLOW, width=1.5),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=_s("ATR14"), name="ATR-14",
        line=dict(color="#a78bfa", width=1.5),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.10)",
    ), row=3, col=1)

    fig.update_layout(
        paper_bgcolor=DARK, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="system-ui"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10), traceorder="normal"),
        margin=dict(l=0, r=10, t=10, b=0),
        hovermode="x unified",
        yaxis=dict(gridcolor=BORDER, gridwidth=0.5),
        yaxis2=dict(gridcolor=BORDER, gridwidth=0.5),
        yaxis3=dict(gridcolor=BORDER, gridwidth=0.5, title="ATR"),
    )
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.5)

    # ── Stats cards ─────────────────────────────────────────────────────────
    last_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close
    day_chg    = (last_close / prev_close - 1) * 100

    def card(label, value, color=TEXT):
        return html.Div(
            style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
                   "borderRadius": "6px", "padding": "7px 12px",
                   "textAlign": "center", "minWidth": "85px"},
            children=[
                html.Div(label, style={"fontSize": "10px", "color": MUTED, "marginBottom": "2px"}),
                html.Div(value, style={"fontSize": "14px", "fontWeight": 700,
                                       "color": color, "fontFamily": "monospace"}),
            ],
        )

    stats = html.Div(
        style={"display": "flex", "gap": "6px", "flexWrap": "wrap"},
        children=[
            card("Price",       f"${last_close:.2f}"),
            card("Day",         f"{day_chg:+.2f}%", GREEN if day_chg >= 0 else RED),
            card("Trend",       f"{trend['score']}/8", GREEN if trend["score"] >= 6 else YELLOW),
            card("VCP Score",   f"{vcp['vcp_score']}/100",
                 GREEN if vcp["vcp_score"] >= 60 else YELLOW),
            card("Tightness",   f"{vcp['tightness']}%"),
            card("Contractions",str(len(vcp["contractions"]))),
            card("52w High",    f"${trend.get('high52', 0):.2f}"),
            card("% Off High",  f"{trend.get('pct_off_high', 0):.1f}%",
                 YELLOW if trend.get("pct_off_high", 0) > -15 else RED),
            card("Pivot",       f"${vcp['pivot']['price']:.2f}" if vcp["pivot"] else "—", GREEN),
        ],
    )

    # ── Criteria panel ───────────────────────────────────────────────────────
    def crit(label, passed):
        return html.Div(
            style={"display": "flex", "alignItems": "center", "padding": "2px 0", "fontSize": "12px"},
            children=[
                html.Span("✓" if passed else "✗",
                          style={"color": GREEN if passed else RED,
                                 "marginRight": "8px", "fontWeight": 700, "width": "14px"}),
                html.Span(label, style={"color": TEXT if passed else MUTED}),
            ],
        )

    criteria_panel = html.Div(
        style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
               "borderRadius": "6px", "padding": "10px 14px"},
        children=[
            html.Div("Minervini Trend Template",
                     style={"fontSize": "12px", "fontWeight": 700,
                            "color": ACCENT, "marginBottom": "6px"}),
            *[crit(lbl, ok) for lbl, ok in trend.get("criteria", {}).items()],
        ],
    )

    # ── VCP contraction table ────────────────────────────────────────────────
    td_s = {"padding": "4px 10px", "fontSize": "12px", "borderBottom": f"1px solid {BORDER}"}
    th_s = {**td_s, "color": MUTED, "fontWeight": 600, "fontSize": "11px"}
    rows = [
        html.Tr([
            html.Td(f"#{i}", style={"color": MUTED, **td_s}),
            html.Td(c["from_date"], style=td_s),
            html.Td(c["to_date"],   style=td_s),
            html.Td(f"{c['amplitude']:.1f}%", style={"color": YELLOW, **td_s}),
            html.Td(f"{c['avg_vol']:,}", style={"color": MUTED, **td_s}),
        ])
        for i, c in enumerate(vcp["contractions"], 1)
    ]

    vcp_detail = html.Div(
        style={"backgroundColor": PANEL, "border": f"1px solid {BORDER}",
               "borderRadius": "6px", "padding": "10px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "16px",
                       "marginBottom": "8px", "fontSize": "12px"},
                children=[
                    html.Span(["Amplitude Shrinking: ",
                               html.Strong("YES" if vcp.get("shrinking_amp") else "NO",
                                           style={"color": GREEN if vcp.get("shrinking_amp") else RED})]),
                    html.Span(["Volume Shrinking: ",
                               html.Strong("YES" if vcp.get("shrinking_vol") else "NO",
                                           style={"color": GREEN if vcp.get("shrinking_vol") else RED})]),
                    html.Span(["VCP Score: ",
                               html.Strong(f"{vcp['vcp_score']}/100", style={"color": ACCENT})]),
                ],
            ),
            html.Table(
                [
                    html.Thead(html.Tr([
                        html.Th("#", style=th_s), html.Th("From", style=th_s),
                        html.Th("To", style=th_s), html.Th("Amplitude", style=th_s),
                        html.Th("Avg Volume", style=th_s),
                    ])),
                    html.Tbody(rows),
                ],
                style={"width": "100%", "borderCollapse": "collapse"},
            ) if rows else html.Div("No contractions detected.",
                                     style={"color": MUTED, "fontSize": "12px"}),
        ],
    )

    # ── Ticker label badges ──────────────────────────────────────────────────
    label = html.Span([
        html.Strong(ticker, style={"color": ACCENT}),
        html.Span(
            " Stage 2" if trend["valid"] else " Not Stage 2",
            style={**BADGE,
                   "backgroundColor": "#0d2818" if trend["valid"] else "#2d0f0f",
                   "color": GREEN if trend["valid"] else RED},
        ),
        html.Span(
            f" VCP {vcp['vcp_score']}",
            style={**BADGE,
                   "backgroundColor": "#0d2818" if vcp["vcp_score"] >= 60 else "#2d1a00",
                   "color": GREEN if vcp["vcp_score"] >= 60 else YELLOW},
        ),
    ])

    return fig, label, stats, criteria_panel, vcp_detail


if __name__ == "__main__":
    print("Starting VCP Dashboard → http://127.0.0.1:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
