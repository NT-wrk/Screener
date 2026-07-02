# -*- coding: utf-8 -*-
"""
Swing-Trade Marktscreener
Sektor-Übersicht (Grobdurchlauf über Kern-Ticker) -> Drill-Down mit
erweiterter Ticker-Liste und Detail-Kennzahlen.

Start:  streamlit run app.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

SECTORS_FILE = Path(__file__).parent / "sectors.json"

st.set_page_config(page_title="Swing-Trade Screener", page_icon="📈", layout="wide")


# ----------------------------------------------------------------------------
# Sektor-Konfiguration laden / speichern
# ----------------------------------------------------------------------------
def load_sectors() -> dict:
    with open(SECTORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sectors(sectors: dict) -> None:
    with open(SECTORS_FILE, "w", encoding="utf-8") as f:
        json.dump(sectors, f, indent=2, ensure_ascii=False)


# ----------------------------------------------------------------------------
# Kursdaten laden (15 min Cache)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_history(tickers: tuple, period: str = "1y") -> dict:
    """Lädt Tagesdaten für alle Ticker. Gibt {ticker: DataFrame} zurück."""
    raw = yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,          # Splits/Dividenden rausrechnen -> saubere Overnight-Statistik
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result = {}
    if raw is None or raw.empty:
        return result
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[t].copy()
            else:  # nur ein Ticker angefragt
                df = raw.copy()
            df = df.dropna(subset=["Close", "Open"])
            if len(df) >= 40:  # genug Historie für 30-Tage-Statistik
                result[t] = df
        except (KeyError, TypeError):
            continue
    return result


# ----------------------------------------------------------------------------
# Währungen / Euro-Umrechnung
# ----------------------------------------------------------------------------
CURRENCY_BY_SUFFIX = {
    ".DE": "EUR", ".F": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR",
    ".ST": "SEK", ".T": "JPY", ".TO": "CAD", ".AX": "AUD", ".L": "GBP",
    ".SW": "CHF",
}


def ticker_currency(t: str) -> str:
    for suffix, cur in CURRENCY_BY_SUFFIX.items():
        if t.endswith(suffix):
            return cur
    return "USD"


@st.cache_data(ttl=900, show_spinner=False)
def load_fx_rates() -> dict:
    """Wechselkurse: Einheiten Fremdwährung pro 1 EUR (z.B. EURUSD=X ~ 1.08)."""
    pairs = {"USD": "EURUSD=X", "SEK": "EURSEK=X", "JPY": "EURJPY=X",
             "CAD": "EURCAD=X", "AUD": "EURAUD=X", "GBP": "EURGBP=X",
             "CHF": "EURCHF=X"}
    rates = {"EUR": 1.0}
    try:
        data = yf.download(list(pairs.values()), period="5d", interval="1d",
                           auto_adjust=True, group_by="ticker",
                           threads=True, progress=False)
        for cur, sym in pairs.items():
            try:
                close = data[sym]["Close"].dropna()
                if len(close):
                    rates[cur] = float(close.iloc[-1])
            except (KeyError, TypeError):
                continue
    except Exception:
        pass
    return rates


def eur_columns(t: str, row: dict, rates: dict) -> dict:
    """Ergänzt Währung sowie Kurs/Stop/Ziel in Euro."""
    cur = ticker_currency(t)
    rate = rates.get(cur)

    def conv(x):
        if rate is None or x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return x / rate

    return {"Währung": cur, "Kurs €": conv(row["Kurs"]),
            "Stop €": conv(row["Stop"]), "Ziel €": conv(row["Ziel (2R)"])}


# ----------------------------------------------------------------------------
# Kennzahlen & Momentum-Score
# ----------------------------------------------------------------------------
def clamp(x: float, scale: float) -> float:
    """Normiert x auf [-1, 1], wobei |x| >= scale voll zählt."""
    if x is None or math.isnan(x):
        return 0.0
    return max(-1.0, min(1.0, x / scale))


def compute_metrics(df: pd.DataFrame) -> dict:
    c, o, v = df["Close"], df["Open"], df["Volume"]
    last = float(c.iloc[-1])

    # --- Prio 1: Durchschnittspreise 10 / 30 Tage (Open & Close) ---
    avg_o10, avg_c10 = float(o.tail(10).mean()), float(c.tail(10).mean())
    avg_o30, avg_c30 = float(o.tail(30).mean()), float(c.tail(30).mean())
    trend_10_30 = avg_c10 / avg_c30 - 1  # zentrales Momentum-Signal

    # --- Prio 1: durchschnittliche Bewegung pro Tag, getrennt nach Quelle ---
    intraday = c / o - 1                 # Open -> Close am selben Tag
    overnight = o / c.shift(1) - 1       # Vortages-Close -> Open
    daily = c.pct_change()               # Close -> Close gesamt

    intra10 = float(intraday.tail(10).mean())
    intra30 = float(intraday.tail(30).mean())
    on10 = float(overnight.tail(10).mean())
    on30 = float(overnight.tail(30).mean())
    day10 = float(daily.tail(10).mean())
    day30 = float(daily.tail(30).mean())

    # --- Prio 2: Moving Averages ---
    sma20 = float(c.rolling(20).mean().iloc[-1])
    sma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else float("nan")

    # --- Prio 2: MACD (12/26/9) ---
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    h_now, h_prev = float(hist.iloc[-1]), float(hist.iloc[-2])
    if h_now > 0:
        macd_state = "bullisch ↑" if h_now > h_prev else "bullisch ↓"
    else:
        macd_state = "bärisch ↑" if h_now > h_prev else "bärisch ↓"

    # --- Ergänzungen: RSI, relatives Volumen, Abstand 52W-Hoch ---
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = float((100 - 100 / (1 + rs)).iloc[-1])

    vol60 = float(v.tail(60).mean())
    relvol = float(v.tail(10).mean()) / vol60 if vol60 > 0 else float("nan")

    high52 = float(c.max())
    dist52 = last / high52 - 1

    # --- Momentum-Score: gewichtete Summe, -100 .. +100 ---
    macd_comp = (1.0 if h_now > h_prev else 0.4) * (1 if h_now > 0 else -1)
    if not math.isnan(sma50):
        if last > sma20 > sma50:
            ma_comp = 1.0
        elif last < sma20 < sma50:
            ma_comp = -1.0
        else:
            ma_comp = 0.3 if last > sma20 else -0.3
    else:
        ma_comp = 0.3 if last > sma20 else -0.3

    relvol_comp = 0.0
    if not math.isnan(relvol):
        # Volumenanstieg verstärkt die Richtung des Trends, ist allein kein Signal
        relvol_comp = clamp(relvol - 1, 0.5) * (1 if trend_10_30 >= 0 else -1)

    score = 100 * (
        0.30 * clamp(trend_10_30, 0.05)        # 10T- vs 30T-Durchschnitt (Prio 1)
        + 0.125 * clamp(intra10, 0.005)        # Ø Intraday-Drift (Prio 1)
        + 0.125 * clamp(on10, 0.005)           # Ø Overnight-Drift (Prio 1)
        + 0.10 * macd_comp                     # MACD (Prio 2)
        + 0.10 * ma_comp                       # MA-Anordnung (Prio 2)
        + 0.10 * relvol_comp                   # relatives Volumen
        + 0.15 * clamp(dist52 + 0.15, 0.15)    # Nähe zum 52W-Hoch
    )

    return {
        "Kurs": last,
        "Score": round(score, 1),
        "Ø Open 10T": avg_o10, "Ø Close 10T": avg_c10,
        "Ø Open 30T": avg_o30, "Ø Close 30T": avg_c30,
        "Trend 10/30": trend_10_30,
        "Ø Intraday 10T": intra10, "Ø Intraday 30T": intra30,
        "Ø Overnight 10T": on10, "Ø Overnight 30T": on30,
        "Ø Tag 10T": day10, "Ø Tag 30T": day30,
        "SMA20": sma20, "SMA50": sma50,
        "MACD": macd_state,
        "RSI": round(rsi, 1),
        "Rel. Volumen": relvol,
        "Abstand 52W-Hoch": dist52,
    }


def compute_setup(df: pd.DataFrame, m: dict) -> dict:
    """Regelbasierte Kauf-/Exit-Signale in Trendrichtung inkl. Stop und 2R-Ziel."""
    c, o, h, l = df["Close"], df["Open"], df["High"], df["Low"]
    last = float(c.iloc[-1])

    # ATR(14) nach Wilder
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])

    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    trend_ok = m["Trend 10/30"] > 0 and m["Score"] >= 10
    below_sma20 = last < float(sma20.iloc[-1])

    # Pullback: SMA10 in den letzten 3 Tagen berührt, heute darüber geschlossen
    touched = bool((l.tail(3) <= sma10.tail(3)).any())
    reclaimed = last > float(sma10.iloc[-1]) and last > float(o.iloc[-1])
    pullback = trend_ok and touched and reclaimed and m["RSI"] < 70

    # Breakout: Schluss über dem 20-Tage-Hoch (ohne heute) mit Volumen
    high20 = float(h.iloc[-21:-1].max())
    relvol = m["Rel. Volumen"] if not math.isnan(m["Rel. Volumen"]) else 0
    breakout = trend_ok and last > high20 and relvol > 1.2

    if m["Score"] <= -10:
        setup = "⚫ Abwärtstrend — meiden"
    elif m["Trend 10/30"] > 0 and below_sma20:
        setup = "🔴 Exit — Schluss unter SMA20"
    elif breakout:
        setup = "🟢 Kauf: Breakout"
    elif pullback:
        setup = "🟢 Kauf: Pullback"
    elif trend_ok:
        setup = "🟡 Beobachten"
    else:
        setup = "⚪ kein Setup"

    stop = ziel = risiko = float("nan")
    if setup.startswith("🟢"):
        swing_low = float(l.tail(5).min())
        stop = max(swing_low, last - 2 * atr)  # der engere der beiden Stops
        if stop >= last:
            stop = last - 1.5 * atr
        risk = last - stop
        ziel = last + 2 * risk               # Ziel bei Chance/Risiko = 2:1
        risiko = risk / last

    return {"Setup": setup, "Stop": stop, "Ziel (2R)": ziel, "Risiko": risiko,
            "ATR": atr}


# ----------------------------------------------------------------------------
# Backtest: gleiche Signal-Logik, vektorisiert über die ganze Historie
# ----------------------------------------------------------------------------
def signal_series(df: pd.DataFrame) -> dict:
    """Berechnet Score und Kauf-Signale für jeden Tag der Historie."""
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]

    sma10 = c.rolling(10).mean()
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    trend = c.rolling(10).mean() / c.rolling(30).mean() - 1
    intra10 = (c / o - 1).rolling(10).mean()
    on10 = (o / c.shift(1) - 1).rolling(10).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, float("nan")))

    relvol = v.rolling(10).mean() / v.rolling(60).mean()
    dist52 = c / c.rolling(252, min_periods=60).max() - 1

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    macd_comp = np.where(hist > hist.shift(1), 1.0, 0.4) * np.where(hist > 0, 1.0, -1.0)
    ma_comp = np.where((c > sma20) & (sma20 > sma50), 1.0,
                       np.where((c < sma20) & (sma20 < sma50), -1.0,
                                np.where(c > sma20, 0.3, -0.3)))
    relvol_comp = ((relvol - 1) / 0.5).clip(-1, 1).fillna(0) * np.where(trend >= 0, 1.0, -1.0)

    score = 100 * (
        0.30 * (trend / 0.05).clip(-1, 1)
        + 0.125 * (intra10 / 0.005).clip(-1, 1)
        + 0.125 * (on10 / 0.005).clip(-1, 1)
        + 0.10 * macd_comp
        + 0.10 * ma_comp
        + 0.10 * relvol_comp
        + 0.15 * ((dist52 + 0.15) / 0.15).clip(-1, 1)
    )

    trend_ok = (trend > 0) & (score >= 10)
    touched = (l <= sma10).rolling(3).max() >= 1
    pullback = trend_ok & touched & (c > sma10) & (c > o) & (rsi < 70)
    high20 = h.shift(1).rolling(20).max()
    breakout = trend_ok & (c > high20) & (relvol > 1.2)

    return {"pullback": pullback.fillna(False), "breakout": breakout.fillna(False),
            "sma20": sma20, "low5": l.rolling(5).min(), "atr": atr}


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(tickers: tuple, period: str, max_hold: int, rr: float = 2.0) -> pd.DataFrame:
    """Simuliert alle historischen Kauf-Signale. Einstieg am Folgetag zur
    Eröffnung; Exit über Stop, Ziel (rr·Risiko), SMA20-Bruch oder Zeit-Exit.
    Bei Stop und Ziel am selben Tag zählt konservativ der Stop."""
    history = load_history(tickers, period)
    trades = []

    for t, df in history.items():
        if len(df) < 80:
            continue
        sig = signal_series(df)
        o = df["Open"].to_numpy()
        h = df["High"].to_numpy()
        l = df["Low"].to_numpy()
        c = df["Close"].to_numpy()
        sma20 = sig["sma20"].to_numpy()
        low5 = sig["low5"].to_numpy()
        atr = sig["atr"].to_numpy()
        pullback = sig["pullback"].to_numpy()
        breakout = sig["breakout"].to_numpy()
        dates = df.index

        last_exit = -1
        for si in np.flatnonzero(pullback | breakout):
            if si <= last_exit or si + 1 >= len(df) or si < 60:
                continue
            entry_i = si + 1
            entry = o[entry_i]
            stop = max(low5[si], c[si] - 2 * atr[si])
            risk = entry - stop
            if not (risk > 0) or math.isnan(risk):
                continue
            target = entry + rr * risk

            exit_i, exit_price, grund = None, None, None
            last_i = min(entry_i + max_hold - 1, len(df) - 1)
            for j in range(entry_i, last_i + 1):
                if j > entry_i and o[j] <= stop:
                    exit_i, exit_price, grund = j, o[j], "Gap-Stop"
                elif j > entry_i and o[j] >= target:
                    exit_i, exit_price, grund = j, o[j], "Gap-Ziel"
                elif l[j] <= stop:
                    exit_i, exit_price, grund = j, stop, "Stop"
                elif h[j] >= target:
                    exit_i, exit_price, grund = j, target, "Ziel"
                elif c[j] < sma20[j]:
                    exit_i, exit_price, grund = j, c[j], "SMA20-Exit"
                if exit_i is not None:
                    break
            if exit_i is None:
                exit_i, exit_price, grund = last_i, c[last_i], "Zeit-Exit"

            last_exit = exit_i
            trades.append({
                "Ticker": t,
                "Setup": "Breakout" if breakout[si] else "Pullback",
                "Einstieg": dates[entry_i].date(),
                "Einstiegskurs": round(float(entry), 2),
                "Exit": dates[exit_i].date(),
                "Exitkurs": round(float(exit_price), 2),
                "Grund": grund,
                "Tage": int(exit_i - entry_i + 1),
                "R": round(float((exit_price - entry) / risk), 2),
                "Rendite": float(exit_price / entry - 1),
            })

    return pd.DataFrame(trades)


def ampel(score: float) -> str:
    if score >= 30:
        return "🟢 stark"
    if score >= 10:
        return "🟡 aufbauend"
    if score >= -10:
        return "⚪ neutral"
    return "🔴 schwach"


def build_table(tickers: dict, history: dict, gruppe: str) -> list:
    rows = []
    rates = load_fx_rates()
    for t, name in tickers.items():
        if t not in history:
            continue
        m = compute_metrics(history[t])
        s = compute_setup(history[t], m)
        row = {"Ticker": t, "Name": name, "Gruppe": gruppe, **m, **s}
        row.update(eur_columns(t, row, rates))
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------
# Chart
# ----------------------------------------------------------------------------
def make_chart(df: pd.DataFrame, title: str, setup: dict | None = None) -> go.Figure:
    c = df["Close"]
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        subplot_titles=(title, "Volumen", "MACD (12/26/9)"),
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="Kurs",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=c.rolling(10).mean(),
                             name="SMA 10", line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=c.rolling(30).mean(),
                             name="SMA 30", line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volumen",
                         marker_color="#8888cc"), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogramm",
                         marker_color=["#26a69a" if v >= 0 else "#ef5350" for v in hist]),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=macd, name="MACD",
                             line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=signal, name="Signal",
                             line=dict(width=1)), row=3, col=1)
    if setup and isinstance(setup.get("Stop"), float) and not math.isnan(setup["Stop"]):
        fig.add_hline(y=setup["Stop"], line_dash="dash", line_color="#ef5350",
                      annotation_text=f"Stop {setup['Stop']:.2f}", row=1, col=1)
        fig.add_hline(y=setup["Ziel (2R)"], line_dash="dash", line_color="#26a69a",
                      annotation_text=f"Ziel 2R {setup['Ziel (2R)']:.2f}", row=1, col=1)
    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.06),
                      margin=dict(t=60, b=20))
    return fig


# ----------------------------------------------------------------------------
# Tabellen-Formatierung
# ----------------------------------------------------------------------------
PCT_COLS = ["Trend 10/30", "Ø Intraday 10T", "Ø Intraday 30T", "Ø Overnight 10T",
            "Ø Overnight 30T", "Ø Tag 10T", "Ø Tag 30T", "Abstand 52W-Hoch",
            "Risiko"]
PRICE_COLS = ["Kurs", "Ø Open 10T", "Ø Close 10T", "Ø Open 30T", "Ø Close 30T",
              "SMA20", "SMA50", "Stop", "Ziel (2R)", "ATR"]
EUR_COLS = ["Kurs €", "Stop €", "Ziel €"]
FRONT_COLS = ["Ticker", "Name", "Sektor", "Sektor-Score", "Gruppe", "Setup",
              "Kurs", "Währung", "Kurs €", "Score", "Signal", "Stop",
              "Ziel (2R)", "Stop €", "Ziel €", "Risiko"]


def show_table(rows: list, key: str):
    df = pd.DataFrame(rows).sort_values("Score", ascending=False)
    df.insert(3, "Signal", df["Score"].map(ampel))
    front = [c for c in FRONT_COLS if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]]
    cfg = {c: st.column_config.NumberColumn(c, format="percent", step=0.0001) for c in PCT_COLS}
    cfg.update({c: st.column_config.NumberColumn(c, format="%.2f") for c in PRICE_COLS})
    cfg.update({c: st.column_config.NumberColumn(c, format="%.2f €") for c in EUR_COLS})
    cfg["Rel. Volumen"] = st.column_config.NumberColumn("Rel. Volumen", format="%.2f×")
    cfg["Score"] = st.column_config.NumberColumn("Score", format="%.1f")
    st.dataframe(df, hide_index=True, key=key, column_config=cfg,
                 use_container_width=True)
    return df


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
sectors = load_sectors()
OVERVIEW = "📊 Sektor-Übersicht"
SCANNER = "🎯 Setup-Scanner"
BACKTEST = "🧪 Backtest"

if "nav" not in st.session_state:
    st.session_state.nav = OVERVIEW
if st.session_state.nav not in (OVERVIEW, SCANNER, BACKTEST) and st.session_state.nav not in sectors:
    st.session_state.nav = OVERVIEW


def goto(view: str) -> None:
    """Callback: Ansicht wechseln (vor Widget-Instanziierung erlaubt)."""
    st.session_state.nav = view


def delete_sector(name: str) -> None:
    """Callback: Sektor löschen und zur Übersicht zurückkehren."""
    s = load_sectors()
    s.pop(name, None)
    save_sectors(s)
    st.session_state.nav = OVERVIEW

with st.sidebar:
    st.title("📈 Swing-Screener")
    st.radio("Ansicht", [OVERVIEW, SCANNER, BACKTEST] + list(sectors.keys()), key="nav")
    st.divider()
    with st.expander("➕ Neuen Sektor anlegen"):
        with st.form("new_sector", clear_on_submit=True):
            ns_name = st.text_input("Sektor-Name")
            ns_tickers = st.text_input("Kern-Ticker (kommagetrennt)", placeholder="z.B. NVDA, AMD")
            if st.form_submit_button("Anlegen") and ns_name and ns_tickers:
                sectors[ns_name] = {
                    "core": {t.strip().upper(): t.strip().upper()
                             for t in ns_tickers.split(",") if t.strip()},
                    "extended": {},
                }
                save_sectors(sectors)
                st.rerun()
    st.caption("Daten: Yahoo Finance (15 min Cache). "
               "Keine Anlageberatung — nur ein Screening-Werkzeug.")

nav = st.session_state.nav

# ============================ Übersicht ============================
if nav == OVERVIEW:
    st.header("Sektor-Übersicht — Grobdurchlauf über die Kern-Ticker")
    st.caption("Der Sektor-Score ist der Durchschnitt der Momentum-Scores der Kern-Ticker. "
               "Für Details und die erweiterte Ticker-Liste links den Sektor wählen oder unten klicken.")

    all_core = tuple(sorted({t for s in sectors.values() for t in s["core"]}))
    with st.spinner("Lade Kursdaten…"):
        history = load_history(all_core)

    failed = [t for t in all_core if t not in history]
    if failed:
        st.warning(f"Keine (ausreichenden) Daten für: {', '.join(failed)}")

    overview_rows = []
    for name, cfg in sectors.items():
        rows = build_table(cfg["core"], history, "Kern")
        if not rows:
            continue
        sdf = pd.DataFrame(rows)
        best = sdf.loc[sdf["Score"].idxmax()]
        overview_rows.append({
            "Sektor": name,
            "Score": round(float(sdf["Score"].mean()), 1),
            "Signal": ampel(float(sdf["Score"].mean())),
            "Ø Trend 10/30": float(sdf["Trend 10/30"].mean()),
            "Ø Intraday 10T": float(sdf["Ø Intraday 10T"].mean()),
            "Ø Overnight 10T": float(sdf["Ø Overnight 10T"].mean()),
            "Stärkster Wert": f"{best['Ticker']} ({best['Score']:.0f})",
            "Kern-Ticker": len(rows),
            "Erweitert": len(cfg["extended"]),
        })

    if overview_rows:
        odf = pd.DataFrame(overview_rows).sort_values("Score", ascending=False)
        st.dataframe(
            odf, hide_index=True, use_container_width=True,
            column_config={
                c: st.column_config.NumberColumn(c, format="percent", step=0.0001)
                for c in ["Ø Trend 10/30", "Ø Intraday 10T", "Ø Overnight 10T"]
            },
        )
        st.subheader("Drill-Down")
        cols = st.columns(min(4, len(odf)))
        for i, sector_name in enumerate(odf["Sektor"]):
            cols[i % len(cols)].button(f"🔍 {sector_name}", key=f"btn_{sector_name}",
                                       use_container_width=True,
                                       on_click=goto, args=(sector_name,))

# ============================ Setup-Scanner ============================
elif nav == SCANNER:
    st.header("Setup-Scanner — aktive Signale über alle Sektoren")
    with st.expander("ℹ️ Wie entstehen die Signale?"):
        st.markdown("""
Alle Signale gelten nur **in Trendrichtung** (Momentum-Score ≥ 10 und 10T-Schnitt über 30T-Schnitt):

- **🟢 Kauf: Pullback** — Kurs hat in den letzten 3 Tagen die SMA10 berührt und
  heute darüber (und über dem Open) geschlossen, RSI < 70. Rücksetzer im Trend.
- **🟢 Kauf: Breakout** — Schlusskurs über dem 20-Tage-Hoch bei relativem Volumen > 1,2×.
- **🟡 Beobachten** — Trend intakt, aber kein Einstiegs-Trigger. Nicht hinterherlaufen,
  auf den Rücksetzer warten.
- **🔴 Exit** — Trend war aufwärts, aber Schlusskurs unter der SMA20. Wer drin ist,
  sollte die Position überprüfen.
- **Stop** = letztes 5-Tage-Tief bzw. 2×ATR unter dem Kurs (der engere der beiden).
  **Ziel (2R)** = Einstieg + 2× Risiko, d.h. Chance/Risiko 2:1.

Ein Signal ist ein **Kandidat, kein Auftrag** — Chart ansehen, News prüfen, Positionsgröße
über das Risiko steuern (z.B. max. 1 % des Depots pro Trade riskieren).
""")

    all_tickers = {}
    ticker_sector = {}
    for sec_name, sec_cfg in sectors.items():
        for grp in ("core", "extended"):
            for t, name in sec_cfg[grp].items():
                if t not in all_tickers:          # Duplikate (z.B. VRT) nur einmal
                    all_tickers[t] = name
                    ticker_sector[t] = sec_name

    with st.spinner(f"Scanne {len(all_tickers)} Ticker…"):
        history = load_history(tuple(sorted(all_tickers)))

    metrics_cache = {t: compute_metrics(history[t]) for t in history}

    # Sektor-Momentum (Ø Score der Kern-Ticker) zu jeder Zeile dazu
    sector_scores = {}
    for sec_name, sec_cfg in sectors.items():
        core_scores = [metrics_cache[t]["Score"] for t in sec_cfg["core"] if t in metrics_cache]
        if core_scores:
            sector_scores[sec_name] = round(sum(core_scores) / len(core_scores), 1)

    scan_rows = []
    rates = load_fx_rates()
    for t, name in all_tickers.items():
        if t not in history:
            continue
        m = metrics_cache[t]
        s = compute_setup(history[t], m)
        row = {"Ticker": t, "Name": name, "Sektor": ticker_sector[t],
               "Sektor-Score": sector_scores.get(ticker_sector[t]), **m, **s}
        row.update(eur_columns(t, row, rates))
        scan_rows.append(row)

    buys = [r for r in scan_rows if r["Setup"].startswith("🟢")]
    exits = [r for r in scan_rows if r["Setup"].startswith("🔴")]
    watch = [r for r in scan_rows if r["Setup"].startswith("🟡")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gescannt", len(scan_rows))
    c2.metric("🟢 Kauf-Setups", len(buys))
    c3.metric("🔴 Exits", len(exits))
    c4.metric("🟡 Beobachten", len(watch))

    show_watch = st.toggle("🟡 Beobachten-Liste mit anzeigen", value=False)
    shown = buys + exits + (watch if show_watch else [])
    if shown:
        df_scan = show_table(shown, key="tbl_scanner")
        st.subheader("Chart")
        chart_ticker = st.selectbox(
            "Ticker", list(df_scan["Ticker"]),
            format_func=lambda t: f"{t} — {dict(zip(df_scan['Ticker'], df_scan['Name']))[t]}",
            key="scan_chart",
        )
        if chart_ticker in history:
            row = df_scan[df_scan["Ticker"] == chart_ticker].iloc[0].to_dict()
            st.plotly_chart(make_chart(history[chart_ticker], chart_ticker, row),
                            use_container_width=True)
    else:
        st.info("Aktuell keine aktiven Kauf- oder Exit-Signale. "
                "Schalte die Beobachten-Liste ein, um Kandidaten zu sehen.")

# ============================ Backtest ============================
elif nav == BACKTEST:
    st.header("Backtest — hätten die Signale funktioniert?")
    with st.expander("ℹ️ Methodik & Grenzen"):
        st.markdown("""
**Ablauf:** Die exakt gleiche Signal-Logik wie im Setup-Scanner läuft über die
historischen Kurse. Für jedes Kauf-Signal wird ein Trade simuliert:

- **Einstieg** am Folgetag zur Eröffnung (nicht zum Signal-Schlusskurs — realistischer)
- **Stop** = 5-Tage-Tief bzw. 2×ATR am Signaltag, **Ziel** = Einstieg + 2× Risiko
- **Exits**: Stop gerissen, Ziel erreicht, Schluss unter SMA20 oder Zeit-Exit.
  Wären Stop *und* Ziel am selben Tag erreichbar, zählt konservativ der **Stop**.
- Pro Ticker immer nur eine offene Position.

**Grenzen — wichtig für die Interpretation:**
- **Survivorship-Bias:** Getestet werden die *heutigen* Listen. Aktien, die stark
  gelaufen sind, haben es eher in deine Listen geschafft — die Ergebnisse sind
  dadurch tendenziell **zu optimistisch**.
- Keine Gebühren, kein Spread, keine Slippage eingerechnet.
- **R** = Gewinn/Verlust in Vielfachen des Einstiegsrisikos. Ø R > 0 über viele
  Trades = die Regeln hätten eine Edge gehabt.
""")

    c1, c2, c3 = st.columns(3)
    scope = c1.selectbox("Universum", ["Alle Sektoren"] + list(sectors.keys()))
    period_label = c2.selectbox("Zeitraum", ["2 Jahre", "3 Jahre", "5 Jahre"])
    max_hold = c3.slider("Max. Haltedauer (Handelstage)", 10, 40, 25)
    period = {"2 Jahre": "2y", "3 Jahre": "3y", "5 Jahre": "5y"}[period_label]

    ticker_sector = {}
    for sec_name, sec_cfg in sectors.items():
        if scope != "Alle Sektoren" and sec_name != scope:
            continue
        for grp in ("core", "extended"):
            for t in sec_cfg[grp]:
                ticker_sector.setdefault(t, sec_name)

    if st.button("🧪 Backtest starten", type="primary"):
        with st.spinner(f"Simuliere Signale für {len(ticker_sector)} Ticker über {period_label}…"):
            st.session_state.bt_result = run_backtest(
                tuple(sorted(ticker_sector)), period, max_hold)
            st.session_state.bt_label = f"{scope} · {period_label} · max. {max_hold} Tage"

    trades = st.session_state.get("bt_result")
    if trades is None:
        st.info("Universum und Zeitraum wählen, dann den Backtest starten.")
    elif trades.empty:
        st.warning("Keine Trades im gewählten Universum/Zeitraum gefunden.")
    else:
        trades = trades.copy()
        trades["Sektor"] = trades["Ticker"].map(ticker_sector).fillna("—")
        st.caption(f"Letzter Lauf: {st.session_state.get('bt_label', '')}")

        wins = (trades["R"] > 0).sum()
        pos_r = trades.loc[trades["R"] > 0, "R"].sum()
        neg_r = abs(trades.loc[trades["R"] < 0, "R"].sum())
        pf = pos_r / neg_r if neg_r > 0 else float("inf")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Trades", len(trades))
        k2.metric("Trefferquote", f"{wins / len(trades):.0%}")
        k3.metric("Ø R pro Trade", f"{trades['R'].mean():.2f}")
        k4.metric("Profitfaktor", f"{pf:.2f}")
        k5.metric("Gesamt-R", f"{trades['R'].sum():.1f}")
        k6.metric("Ø Haltedauer", f"{trades['Tage'].mean():.1f} Tage")

        # Equity-Kurve (kumuliertes R nach Exit-Datum)
        eq = trades.sort_values("Exit")[["Exit", "R"]].copy()
        eq["Kumuliertes R"] = eq["R"].cumsum()
        fig = go.Figure(go.Scatter(x=eq["Exit"], y=eq["Kumuliertes R"],
                                   mode="lines", name="Kumuliertes R"))
        fig.add_hline(y=0, line_dash="dot", line_color="#888")
        fig.update_layout(title="Equity-Kurve (in R, ein Trade = gleiches Risiko)",
                          height=350, margin=dict(t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        def agg_table(group_col: str) -> pd.DataFrame:
            g = trades.groupby(group_col).agg(
                Trades=("R", "size"),
                Trefferquote=("R", lambda x: (x > 0).mean()),
                Ø_R=("R", "mean"),
                Gesamt_R=("R", "sum"),
            ).reset_index().sort_values("Gesamt_R", ascending=False)
            g.columns = [group_col, "Trades", "Trefferquote", "Ø R", "Gesamt-R"]
            return g

        pct_cfg = {"Trefferquote": st.column_config.NumberColumn(
            "Trefferquote", format="percent", step=0.01)}
        cb1, cb2 = st.columns(2)
        with cb1:
            st.subheader("Nach Setup-Typ")
            st.dataframe(agg_table("Setup"), hide_index=True,
                         use_container_width=True, column_config=pct_cfg)
            st.subheader("Nach Exit-Grund")
            st.dataframe(agg_table("Grund"), hide_index=True,
                         use_container_width=True, column_config=pct_cfg)
        with cb2:
            st.subheader("Nach Sektor")
            st.dataframe(agg_table("Sektor"), hide_index=True,
                         use_container_width=True, column_config=pct_cfg)

        with st.expander(f"Alle {len(trades)} Trades anzeigen"):
            st.dataframe(
                trades.sort_values("Einstieg", ascending=False),
                hide_index=True, use_container_width=True,
                column_config={"Rendite": st.column_config.NumberColumn(
                    "Rendite", format="percent", step=0.0001)},
            )

# ============================ Sektor-Detail ============================
else:
    cfg = sectors[nav]
    st.header(f"{nav} — Detailansicht")

    tickers = tuple(sorted(set(cfg["core"]) | set(cfg["extended"])))
    with st.spinner("Lade Kursdaten…"):
        history = load_history(tickers)

    failed = [t for t in tickers if t not in history]
    if failed:
        st.warning(f"Keine (ausreichenden) Daten für: {', '.join(failed)}")

    rows = build_table(cfg["core"], history, "Kern") + \
           build_table(cfg["extended"], history, "Erweitert")

    if rows:
        core_scores = [r["Score"] for r in rows if r["Gruppe"] == "Kern"]
        c1, c2, c3 = st.columns(3)
        if core_scores:
            sector_score = sum(core_scores) / len(core_scores)
            c1.metric("Sektor-Score (Kern)", f"{sector_score:.1f}", ampel(sector_score))
        best = max(rows, key=lambda r: r["Score"])
        c2.metric("Stärkster Wert", best["Ticker"], f"{best['Score']:.1f}")
        worst = min(rows, key=lambda r: r["Score"])
        c3.metric("Schwächster Wert", worst["Ticker"], f"{worst['Score']:.1f}")

        df_shown = show_table(rows, key=f"tbl_{nav}")

        st.subheader("Chart")
        chart_ticker = st.selectbox(
            "Ticker", list(df_shown["Ticker"]),
            format_func=lambda t: f"{t} — {dict(zip(df_shown['Ticker'], df_shown['Name']))[t]}",
        )
        if chart_ticker in history:
            row = df_shown[df_shown["Ticker"] == chart_ticker].iloc[0].to_dict()
            st.plotly_chart(make_chart(history[chart_ticker], chart_ticker, row),
                            use_container_width=True)
    else:
        st.info("Keine Daten für diesen Sektor verfügbar.")

    # --------- Sektor erweitern ---------
    st.divider()
    with st.expander("⚙️ Sektor bearbeiten (Ticker hinzufügen / entfernen)"):
        col_add, col_del = st.columns(2)
        with col_add:
            with st.form(f"add_{nav}", clear_on_submit=True):
                st.markdown("**Ticker hinzufügen**")
                new_t = st.text_input("Ticker (Yahoo-Format, z.B. LPK.DE, SIVE.ST)")
                new_name = st.text_input("Anzeigename (optional)")
                group = st.radio("Gruppe", ["Erweitert", "Kern"], horizontal=True)
                if st.form_submit_button("Hinzufügen") and new_t.strip():
                    key = "core" if group == "Kern" else "extended"
                    t = new_t.strip().upper()
                    sectors[nav][key][t] = new_name.strip() or t
                    save_sectors(sectors)
                    st.rerun()
        with col_del:
            st.markdown("**Ticker entfernen**")
            all_t = list(cfg["core"]) + list(cfg["extended"])
            to_del = st.multiselect("Auswahl", all_t, key=f"del_{nav}")
            if st.button("Entfernen", key=f"delbtn_{nav}") and to_del:
                for t in to_del:
                    sectors[nav]["core"].pop(t, None)
                    sectors[nav]["extended"].pop(t, None)
                save_sectors(sectors)
                st.rerun()
        st.button(f"🗑️ Sektor „{nav}“ löschen", key=f"delsec_{nav}",
                  on_click=delete_sector, args=(nav,))
