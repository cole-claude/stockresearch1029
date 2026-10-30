"""Data access: Yahoo Finance (yfinance), SEC EDGAR filings, optional peers. Cached; never invents values."""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timezone

import research as R
import sec as SEC


@st.cache_data(ttl=3600, show_spinner=False)
def load(sym):
    t = yf.Ticker(sym)

    def safe(f):
        try:
            return f()
        except Exception:
            return None
    hist = safe(lambda: t.history(period="10y", auto_adjust=True))
    est = {k: safe(f) for k, f in (("rev", lambda: t.revenue_estimate), ("eps", lambda: t.earnings_estimate),
                                   ("trend", lambda: t.eps_trend), ("hist", lambda: t.earnings_history))}
    spy = safe(lambda: yf.Ticker("SPY").history(period="5y", auto_adjust=True))
    rf = safe(lambda: float(yf.Ticker("^TNX").history(period="5d")["Close"].dropna().iloc[-1]) / 100)
    extra = {"est": est, "spy": spy, "rf": rf, "fetched": datetime.now(timezone.utc)}
    return (safe(lambda: t.info) or {}, hist, safe(lambda: t.income_stmt),
            safe(lambda: t.cashflow), safe(lambda: t.balance_sheet), extra)


@st.cache_data(ttl=86400, show_spinner=False)
def load_peers(sym, industry_key):
    try:
        syms = [s for s in yf.Industry(industry_key).top_companies.index if s != sym][:6]
    except Exception:
        return None
    rows = []
    for s in syms:
        try:
            i = yf.Ticker(s).info
            rows.append({"Ticker": s, "Fwd P/E": i.get("forwardPE"), "EV/EBITDA": i.get("enterpriseToEbitda"),
                         "P/S": i.get("priceToSalesTrailing12Months"), "Op margin": i.get("operatingMargins"),
                         "Rev growth": i.get("revenueGrowth")})
        except Exception:
            pass
    return pd.DataFrame(rows) if rows else None


def fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


def render(res, df, price):
    st.write(f"Metric-model score: **{res['final']:+.2f}** (an input to the recommendation, not a separate rating)")
    st.subheader("Scenarios")
    rows = []
    for n, p, pr, *ann in res["scen"]:
        r = {"Case": n, "Target": f"${p:,.2f}", "vs. now": f"{p / price - 1:+.1%}"}
        if ann:
            r["Per year"] = f"{ann[0]:+.1%}"
        r["Probability"] = f"{pr:.0%}"
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.write(f"Probability-weighted expected {'annual return' if len(res['scen'][0]) > 3 else 'move'}: "
             f"**{res['exp_ret']:+.1%}**")
    if res.get("note"):
        st.caption(res["note"])
    used = [k for k, v in res["weights"].items() if v > 0]
    st.subheader("Category scores")
    st.bar_chart(res["cat_scores"].reindex(used).dropna().round(2))
    d = df[df.Category.isin(used)]
    st.subheader("🐂 Bull case: what's working")
    for _, m in d[d.Score > .3].sort_values("Score", ascending=False).head(6).iterrows():
        st.write(f"🟢 {m.Metric}: {fmt(m.Value)}")
    st.subheader("🐻 Bear case: what's worrying")
    for _, m in d[d.Score < -.3].sort_values("Score").head(6).iterrows():
        st.write(f"🔴 {m.Metric}: {fmt(m.Value)}")


S2C = {"Technology": 3, "Communication Services": 1.2, "Healthcare": 1.5, "Consumer Cyclical": 2, "Consumer Defensive": 2,
       "Industrials": 1.5, "Energy": .8, "Basic Materials": .8, "Utilities": .4, "Real Estate": .3}
NOT_EVALUATED = ["AI opportunity / disruption scoring (needs filing and transcript text analysed by an LLM)",
                 "Competitive moat and management quality (qualitative, needs transcripts and filings)",
                 "SEC XBRL, earnings-call transcripts, segment and geographic revenue, backlog and retention KPIs",
                 "Options-implied volatility, credit ratings, insider-transaction analysis",
                 "Industry operating indicators (wafer starts, ARR, comps, NIM, etc.)",
                 "Portfolio analyzer, screener across a universe, database persistence, background jobs"]


def _g(df, r, c):
    try:
        return float(df.loc[r, c])
    except Exception:
        return None


def pct(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.1%}"


def _secret(name):
    try:
        return st.secrets[name]
    except Exception:
        import os
        return os.environ.get(name)


@st.cache_data(ttl=86400, show_spinner=False)
def load_sec(sym, ua):
    """Official SEC XBRL statements for US filers. Returns {'err': reason} when unavailable; never raises."""
    tj, err = SEC.get_json(SEC.TICKERS, ua)
    if err:
        return {"err": f"SEC ticker list unreachable ({err})"}
    cik = SEC.cik_for(sym, tj)
    if cik is None:
        return {"err": "ticker not in the SEC filer list (ETF, non-US company, or foreign private issuer filing 20-F/40-F)"}
    facts, err = SEC.get_json(SEC.BASE.format(cik=cik), ua)
    if err:
        return {"err": f"SEC company facts unreachable ({err})"}
    inc, cf, bs, meta = SEC.normalize(facts)
    return {"inc": inc, "cf": cf, "bs": bs, "meta": meta, "err": None if inc is not None else "no annual 10-K revenue facts in the filing data"}


def _years(df):
    r = R.getrow(df, ["Total Revenue", "Operating Revenue"])
    return 0 if r is None else len(r)


def prepare(sym):
    """One place that decides which source feeds each input, and records it. Never invents a value."""
    info, hist, inc, cf, bs, extra = load(sym)
    info, extra = dict(info), dict(extra)
    src, conflicts = [], []
    ua = _secret("SEC_USER_AGENT") or "StockRater/1.0 (set SEC_USER_AGENT in app secrets: app name + your email)"
    sd = load_sec(sym, ua)
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(hist["Close"].iloc[-1]) if hist is not None and not hist.empty else None)
    yahoo_years = _years(inc)
    sec_ok = sd.get("inc") is not None and _years(sd["inc"]) >= 3 and _years(sd["inc"]) >= yahoo_years
    if sec_ok:
        inc, cf, bs = sd["inc"], sd["cf"], sd["bs"]
        m = sd["meta"]
        src.append(("Financial statements", "SEC EDGAR XBRL (official)", f"{_years(inc)} fiscal years; latest FY ending {m['latest_fy_end']}, filed {m['filed']}"))
        if m.get("age_days", 0) > 450:
            conflicts.append(f"SEC statements are {m['age_days']} days old (latest 10-K FY {m['latest_fy_end']}); newer quarters are not included.")
        fb, notes = SEC.fallback_info(m, inc, cf, bs, price)
        yrev = info.get("totalRevenue")
        for k, v in fb.items():
            if info.get(k) in (None, 0):
                info[k] = v; src.append((k, "SEC EDGAR", notes.get(k, "latest fiscal year")))
        if yrev and fb.get("totalRevenue") and not .6 <= yrev / fb["totalRevenue"] <= 1.6:
            conflicts.append(f"Yahoo TTM revenue ({yrev:,.0f}) and SEC latest-FY revenue ({fb['totalRevenue']:,.0f}) differ by more than 60%: check for a recent acquisition, fiscal-year change or units mismatch.")
    else:
        src.append(("Financial statements", "Yahoo Finance via yfinance (unofficial)", f"{yahoo_years} fiscal years" + (f"; SEC unavailable: {sd.get('err')}" if sd.get("err") else "")))
    src.append(("Price and history", "Yahoo Finance via yfinance (delayed)", f"last close {hist.index[-1]:%Y-%m-%d}" if hist is not None and not hist.empty else "none returned"))
    src.append(("Analyst estimates", "Yahoo Finance via yfinance" if any(v is not None for v in extra["est"].values()) else "unavailable", "consensus; no free source is fully reliable"))
    if info.get("beta") is None and hist is not None and not hist.empty:
        q = R.quant(hist["Close"], extra["spy"])
        if "Beta vs SPY" in q:
            info["beta"] = q["Beta vs SPY"]; src.append(("Beta", "computed from price history vs SPY", "Yahoo returned none"))
    src.append(("Risk-free rate", "Yahoo ^TNX (live)" if extra["rf"] else "assumption 4.2% (live rate unavailable)", ""))
    extra["sources"], extra["conflicts"], extra["sec_status"] = src, conflicts, ("used" if sec_ok else (sd.get("err") or "not used (fewer fiscal years than Yahoo)"))
    return info, hist, inc, cf, bs, extra
