"""Quantitative metric library: scores each metric from -1 to +1 against sector-aware thresholds.
Metrics that are not meaningful for the company's situation (e.g. P/E for a loss-maker) are skipped and listed in df.attrs["na"], never scored."""
import numpy as np
import pandas as pd
import research as R

# (category, label, yfinance info key, "good" threshold, "bad" threshold)
# If good < bad, lower is better.
INFO_METRICS = [
    ("Valuation", "Trailing P/E", "trailingPE", 15, 35),
    ("Valuation", "Forward P/E", "forwardPE", 14, 32),
    ("Valuation", "PEG ratio", "pegRatio", 1, 2.5),
    ("Valuation", "Price / Book", "priceToBook", 2, 10),
    ("Valuation", "Price / Sales", "priceToSalesTrailing12Months", 2, 10),
    ("Valuation", "EV / EBITDA", "enterpriseToEbitda", 9, 25),
    ("Valuation", "EV / Revenue", "enterpriseToRevenue", 2, 10),
    ("Profitability", "Net margin", "profitMargins", .15, .03),
    ("Profitability", "Operating margin", "operatingMargins", .20, .05),
    ("Profitability", "Gross margin", "grossMargins", .50, .20),
    ("Profitability", "EBITDA margin", "ebitdaMargins", .25, .08),
    ("Profitability", "Return on equity", "returnOnEquity", .18, .04),
    ("Profitability", "Return on assets", "returnOnAssets", .08, .01),
    ("Growth", "Revenue growth (YoY)", "revenueGrowth", .12, -.02),
    ("Growth", "Earnings growth (YoY)", "earningsGrowth", .12, -.05),
    ("Growth", "Quarterly earnings growth", "earningsQuarterlyGrowth", .12, -.05),
    ("Financial health", "Current ratio", "currentRatio", 2, .9),
    ("Financial health", "Quick ratio", "quickRatio", 1.2, .6),
    ("Financial health", "Debt / Equity (%)", "debtToEquity", 40, 180),
    ("Analyst", "Analyst rating (1=Strong Buy, 5=Sell)", "recommendationMean", 2.0, 3.5),
    ("Risk", "Beta", "beta", .9, 1.8),
    ("Risk", "Short interest (% of float)", "shortPercentOfFloat", .02, .12),
    ("Risk", "Institutional ownership", "heldPercentInstitutions", .70, .20),
]
NEG_IS_BAD = {"trailingPE", "forwardPE", "pegRatio", "enterpriseToEbitda"}  # loss-making = bearish


CYCLICAL = {"Energy", "Basic Materials", "Industrials", "Consumer Cyclical"}
_FIN_SKIP = {"Debt / Equity (%)", "Current ratio", "Quick ratio", "Cash / Debt", "Net debt / EBITDA", "EV / EBITDA",
             "EV / Revenue", "Price / Sales", "Gross margin", "EBITDA margin", "Operating margin",
             "Free cash flow yield", "FCF / Net income", "Interest coverage", "Total debt change (YoY)",
             "Share of years with positive FCF", "Gross margin change (YoY)", "Operating margin change (YoY)",
             "Operating margin stability (std dev)"}
_DEF_GROWTH = {k: (.05, -.01) for k in ("Revenue growth (YoY)", "Revenue growth (latest fiscal yr)", "Revenue 3-yr CAGR")}
# Sector rules of thumb: "over" = (good, bad) thresholds replacing the generic ones, "skip" = metrics that don't
# mean much for the sector, "exit" = plausible long-run P/E range used in 5-year scenarios.
PROFILES = {
    "Technology": {"over": {"Trailing P/E": (25, 55), "Forward P/E": (22, 40), "Price / Sales": (5, 16), "Price / Book": (6, 20),
                            "EV / EBITDA": (18, 38), "EV / Revenue": (5, 14), "PEG ratio": (1.5, 3), "Gross margin": (.65, .40),
                            "Operating margin": (.25, .08), "Debt / Equity (%)": (40, 120)}, "skip": set(), "exit": (18, 32)},
    "Financial Services": {"over": {"Price / Book": (1.0, 2.5), "Trailing P/E": (10, 20), "Forward P/E": (10, 18),
                                    "Return on equity": (.13, .06), "Return on assets": (.012, .004), "Net margin": (.25, .10)},
                           "skip": _FIN_SKIP, "exit": (9, 16)},
    "Healthcare": {"over": {"Trailing P/E": (22, 45), "Forward P/E": (18, 32), "Gross margin": (.65, .35), "Operating margin": (.20, .02),
                            "Net margin": (.15, 0), "Price / Sales": (3, 10), "EV / EBITDA": (12, 25)}, "skip": set(), "exit": (16, 28)},
    "Consumer Cyclical": {"over": {"Trailing P/E": (15, 32), "Forward P/E": (14, 28), "Net margin": (.08, .02), "Operating margin": (.10, .03),
                                   "Gross margin": (.35, .15), "EV / EBITDA": (9, 18), "Price / Sales": (1, 4),
                                   "Debt / Equity (%)": (70, 200)}, "skip": set(), "exit": (12, 22)},
    "Consumer Defensive": {"over": {"Trailing P/E": (18, 30), "Forward P/E": (17, 26), "Net margin": (.07, .02), "Operating margin": (.12, .04),
                                    "Gross margin": (.35, .15), "EV / EBITDA": (11, 18), "Price / Sales": (1, 3),
                                    "Debt / Equity (%)": (80, 200), **_DEF_GROWTH}, "skip": set(), "exit": (16, 24)},
    "Industrials": {"over": {"Trailing P/E": (17, 32), "Forward P/E": (16, 28), "Net margin": (.09, .03), "Operating margin": (.14, .05),
                             "Gross margin": (.32, .15), "EV / EBITDA": (11, 20), "Price / Sales": (1.5, 4),
                             "Debt / Equity (%)": (70, 170)}, "skip": set(), "exit": (14, 22)},
    "Energy": {"over": {"Trailing P/E": (9, 20), "Forward P/E": (9, 18), "EV / EBITDA": (4.5, 9), "Price / Sales": (1, 3), "Net margin": (.10, .02),
                        "Operating margin": (.15, .03), "Gross margin": (.40, .15), "Debt / Equity (%)": (35, 100),
                        "Price / Book": (1.2, 3)}, "skip": set(), "exit": (8, 14)},
    "Utilities": {"over": {"Trailing P/E": (16, 25), "Forward P/E": (15, 22), "Price / Book": (1.6, 3), "EV / EBITDA": (10, 15),
                           "Debt / Equity (%)": (110, 260), "Net margin": (.12, .05), "Operating margin": (.20, .10),
                           "Return on equity": (.10, .05), "Current ratio": (.9, .5), "Quick ratio": (.6, .3),
                           "Interest coverage": (3.5, 1.8), "Net debt / EBITDA": (4.5, 7), **_DEF_GROWTH},
                  "skip": {"Free cash flow yield", "FCF / Net income", "Share of years with positive FCF", "Gross margin",
                           "Gross margin change (YoY)"}, "exit": (14, 20)},
    "Real Estate": {"over": {"Price / Book": (1.3, 3), "EV / EBITDA": (14, 25), "Debt / Equity (%)": (90, 220), "Net margin": (.20, .05),
                             "Return on equity": (.08, .02), "Interest coverage": (3, 1.5)},
                    "skip": {"Trailing P/E", "Forward P/E", "PEG ratio", "Free cash flow yield", "FCF / Net income", "Gross margin",
                             "Gross margin change (YoY)", "Current ratio", "Quick ratio"}, "exit": (14, 22)},
    "Basic Materials": {"over": {"Trailing P/E": (11, 25), "Forward P/E": (11, 22), "EV / EBITDA": (6, 12), "Price / Sales": (1.2, 3.5),
                                 "Net margin": (.10, .03), "Operating margin": (.15, .05), "Gross margin": (.30, .12),
                                 "Debt / Equity (%)": (35, 100)}, "skip": set(), "exit": (10, 17)},
    "Communication Services": {"over": {"Trailing P/E": (18, 35), "Forward P/E": (16, 28), "EV / EBITDA": (9, 18), "Operating margin": (.20, .08),
                                        "Net margin": (.12, .04), "Debt / Equity (%)": (70, 200), "Price / Sales": (2.5, 8)},
                               "skip": set(), "exit": (14, 26)},
}
PROFILES["Financial (asset-light)"] = {"over": {"Trailing P/E": (22, 40), "Forward P/E": (20, 34), "Operating margin": (.30, .10),
                                             "Net margin": (.25, .08), "Price / Book": (8, 25)}, "skip": set(), "exit": (20, 30)}
DEFAULT = {"over": {}, "skip": set(), "exit": (15, 25)}
for _s in ("Utilities", "Real Estate", "Energy", "Basic Materials"):   # capex intensity is the business model there, not a warning
    PROFILES[_s]["skip"] = set(PROFILES[_s]["skip"]) | {"Capex / revenue"}
# Not meaningful while a company is loss-making. Skipped (never scored, to avoid penalising one underlying fact several times);
# growth-adjusted valuation, funding runway and margin trajectory are scored instead.
LOSS_SKIP = {"Trailing P/E", "Forward P/E", "PEG ratio", "EV / EBITDA", "Net margin", "Operating margin", "EBITDA margin", "Return on equity",
             "Return on assets", "Earnings growth (YoY)", "Quarterly earnings growth", "Net income growth (latest fiscal yr)", "Operating income growth",
             "FCF / Net income", "Interest coverage", "Share of years profitable", "Share of years with positive FCF", "Operating margin stability (std dev)",
             "Free cash flow yield", "Free cash flow growth", "Net debt / EBITDA", "Price / Sales", "EV / Revenue", "Capex / revenue"}


def profile_for(info, sector):
    if sector == "Financial Services" and not R.use_book_value(info, sector):
        return PROFILES["Financial (asset-light)"]
    return PROFILES.get(sector, DEFAULT)


def lin(v, good, bad):
    return float(np.clip(2 * (v - bad) / (good - bad) - 1, -1, 1))


def row(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


def growth(s):
    if s is None or len(s) < 2 or s.iloc[1] <= 0:
        return None
    return s.iloc[0] / s.iloc[1] - 1


def build_metrics(info, hist, inc, cf, bs, prof=None, loss_making=False):
    prof = prof or DEFAULT
    rows, na = [], {}

    def add(cat, label, v, good=None, bad=None, neg_bad=False, score=None):
        if label in prof["skip"]:
            na[label] = "not applicable to this sector"
            return
        if loss_making and label in LOSS_SKIP:
            na[label] = "not meaningful while the company is loss-making"
            return
        if label in prof["over"] and score is None:
            good, bad = prof["over"][label]
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if np.isnan(v) or np.isinf(v):
            return
        if score is None:
            score = -1.0 if (neg_bad and v <= 0) else lin(v, good, bad)
        rows.append({"Category": cat, "Metric": label, "Value": v, "Score": score})

    # --- Fundamentals from the info table ---
    for cat, label, key, good, bad in INFO_METRICS:
        add(cat, label, info.get(key), good, bad, key in NEG_IS_BAD)

    mc, fcf_ttm = info.get("marketCap"), info.get("freeCashflow")
    if mc and fcf_ttm:
        add("Valuation", "Free cash flow yield", fcf_ttm / mc, .05, 0)
    cash, debt_i, ebitda = info.get("totalCash"), info.get("totalDebt"), info.get("ebitda")
    if cash is not None and debt_i:
        add("Financial health", "Cash / Debt", cash / debt_i, 1, .2)
    if debt_i is not None and cash is not None and ebitda and ebitda > 0:
        add("Financial health", "Net debt / EBITDA", (debt_i - cash) / ebitda, 1, 4)
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    for key, label, g, b in (("targetMeanPrice", "Upside to mean analyst target", .15, -.05),
                             ("targetMedianPrice", "Upside to median analyst target", .15, -.05)):
        if price and info.get(key):
            add("Analyst", label, info[key] / price - 1, g, b)

    fe, te = info.get("forwardEps"), info.get("trailingEps")
    if fe and te and te > 0 and fe > 0:
        add("Growth", "Forward EPS growth", fe / te - 1, .12, 0)

    # --- Statement-derived trends ---
    rev = row(inc, ["Total Revenue", "Operating Revenue"])
    ni = row(inc, ["Net Income"])
    oi = row(inc, ["Operating Income"])
    gp = row(inc, ["Gross Profit"])
    ebit = row(inc, ["EBIT"])
    intx = row(inc, ["Interest Expense"])
    fcf = row(cf, ["Free Cash Flow"])
    sh = row(bs, ["Ordinary Shares Number", "Share Issued"])
    debt = row(bs, ["Total Debt"])

    add("Growth", "Revenue growth (latest fiscal yr)", growth(rev), .12, 0)
    if rev is not None and len(rev) >= 4 and rev.iloc[3] > 0 and rev.iloc[0] > 0:
        add("Growth", "Revenue 3-yr CAGR", (rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1, .12, 0)
    add("Growth", "Net income growth (latest fiscal yr)", growth(ni), .12, 0)
    add("Growth", "Operating income growth", growth(oi), .12, 0)
    add("Growth", "Free cash flow growth", growth(fcf), .12, 0)
    if rev is not None and len(rev) >= 2:
        if gp is not None and len(gp) >= 2:
            add("Profitability", "Gross margin change (YoY)", gp.iloc[0] / rev.iloc[0] - gp.iloc[1] / rev.iloc[1], .01, -.02)
        if oi is not None and len(oi) >= 2:
            add("Profitability", "Operating margin change (YoY)", oi.iloc[0] / rev.iloc[0] - oi.iloc[1] / rev.iloc[1], .01, -.02)
    if fcf is not None and ni is not None and ni.iloc[0] > 0:
        add("Profitability", "FCF / Net income", fcf.iloc[0] / ni.iloc[0], 1, .5)
    if sh is not None and len(sh) >= 2:
        add("Financial health", "Share count change (buybacks < 0)", sh.iloc[0] / sh.iloc[1] - 1, -.01, .03)
    if debt is not None and len(debt) >= 2 and debt.iloc[1] > 0:
        add("Financial health", "Total debt change (YoY)", debt.iloc[0] / debt.iloc[1] - 1, -.05, .15)
    if ebit is not None and intx is not None and abs(intx.iloc[0]) > 0:
        add("Financial health", "Interest coverage", ebit.iloc[0] / abs(intx.iloc[0]), 8, 2)

    # --- Long-term track record from statements ---
    if rev is not None and len(rev) >= 3:
        add("Track record", "Share of years with revenue growth", (rev.diff(-1).dropna() > 0).mean(), 1, .5)
    if ni is not None and len(ni) >= 3:
        add("Track record", "Share of years profitable", (ni > 0).mean(), 1, .5)
    if fcf is not None and len(fcf) >= 3:
        add("Track record", "Share of years with positive FCF", (fcf > 0).mean(), 1, .5)
    if oi is not None and rev is not None:
        m = (oi / rev).dropna()
        if len(m) >= 3:
            add("Track record", "Operating margin stability (std dev)", m.std(), .02, .08)

    # --- Funding, capital intensity and growth-adjusted valuation ---
    fcf0 = info.get("freeCashflow") if info.get("freeCashflow") is not None else (float(fcf.iloc[0]) if fcf is not None else None)
    cash0 = info.get("totalCash")
    if fcf0 is not None and fcf0 < 0 and cash0:
        add("Financial health", "Cash runway (years)", cash0 / -fcf0, 3, 1.0)
    capex = row(cf, ["Capital Expenditure", "Capital Expenditures"])
    if capex is not None and rev is not None and rev.iloc[0] > 0:
        add("Financial health", "Capex / revenue", abs(float(capex.iloc[0])) / float(rev.iloc[0]), .10, .45)
    evs = info.get("enterpriseToRevenue") or ((info.get("enterpriseValue") or 0) / info["totalRevenue"] if info.get("totalRevenue") else None)
    g_ = info.get("revenueGrowth")
    if loss_making and evs and evs > 0 and g_ and g_ > .05:
        add("Valuation", "EV / Sales per point of growth", evs / (g_ * 100), .15, .60)

    # --- Technicals and risk from price history ---
    c = hist["Close"].dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    if len(c) > 60:
        last = c.iloc[-1]
        r = c.pct_change().dropna()
        for n in (20, 50, 100, 200):
            if len(c) > n:
                add("Technicals", f"Price vs {n}-day average", last / c.rolling(n).mean().iloc[-1] - 1, .05, -.05)
        if len(c) > 200:
            add("Technicals", "50-day vs 200-day average",
                c.rolling(50).mean().iloc[-1] / c.rolling(200).mean().iloc[-1] - 1, .03, -.03)
        for n, lab, g, b in ((21, "1-month", .05, -.05), (63, "3-month", .10, -.10),
                             (126, "6-month", .15, -.10), (252, "12-month", .25, -.10)):
            if len(c) > n:
                add("Technicals", f"{lab} return", last / c.iloc[-n - 1] - 1, g, b)
        d = c.diff()
        up, dn = d.clip(lower=0).rolling(14).mean().iloc[-1], (-d.clip(upper=0)).rolling(14).mean().iloc[-1]
        if dn and dn > 0:
            rsi = 100 - 100 / (1 + up / dn)
            add("Technicals", "RSI (14)", rsi, score=1.0 if rsi < 30 else (-1.0 if rsi > 75 else 0.0))
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        add("Technicals", "MACD histogram / price", (macd - macd.ewm(span=9).mean()).iloc[-1] / last, .01, -.01)
        yr = c.tail(252)
        add("Technicals", "Distance from 52-week high", last / yr.max() - 1, -.05, -.30)
        vol = r.tail(252).std() * np.sqrt(252)
        add("Risk", "Annualized volatility", vol, .25, .60)
        add("Risk", "Max drawdown (1 yr)", (yr / yr.cummax() - 1).min(), -.15, -.45)
        if vol > 0:
            add("Risk", "1-yr return / volatility", r.tail(252).mean() * 252 / vol, 1, 0)
    if len(c) > 756:  # 3+ years of history
        for yrs in (3, 5, 10):
            n = 252 * yrs
            if len(c) > n:
                add("Track record", f"{yrs}-yr annualized price return", (c.iloc[-1] / c.iloc[-n - 1]) ** (1 / yrs) - 1, .15, .03)
        for yrs in (5, 10):
            n = 252 * yrs
            if len(c) > n:
                win = c.tail(n)
                add("Track record", f"Max drawdown ({yrs} yr)", (win / win.cummax() - 1).min(), -.30, -.60)
        yearly = c.groupby(c.index.year).last().pct_change().dropna()
        if len(yearly) >= 3:
            add("Track record", "Share of calendar years up", (yearly > 0).mean(), .8, .5)
    out = pd.DataFrame(rows)
    out.attrs["na"] = na
    return out
