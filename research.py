"""Shared analytics helpers: statement parsing, forensic checks, data validation, price statistics. No valuation models."""
import numpy as np
import pandas as pd


def getrow(df, names):
    if df is None or getattr(df, "empty", True):
        return None
    for n in names:
        if n in df.index:
            s = df.loc[n].dropna().astype(float)
            if len(s):
                return s
    return None


def forensic(inc, cf, bs):
    flags, m = [], {}
    g = lambda df, n: getrow(df, n)
    ni, rev = g(inc, ["Net Income"]), g(inc, ["Total Revenue", "Operating Revenue"])
    cfo, fcf, sbc = g(cf, ["Operating Cash Flow"]), g(cf, ["Free Cash Flow"]), g(cf, ["Stock Based Compensation"])
    ar, inv = g(bs, ["Accounts Receivable", "Receivables"]), g(bs, ["Inventory"])
    gw, eq, ta = g(bs, ["Goodwill"]), g(bs, ["Stockholders Equity", "Common Stock Equity"]), g(bs, ["Total Assets"])
    sh = g(bs, ["Ordinary Shares Number", "Share Issued"])
    gr = lambda s: s.iloc[0] / s.iloc[1] - 1 if s is not None and len(s) > 1 and s.iloc[1] > 0 else None
    if ni is not None and cfo is not None and ni.iloc[0] > 0:
        m["Cash conversion (CFO / net income)"] = cfo.iloc[0] / ni.iloc[0]
        if m["Cash conversion (CFO / net income)"] < .8:
            flags.append(("high", "Weak cash conversion", "Operating cash flow is well below reported earnings; investigate accruals."))
    if ni is not None and cfo is not None and ta is not None and ta.iloc[0] > 0:
        m["Accruals / assets"] = (ni.iloc[0] - cfo.iloc[0]) / ta.iloc[0]
        if m["Accruals / assets"] > .10:
            flags.append(("high", "High accruals", "Earnings rely heavily on non-cash items."))
    for nm, s in (("Receivables", ar), ("Inventory", inv)):
        if gr(s) is not None and gr(rev) is not None:
            m[f"{nm} growth minus revenue growth"] = gr(s) - gr(rev)
            if gr(s) - gr(rev) > .15:
                flags.append(("med", f"{nm} outpacing sales", "Can signal weak demand, channel stuffing or aggressive recognition (a reason to investigate, not proof)."))
    if sbc is not None and rev is not None:
        m["Stock comp / revenue"] = sbc.iloc[0] / rev.iloc[0]
        if fcf is not None:
            m["FCF after stock comp"] = fcf.iloc[0] - sbc.iloc[0]
        if m["Stock comp / revenue"] > .10:
            flags.append(("med", "Heavy stock-based compensation", "Reported free cash flow overstates what shareholders keep."))
    if gw is not None and eq is not None and eq.iloc[0] > 0:
        m["Goodwill / equity"] = gw.iloc[0] / eq.iloc[0]
        if m["Goodwill / equity"] > .5:
            flags.append(("med", "Goodwill is large vs equity", "Impairment risk if acquisitions underperform."))
    if sh is not None and len(sh) > 1:
        m["Share count change (YoY)"] = sh.iloc[0] / sh.iloc[1] - 1
        if m["Share count change (YoY)"] > .03:
            flags.append(("med", "Meaningful dilution", "Per-share value grows slower than the business."))
    return flags, m


def contradictions(c):
    """c: dict of optional numbers. Each rule fires only when every input it needs is present."""
    has = lambda *k: all(c.get(x) is not None for x in k)
    rules = [
        (has("rev_g", "margin_chg") and c["rev_g"] > .15 and c["margin_chg"] < -.02, "High revenue growth but operating margins are shrinking: growth may be bought, not earned."),
        (has("eps_g", "oi_g", "share_chg") and c["eps_g"] > .10 and c["oi_g"] < .03 and c["share_chg"] < -.02, "EPS growth is mostly buybacks; operating income is flat."),
        (has("cash_conv") and c["cash_conv"] < .7, "Reported earnings are not converting to cash."),
        (has("fwd_pe", "margin_peak") and c["fwd_pe"] < 12 and c["margin_peak"] and c["cyclical"], "Cheap forward P/E in a cyclical sector at peak margins: earnings may be unsustainable."),
        (has("ps", "gross_m") and c["ps"] < 2 and c["gross_m"] < .25, "Low P/S is not cheap when gross margins are structurally thin."),
        (has("fcf_yield", "rev_g") and c["fcf_yield"] > .06 and c["rev_g"] < 0, "High FCF yield alongside shrinking sales: the yield may be a melting ice cube."),
        (has("quality", "exp_ret", "hurdle") and c["quality"] > .4 and c["exp_ret"] < c["hurdle"], "Strong business, but the modeled return at today's price is below the required return."),
        (has("analyst", "rev_trend") and c["analyst"] > .5 and c["rev_trend"] < -.02, "Analysts are bullish while their own EPS estimates are being cut."),
    ]
    return [txt for ok, txt in rules if ok]


def validate(info, hist, price, now):
    w = []
    if hist is None or hist.empty:
        return ["No price history returned."]
    age = (now - hist.index[-1].tz_localize(None)).days if hist.index[-1].tzinfo else (now - hist.index[-1]).days
    if age > 5:
        w.append(f"Latest price bar is {age} days old (stale or market closed for a long period).")
    mc, sh = info.get("marketCap"), info.get("sharesOutstanding")
    if mc and sh and abs(mc / (price * sh) - 1) > .15:
        w.append("Market cap and shares x price disagree by >15% (multiple share classes or stale data).")
    if not info.get("sector"):
        w.append("No sector reported; generic thresholds used and industry logic limited.")
    return w


def _naive(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) is not None else pd.to_datetime(s.index)
    return s


def quant(c, spy, rf=.04):
    c = _naive(c.dropna())
    if len(c) < 250:
        return {}
    r = c.pct_change().dropna()
    cagr = (c.iloc[-1] / c.iloc[0]) ** (252 / len(c)) - 1
    vol = r.std() * np.sqrt(252)
    dd = r[r < 0].std() * np.sqrt(252)
    out = {"Annualized return (history)": cagr, "Volatility": vol, "Max drawdown": (c / c.cummax() - 1).min(),
           "Downside deviation": dd, "Sharpe": (cagr - rf) / vol, "Sortino": (cagr - rf) / dd if dd else np.nan}
    if len(c) > 253:
        out["Momentum 12-1 months"] = c.iloc[-22] / c.iloc[-253] - 1
    if spy is not None and len(spy) > 250:
        s = _naive(spy["Close"].dropna()).pct_change().dropna()
        j = pd.concat([r, s], axis=1, join="inner").dropna()
        j.columns = ["a", "m"]
        if len(j) > 200:
            beta = j.cov().loc["a", "m"] / j.cov().loc["m", "m"]
            out.update({"Beta vs SPY": beta, "Correlation vs SPY": j.corr().loc["a", "m"],
                        "Alpha (annual, vs SPY)": (j.a.mean() - beta * j.m.mean()) * 252,
                        "6-mo relative strength vs SPY": (1 + j.a.tail(126)).prod() / (1 + j.m.tail(126)).prod() - 1})
    return out


def net_debt(info, bs):
    """(net debt excluding minority interest, minority interest, note). For US filers operating-lease liabilities are
    excluded from debt because EBIT already bears the lease cost (including them double-counts the burden)."""
    debt = getrow(bs, ["Total Debt"])
    cash = getrow(bs, ["Cash Cash Equivalents And Short Term Investments"])
    lease, mi = getrow(bs, ["Capital Lease Obligations"]), getrow(bs, ["Minority Interest"])
    d = float(debt.iloc[0]) if debt is not None else float(info.get("totalDebt") or 0)
    note = "balance sheet" if debt is not None else "Yahoo info (may include leases)"
    if debt is not None and lease is not None and info.get("country") == "United States":
        d -= float(lease.iloc[0])
        note += ", US operating leases excluded"
    c = float(cash.iloc[0]) if cash is not None else float(info.get("totalCash") or 0)
    return d - c, (max(float(mi.iloc[0]), 0.0) if mi is not None else 0.0), note


def use_book_value(info, sector):
    """Book-value (justified P/B) valuation is for balance-sheet businesses only: banks, insurers, lenders, REITs.
    Asset-light financials (payment networks, exchanges, asset managers) are valued on cash flows."""
    ind = (info.get("industry") or "")
    if sector == "Real Estate" or ind.startswith(("Banks", "Insurance", "Mortgage", "Thrifts")):
        return True
    if ind == "Credit Services":
        return (info.get("totalDebt") or 0) > (info.get("marketCap") or 1e18)  # lenders carry debt > equity value
    return sector == "Financial Services" and not ind
