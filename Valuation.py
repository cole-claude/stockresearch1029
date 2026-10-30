"""Valuation from multiples, own history, peers and explicit scenarios.

Methods: (1) multiples chosen by what is meaningful for this company, (2) the company's own valuation history,
(3) peer comparison, (4) scenario outcomes built from explicit growth + exit-multiple + funding/dilution assumptions over a
3-year horizon, (5) "required growth": what growth the current price needs to earn the framework's required return.
Every assumption is exposed. Nothing is filled in when data is missing; the function says what is unavailable and why."""
import numpy as np
import pandas as pd
import research as R

H = 3  # scenario horizon in years
BASE_RATES = {"mature": (.25, .50, .25), "established_growth": (.25, .48, .27), "growth_profitable": (.25, .45, .30), "emerging_growth": (.25, .40, .35), "speculative": (.20, .35, .45),
              "turnaround": (.20, .40, .40), "cyclical": (.25, .45, .30), "semis_ai": (.25, .45, .30), "capital_intensive": (.25, .50, .25),
              "financial": (.25, .50, .25)}
clip = lambda x, a, b: float(np.clip(x, a, b))


def features(info, inc, cf, bs, est_growth=None):
    """Facts the framework needs, each from reported data. Anything not reported stays None."""
    rev, oi, ni = (R.getrow(inc, n) for n in (["Total Revenue", "Operating Revenue"], ["Operating Income"], ["Net Income"]))
    gp, fcf_s = R.getrow(inc, ["Gross Profit"]), R.getrow(cf, ["Free Cash Flow"])
    sh, sbc = R.getrow(bs, ["Ordinary Shares Number", "Share Issued"]), R.getrow(cf, ["Stock Based Compensation"])
    f = {"est_growth": est_growth}
    f["rev0"] = info.get("totalRevenue") or (float(rev.iloc[0]) if rev is not None else None)
    m0 = info.get("operatingMargins")
    if m0 is None and oi is not None and rev is not None:
        m0 = float(oi.iloc[0] / rev.iloc[0])
    f["m0"], f["net_margin"] = m0, info.get("profitMargins")
    f["gm"] = info.get("grossMargins") or (float(gp.iloc[0] / rev.iloc[0]) if gp is not None and rev is not None else None)
    f["fcf0"] = info.get("freeCashflow") if info.get("freeCashflow") is not None else (float(fcf_s.iloc[0]) if fcf_s is not None else None)
    nd, mi, _ = R.net_debt(info, bs)
    f["cash0"] = info.get("totalCash") if info.get("totalCash") is not None else None
    f["debt0"] = (nd + (f["cash0"] or 0)) + mi if f["cash0"] is not None or info.get("totalDebt") is not None else None
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    mc = info.get("marketCap") or 0
    f["shares"] = (mc / price) if price and mc else info.get("sharesOutstanding")
    f["mc"] = mc or (price * f["shares"] if price and f["shares"] else None)
    f["hist_g"] = float((rev.iloc[0] / rev.iloc[3]) ** (1 / 3) - 1) if rev is not None and len(rev) >= 4 and rev.iloc[3] > 0 else None
    gs = [x for x in (est_growth, info.get("revenueGrowth"), f["hist_g"]) if x is not None]
    f["g_raw"] = sum(gs) / len(gs) if gs else None
    f["g"] = clip(f["g_raw"], -.30, 1.50) if gs else None
    f["avg_m"] = float((oi / rev).dropna().mean()) if oi is not None and rev is not None else m0
    f["prior_profit_share"] = float((ni.iloc[1:] > 0).mean()) if ni is not None and len(ni) >= 3 else None
    f["runway"] = f["cash0"] / -f["fcf0"] if f["fcf0"] is not None and f["fcf0"] < 0 and f["cash0"] else None
    f["fcf_negative"] = f["fcf0"] is not None and f["fcf0"] < 0
    dil = None
    if sh is not None and len(sh) >= 2 and sh.iloc[-1] > 0:
        dil = clip((sh.iloc[0] / sh.iloc[-1]) ** (1 / (len(sh) - 1)) - 1, -.03, .12)
        f["dilution_source"] = f"share count history ({len(sh)} fiscal years)"
    elif sbc is not None and f["mc"]:
        dil = clip(float(sbc.iloc[0]) / f["mc"], 0, .10); f["dilution_source"] = "stock comp as % of market cap (proxy)"
    f["dilution"] = dil
    f["loss_making"] = bool((m0 is not None and m0 < 0) or (m0 is None and (f["net_margin"] or 0) < 0))
    f["eps_fwd"], f["eps_ttm"] = info.get("forwardEps"), info.get("trailingEps")
    f["eps_note"] = None
    if not f["eps_ttm"] and ni is not None and f["shares"] and f["shares"] > 0:   # reported net income / share count: a calculation, flagged as such
        f["eps_ttm"] = float(ni.iloc[0]) / f["shares"]
        f["eps_note"] = "calculated from latest fiscal-year net income and the current diluted share count (no reported EPS available)"
    capex, ocf = R.getrow(cf, ["Capital Expenditure"]), R.getrow(cf, ["Operating Cash Flow"])
    if capex is not None and f["rev0"]: f["capex_int"] = abs(float(capex.iloc[0])) / f["rev0"]
    elif ocf is not None and fcf_s is not None and f["rev0"]: f["capex_int"] = max(float(ocf.iloc[0] - fcf_s.iloc[0]), 0.0) / f["rev0"]
    else: f["capex_int"] = None
    om = (oi / rev).dropna() if oi is not None and rev is not None else None
    f["m_slope"] = float((om.iloc[0] - om.iloc[-1]) / (len(om) - 1)) if om is not None and len(om) >= 3 else None
    f["m_vol"] = float(om.std()) if om is not None and len(om) >= 3 else None
    gr = (rev.iloc[:-1].values / rev.iloc[1:].values - 1) if rev is not None and len(rev) >= 4 and (rev.iloc[1:] > 0).all() else None
    f["rev_vol"] = float(np.std(gr)) if gr is not None else None
    f["rev_years"] = len(rev) if rev is not None else 0
    f["book"], f["roe"] = info.get("bookValue"), info.get("returnOnEquity")
    f["payout"] = info.get("payoutRatio")
    return f


LINES = {"E-commerce and retail": ("online store", "e-commerce", "marketplace", "retail", " stores", "consumer products", "merchandise"),
         "Cloud and infrastructure": ("cloud", "data center", "infrastructure", "hosting", "compute"),
         "Software": ("software", "saas", "applications"),
         "Digital advertising": ("advertising", "advertisers"),
         "Subscriptions and memberships": ("subscription", "membership", "streaming"),
         "Semiconductors and hardware": ("semiconductor", "chips", "hardware", "devices", "gpu"),
         "Payments and financial networks": ("payment", "card", "transaction processing"),
         "Banking and lending": ("loans", "deposits", "lending"),
         "Insurance": ("insurance", "underwrit"),
         "Energy and utilities": ("oil", "natural gas", "electric", "utility", "power generation", "renewable"),
         "Healthcare and life sciences": ("drug", "clinical", "therapeutic", "medical device", "biotechnology")}
SECTOR_CYCL = {"Energy": "High", "Basic Materials": "High", "Industrials": "Moderate", "Consumer Cyclical": "Moderate", "Real Estate": "Moderate", "Financial Services": "Moderate"}
LEVELS = ["Low", "Moderate", "High"]


def profile(f, info, sector, industry):
    """Separate attributes: sector, business lines, maturity, growth, cyclicality, capital intensity, profitability, risk.
    The framework follows from the economics (profitability, growth, funding, business lines). A sector label is shown but never selects it by itself."""
    ind, g, gm = industry or "", f["g"], f["gm"]
    txt = ((info.get("longBusinessSummary") or "") + " " + ind).lower()
    lines = [k for k, kws in LINES.items() if any(w in txt for w in kws)]
    diversified = len(lines) >= 3
    # cyclicality: measured from the company's own history where possible, with the sector as a prior
    rv, mv = f.get("rev_vol"), f.get("m_vol")
    prior = "Moderate" if "Semiconductor" in ind else SECTOR_CYCL.get(sector, "Low")
    if rv is None and mv is None:
        level, cev = prior, [f"not measured (fewer than 4 fiscal years); sector prior only ({sector or 'unknown'})"]
    else:
        meas = "High" if (rv or 0) >= .08 or (mv or 0) >= .035 else "Moderate" if (rv or 0) >= .04 or (mv or 0) >= .015 else "Low"
        level = max(prior, meas, key=LEVELS.index) if sector in ("Energy", "Basic Materials") else meas
        cev = [f"revenue growth varies {rv:.1%} year to year" if rv is not None else "", f"operating margin varies {mv:.1%}" if mv is not None else ""]
        cev = [x for x in cev if x] + ([f"commodity-linked sector ({sector})"] if sector in ("Energy", "Basic Materials") else [])
    if diversified and level == "High":
        level = "Moderate"; cev.append("diversified business lines dampen exposure to any one cycle")
    ci = f.get("capex_int")
    ci_level = "unknown" if ci is None else "Very high" if ci >= .25 else "High" if ci >= .10 else "Moderate" if ci >= .05 else "Light"
    maturity = ("Early-stage" if f["loss_making"] and (f["rev0"] or 0) < 5e8 else "High growth" if g is not None and g >= .20 else "Established growth" if g is not None and g >= .08
                else "Mature" if g is not None and g >= 0 else "Declining" if g is not None else "Unknown")
    m0, sl = f["m0"], f.get("m_slope")
    profitability = ("Loss-making" + (" with improving margins" if sl is not None and sl > .02 else "")) if f["loss_making"] else "Marginally profitable" if m0 is not None and m0 < .03 else "Profitable" if m0 is not None else "Unknown"
    tags, why = [], []
    if diversified: tags.append("Diversified: " + ", ".join(lines[:4]) + (" and more" if len(lines) > 4 else ""))
    book = ind.startswith(("Banks", "Insurance", "Mortgage", "Thrifts")) or (R.use_book_value(info, sector) and sector != "Real Estate")
    if any(k in ind for k in ("Credit Services", "Financial Data", "Capital Markets", "Asset Management")): tags.append("Payments / asset-light financial")
    if any(k in ind for k in ("Semiconductor", "Computer Hardware", "Data Storage", "Information Technology Services", "Communication Equipment")) or (g is not None and g > .3 and sector == "Technology" and f["loss_making"]):
        tags.append("Semiconductor / AI infrastructure / technology hardware")
    if level != "Low" and (sector in SECTOR_CYCL or "Semiconductor" in ind): tags.append(f"Cyclical exposure ({level.lower()}): a risk to monitor, not a verdict")
    if ci_level in ("High", "Very high"): tags.append(f"Capital-intensive reinvestment (capex about {ci:.0%} of revenue)")
    if book:
        fk, tags, why = "financial", tags + ["Financial services"], ["Balance-sheet financial business: valued on book value and returns on equity."]
    elif f["loss_making"]:
        run = f["runway"]
        if g is not None and g >= .25 and gm is not None and gm >= .25 and (run is None or run >= 1.5):
            fk = "emerging_growth"
            why = [f"Loss-making but growing about {g:.0%} with a {gm:.0%} gross margin and funding runway {'not constrained' if run is None else f'of {run:.1f} years'}: judged on growth, unit economics and funding, not on current profit."]
        elif f["prior_profit_share"] and f["prior_profit_share"] >= .5 and (g is None or g < .10):
            fk, tags, why = "turnaround", tags + ["Turnaround / restructuring"], ["Previously profitable, now loss-making with weak growth."]
        else:
            bits = ([f"funding runway of only {run:.1f} years"] if g is not None and g >= .25 and gm is not None and gm >= .25 else []) + ([f"gross margin of only {gm:.0%}"] if gm is not None and gm < .25 else []) + (["growth below 25%"] if g is None or g < .25 else [])
            fk, tags, why = "speculative", tags + ["Funding-dependent"], ["Loss-making with: " + "; ".join(bits or ["limited evidence of unit economics"]) + "."]
    elif sector in ("Utilities", "Real Estate", "Energy", "Basic Materials"):
        fk, tags, why = "capital_intensive", tags + ["Capital-intensive / asset-heavy"], ["Asset-heavy or commodity-linked business: returns on capital, leverage and asset valuation matter."]
    elif "Semiconductor" in ind and not diversified:
        fk, why = "semis_ai", ["Profitable semiconductor / hardware business: cyclical earnings normalized, supply-demand and customer concentration matter."]
    elif sector in ("Industrials", "Consumer Cyclical") and not diversified and (g is None or g < .08) and level == "High":
        fk, why = "cyclical", ["Slow-growing business with measured, high cyclicality: judged on mid-cycle earnings rather than current margins."]
    elif g is not None and g >= .20:
        fk, why = "growth_profitable", [f"Profitable and growing about {g:.0%}: growth durability and what is priced in matter most."]
    elif g is not None and g >= .08:
        fk, why = "established_growth", [f"Profitable and growing about {g:.0%} while reinvesting: growth durability, returns on reinvestment and price paid matter most." + (" Business lines differ in economics; free data gives consolidated figures only." if diversified else "")]
    else:
        fk, why = "mature", ["Profitable business growing below 8% a year: earnings quality, cash conversion, returns on capital and price paid matter most."]
    risk = "Speculative" if fk == "speculative" else "High (funding-dependent)" if f["loss_making"] else "High" if (info.get("beta") or 0) >= 1.5 else "Moderate-high" if (info.get("beta") or 0) >= 1.1 else "Moderate or low"
    good, bad = [], []
    (good if len(info.get("longBusinessSummary") or "") > 200 else bad).append("company description" if len(info.get("longBusinessSummary") or "") > 200 else "no company description available")
    (good if g is not None else bad).append("growth evidence" if g is not None else "no growth evidence")
    (good if f.get("rev_years", 0) >= 3 else bad).append("3+ fiscal years of statements" if f.get("rev_years", 0) >= 3 else "fewer than 3 fiscal years of statements")
    if diversified: bad.append("several business lines with different economics; segment financials are not in free structured data")
    clabel = "High" if len(good) == 3 and not bad else "Medium" if len(good) >= 2 else "Low"
    note = (f"The sector label ({sector}) is shown for reference only; the framework was chosen from growth, profitability, funding and business lines."
            if sector in SECTOR_CYCL and fk not in ("cyclical", "capital_intensive", "financial") else None)
    return dict(framework=fk, tags=tags, why=why, sector=sector, industry=industry, business_lines=lines, diversified=diversified, maturity=maturity, profitability=profitability,
                cyclicality=dict(level=level, evidence=cev), capital_intensity=dict(level=ci_level, value=ci), risk_category=risk, sector_note=note,
                growth_profile=(f"{g:.0%} blended revenue growth" if g is not None else "unknown"), confidence=dict(label=clabel, good=good, bad=bad))


def classify(f, info, sector, industry):
    p = profile(f, info, sector, industry)
    return p["framework"], p["tags"], p["why"]


def pe_anchor(hist_stats, peer_df):
    """Company-specific support for the exit multiple: own fiscal-year P/E history and industry peers. None means sector rule of thumb only."""
    c = []   # (value, basis, description). History is trailing P/E; peers are forward P/E: scenarios convert them to a common basis.
    if hist_stats and "P/E" in hist_stats and hist_stats["P/E"]["n"] >= 3: c.append((hist_stats["P/E"]["median"], "trailing", f"the company's own fiscal-year-end P/E history (median of {hist_stats['P/E']['n']} years)"))
    if peer_df is not None:
        r = peer_df[peer_df.Metric == "Fwd P/E"]
        if len(r): c.append((float(r["Peer median"].iloc[0]), "forward", "the industry peer median forward P/E"))
    if not c: return None
    pull = .5 if (peer_df is not None and len(peer_df[peer_df.Metric == "Fwd P/E"])) or (hist_stats and "P/E" in hist_stats and hist_stats["P/E"]["n"] >= 6) else .25   # a few year-end points are noisy: pull less
    return dict(items=[(v, b) for v, b, _ in c], source=" and ".join(t for _, _, t in c), pull=pull)


# ---------------------------------------------------------------- multiples and their applicability
PRIMARY = {"mature": ("Forward P/E", "EV / EBITDA", "FCF yield"), "established_growth": ("Forward P/E", "EV / EBITDA", "EV / Sales"), "growth_profitable": ("Forward P/E", "EV / Sales", "PEG"),
           "emerging_growth": ("EV / Sales", "EV / Gross profit"), "speculative": ("EV / Sales", "EV / Gross profit"),
           "turnaround": ("EV / Sales", "Price / Book"), "cyclical": ("EV / EBITDA", "Price / Book", "Forward P/E"),
           "semis_ai": ("Forward P/E", "EV / EBITDA", "EV / Sales"), "capital_intensive": ("EV / EBITDA", "Price / Book", "FCF yield"),
           "financial": ("Price / Book", "Forward P/E")}


def multiples_table(info, f, framework, price):
    """Every multiple with its status: Meaningful / Not meaningful (and why) / Unavailable (and why)."""
    mc = f["mc"]; ev = mc + (f["debt0"] or 0) - (f["cash0"] or 0) if mc and f["debt0"] is not None and f["cash0"] is not None else info.get("enterpriseValue")
    rows, prim = [], PRIMARY.get(framework, ())

    def add(name, val, ok, why_not, note=""):
        status = "Meaningful" if ok else ("Unavailable" if val is None and not why_not.startswith("negative") else "Not meaningful")
        rows.append({"Multiple": name, "Value": val if ok else None, "Status": status, "Why": note if ok else why_not,
                     "Relevance": "Primary" if name in prim else "Context"})
    teps, feps = f["eps_ttm"], f["eps_fwd"]
    pe_t, pe_f = info.get("trailingPE"), info.get("forwardPE")
    add("Trailing P/E", pe_t, bool(pe_t and pe_t > 0 and (teps is None or teps > 0)), "negative earnings" if (teps is not None and teps <= 0) or (pe_t is not None and pe_t <= 0) else "not reported")
    add("Forward P/E", pe_f, bool(pe_f and pe_f > 0 and (feps is None or feps > 0)), "negative forward earnings" if (feps is not None and feps <= 0) else "no consensus forward EPS from free data")
    peg = info.get("pegRatio")
    add("PEG", peg, bool(peg and peg > 0), "not reported or growth/earnings not positive")
    ebitda = info.get("ebitda")
    eve = (ev / ebitda) if ev and ebitda and ebitda > 0 else None
    add("EV / EBITDA", eve, eve is not None, "negative EBITDA" if ebitda is not None and ebitda <= 0 else "EBITDA or EV not reported", "EBITDA ignores capex and stock comp")
    evs = ev / f["rev0"] if ev and f["rev0"] else None
    add("EV / Sales", evs, evs is not None and evs > 0, "revenue or EV unavailable")
    ps = mc / f["rev0"] if mc and f["rev0"] else None
    add("Price / Sales", ps, ps is not None, "revenue or market cap unavailable")
    evg = ev / (f["rev0"] * f["gm"]) if ev and f["rev0"] and f["gm"] and f["gm"] > 0 else None
    add("EV / Gross profit", evg, evg is not None and evg > 0, "gross margin unavailable")
    pb = info.get("priceToBook") or (price / f["book"] if price and f["book"] else None)
    add("Price / Book", pb, bool(pb and pb > 0), "book value unavailable or negative", "most relevant for financials and asset-heavy firms")
    fy = f["fcf0"] / mc if f["fcf0"] is not None and mc else None
    add("FCF yield", fy, fy is not None and fy > 0, "negative free cash flow (cash burn)" if f["fcf0"] is not None and f["fcf0"] <= 0 else "free cash flow unavailable")
    return rows, ev


def own_history(info, hist, inc, bs, price):
    """The company's own fiscal-year-end valuation multiples (price at FY end vs reported results)."""
    rev, ni = R.getrow(inc, ["Total Revenue", "Operating Revenue"]), R.getrow(inc, ["Net Income"])
    sh = R.getrow(bs, ["Ordinary Shares Number", "Share Issued"])
    if hist is None or hist.empty or rev is None or sh is None:
        return None, None
    c = R._naive(hist["Close"].dropna())
    rows = []
    for col in rev.index:
        d = pd.to_datetime(col)
        if d not in sh.index and col not in sh.index:
            continue
        p = c.asof(d)
        shares = sh.get(col, sh.get(d))
        if pd.isna(p) or not shares or shares <= 0:
            continue
        eps = float(ni[col]) / shares if ni is not None and col in ni.index else None
        rows.append({"FY end": d.strftime("%Y-%m-%d"), "Price": float(p), "P/E": float(p) / eps if eps and eps > 0 else None, "P/S": float(p) * shares / float(rev[col])})
    if len(rows) < 3:
        return None, None
    df = pd.DataFrame(rows).sort_values("FY end", ascending=False).reset_index(drop=True)
    stats = {}
    cur = {"P/E": info.get("trailingPE"), "P/S": info.get("priceToSalesTrailing12Months")}
    for k, v in cur.items():
        s = df[k].dropna()
        if len(s) >= 3 and v and v > 0:
            stats[k] = dict(current=float(v), low=float(s.min()), median=float(s.median()), high=float(s.max()), pct=float((s < v).mean()), n=len(s))
    return df, stats


def peer_compare(info, f, peers):
    if peers is None or len(peers) < 2:
        return None
    rows = []
    for lab, own in (("Fwd P/E", info.get("forwardPE")), ("EV/EBITDA", info.get("enterpriseToEbitda")), ("P/S", info.get("priceToSalesTrailing12Months")),
                     ("Op margin", f["m0"]), ("Rev growth", info.get("revenueGrowth"))):
        med = pd.to_numeric(peers[lab], errors="coerce").median() if lab in peers else None
        if own is not None and med is not None and np.isfinite(med) and med != 0:
            rows.append({"Metric": lab, "Company": own, "Peer median": float(med), "Premium": (own / med - 1) if lab not in ("Op margin", "Rev growth") else own - med})
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------- scenarios (explicit assumptions)
def _probs(framework, q):
    b, m, r = BASE_RATES[framework]
    pb, pr = clip(b + .06 * q, .10, .55), clip(r - .06 * q, .10, .55)
    return {"Bull": pb, "Base": 1 - pb - pr, "Bear": pr}


def scenarios(f, info, price, framework, pe_rng, q, g_override=None, anchor=None):
    out = dict(ok=False, mode=None, rows=None, reason="", limitations=[], missing=[], framework=framework)
    sh0, g = f["shares"], (g_override if g_override is not None else f["g"])
    if not price or not sh0:
        out["reason"] = "price or share count unavailable"; out["missing"] = ["share price", "shares outstanding"]; return out
    if g is None:
        out["reason"] = "no growth evidence (no consensus, no TTM growth, fewer than 4 fiscal years)"; out["missing"] = ["revenue growth"]; return out
    pr = _probs(framework, q)
    d0 = f["dilution"]
    if d0 is None:
        d0 = 0.0; out["limitations"].append("Dilution could not be measured (no share-count history or stock-comp data): assumed 0%, which flatters per-share outcomes.")
    out["dilution_source"] = f.get("dilution_source", "unknown")
    mc, rows = f["mc"], {}
    if framework == "financial" or (framework == "capital_intensive" and f["book"] and f["roe"] is not None and info.get("sector") == "Real Estate"):
        if not f["book"] or f["roe"] is None:
            out["reason"] = "book value or return on equity unavailable"; out["missing"] = ["book value per share", "return on equity"]; return out
        out["mode"] = "book"
        bv, roe, pay = float(f["book"]), clip(f["roe"], 0, .30), clip(f["payout"] if f["payout"] is not None else .35, 0, .8)
        cur = price / bv; mid = 1.2
        base_pb = clip((cur * mid) ** .5, .6, 4.0)
        for k, (dr, mult) in {"Bull": (.02, 1.15), "Base": (0, 1.0), "Bear": (-.03, .70)}.items():
            r_ = max(roe + dr, 0.0)
            bvh = bv * (1 + r_ * (1 - pay)) ** H
            divs = bv * r_ * pay * H
            ph = bvh * base_pb * mult + divs
            rows[k] = dict(assump={"ROE": r_, "Book value per share at horizon": bvh, "Exit P/B": base_pb * mult, "Dividends received (approx.)": divs}, price_h=ph)
        out["limitations"] += ["Exit P/B is anchored between today's P/B and an assumed 1.2x; ROE is held near its current level.", "Credit losses and rate changes are not modeled explicitly."]
    elif f["loss_making"] or (not f["eps_fwd"] and not f["eps_ttm"] and f["rev0"] and f["m0"] is not None and f["m0"] < 0):
        if not f["rev0"]:
            out["reason"] = "revenue unavailable"; out["missing"] = ["revenue"]; return out
        out["mode"] = "revenue"
        rev0, cash0, debt0 = f["rev0"], f["cash0"], f["debt0"]
        burn0 = max(-f["fcf0"], 0.0) if f["fcf0"] is not None else None
        if burn0 is None or cash0 is None or debt0 is None:
            out["limitations"].append("Cash burn, cash or debt unavailable: the funding requirement could not be modeled, so dilution is understated.")
        burn0, cash0, debt0 = burn0 or 0.0, cash0 or 0.0, debt0 or 0.0
        cur_evs = (price * sh0 + debt0 - cash0) / rev0
        cur_evs = cur_evs if cur_evs > 0 else price * sh0 / rev0
        g0 = clip(g, -.30, 1.5)
        base_revs = None   # committed capacity: a weaker case never spends less than the base plan
        floor = min(1.0, .5 * cur_evs)   # a 1.0x floor would inflate businesses that trade below 1x sales
        spec = {"Bull": dict(g=clip(g0 * 1.25 + .05, -.1, 2.0) if g0 >= 0 else clip(g0 * .5 + .03, -.15, .1), fade=.90, comp=.85, tb=3, disc=.05, dil=.7),
                "Base": dict(g=g0, fade=.80, comp=.55, tb=5, disc=.10, dil=1.0),
                "Bear": dict(g=g0 * .5 if g0 >= 0 else g0 * 1.5 - .02, fade=.70, comp=.30, tb=None, disc=.30, dil=1.3)}
        for k, s in spec.items():
            fade = s["fade"] - (.15 if s["g"] >= .5 else 0)
            rev, cum, revs = rev0, 0.0, []
            br0 = burn0 / rev0
            for t in range(1, H + 1):
                rev *= 1 + s["g"] * fade ** (t - 1)
                br = br0 * (max(0.0, 1 - t / s["tb"]) if s["tb"] else 1.0)
                cum += (max(rev, base_revs[t - 1]) if k == "Bear" and base_revs else rev) * br; revs.append(rev)
            if k == "Base": base_revs = revs
            buf = .15 * rev
            raise_ = max(cum + buf - cash0, 0.0)
            debt_new, eq = raise_ * .35, raise_ * .65
            issue_px = price * (1 - s["disc"])
            sh_h = sh0 * (1 + d0 * s["dil"]) ** H + eq / issue_px
            cash_h = buf if raise_ > 0 else cash0 - cum
            comp = s["comp"] if cur_evs >= 4 else 1 - (1 - s["comp"]) * cur_evs / 4   # compression only applies to rich growth multiples
            ev_h = rev * max(cur_evs * comp, floor)
            equity = ev_h - (debt0 + debt_new) + cash_h
            rows[k] = dict(assump={"Revenue growth, year 1": s["g"], "Revenue at horizon": rev, "Exit EV/Sales": max(cur_evs * comp, floor),
                                   "Cumulative cash burn": cum, "Capital to raise": raise_, "Share count change": sh_h / sh0 - 1, "Years to break-even (assumed)": s["tb"] or "none in horizon"},
                           price_h=max(equity, 0.0) / sh_h)
        out["limitations"] += ["Exit EV/Sales is an assumption (today's multiple compressed 15% / 45% / 70% in bull / base / bear as growth normalizes; scaled down for businesses already trading below 4x sales), not market data.",
                               "Burn per revenue dollar is assumed to improve linearly to break-even; 35% of new capital is assumed to be debt, 65% equity.",
                               "New equity is assumed to be issued at today's price less a discount; a lower share price would mean more dilution.",
                               "Capacity spending is treated as committed: the bear case does not spend less than the base plan even though revenue is lower."]
    else:
        eps1, src = f["eps_fwd"], "consensus forward EPS"
        ge = clip(g, -.10, .30)
        if not eps1 or eps1 <= 0:
            if f["eps_ttm"] and f["eps_ttm"] > 0:
                eps1, src = f["eps_ttm"] * (1 + ge), "trailing EPS grown at the growth estimate (no consensus available)" + (f"; trailing EPS {f['eps_note']}" if f.get("eps_note") else "")
            else:
                out["reason"] = "no positive forward or trailing EPS"; out["missing"] = ["EPS"]; return out
        out["mode"] = "earnings"
        if framework in ("cyclical", "semis_ai", "capital_intensive") and f["m0"] and f["avg_m"] and f["m0"] > f["avg_m"] * 1.15:
            eps1 *= f["avg_m"] / f["m0"]; src += f"; normalized for above-average margins ({f['m0']:.0%} vs {f['avg_m']:.0%} average)"
        lo, hi = pe_rng
        cur_pe = price / eps1
        vals = [v * f["eps_ttm"] / eps1 if b == "trailing" else v for v, b in (anchor["items"] if anchor else []) if b != "trailing" or (f.get("eps_ttm") and eps1)]
        anch, asrc = (float(np.median(vals)), anchor["source"]) if vals else ((lo + hi) / 2, "the sector rule of thumb only (no company-specific multiple history or peer data), so weakly supported")
        pull = anchor.get("pull", .5) if anchor else .5
        base_pe = max(cur_pe + pull * (anch - cur_pe), .75 * cur_pe) if anch < cur_pe else cur_pe   # a premium multiple compresses toward its anchor (halfway with rich evidence, a quarter with sparse history; at most 25%); it never expands in the base case
        mg = clip(.5 * f["m_slope"] / f["m0"], -.04, .05) if f.get("m_slope") is not None and f["m0"] and f["m0"] > 0 else 0.0
        spec = {"Bull": dict(g=ge + max(.03, .25 * abs(ge)) + max(2 * mg, 0.0), fade=.90, pe=cur_pe if anch < cur_pe else min(cur_pe * 1.10, anch), dil=.7),
                "Base": dict(g=ge + mg, fade=.85, pe=base_pe, dil=1.0),
                "Bear": dict(g=max(ge - max(.04, .6 * abs(ge)), -.25), fade=.80, pe=base_pe * .70, dil=1.3)}
        for k, s in spec.items():
            eps = eps1
            for t in range(1, H + 1):
                eps *= 1 + s["g"] * s["fade"] ** t
            ph = eps * s["pe"] / (1 + d0 * s["dil"]) ** H
            rows[k] = dict(assump={"EPS growth, year 2": s["g"] * s["fade"], "Forward EPS at horizon": eps / (1 + d0 * s["dil"]) ** H, "Exit forward P/E": s["pe"], "Annual dilution": d0 * s["dil"]}, price_h=ph)
        out["anchor"] = dict(pe=anch, source=asrc, current=cur_pe, supported=bool(vals))
        out["eps_source"] = src
        out["limitations"] += [f"EPS base: {src}.", f"Exit P/E: today's {cur_pe:.0f}x moves {'halfway' if pull >= .5 else 'a quarter of the way'} toward {anch:.0f}x if that is lower (anchor: {asrc}); the base case never assumes the multiple expands. Dividends are excluded.",
                               f"EPS growth is blended revenue growth plus half the recent operating-margin change per year ({mg:+.1%} a year in the base case); consensus earnings growth is not available from free data."]
    for k in rows:
        rows[k]["prob"] = pr[k]
        rows[k]["ann"] = (rows[k]["price_h"] / price) ** (1 / H) - 1 if rows[k]["price_h"] > 0 else -1.0
        rows[k]["total"] = rows[k]["price_h"] / price - 1
    ev_p = sum(rows[k]["prob"] * rows[k]["price_h"] for k in rows)
    out.update(ok=True, rows=rows, ev_price=ev_p, ev_ann=(ev_p / price) ** (1 / H) - 1 if ev_p > 0 else -1.0, horizon=H,
               probs_note=(f"Probabilities are ESTIMATES: base rates assumed for a {framework.replace('_', ' ')} business "
                           f"({BASE_RATES[framework][0]:.0%} / {BASE_RATES[framework][1]:.0%} / {BASE_RATES[framework][2]:.0%}), tilted up to ±6 points by evidence of quality and funding."))
    return out


def required_growth(f, info, price, framework, pe_rng, q, hurdle, hist_g, anchor=None):
    """What year-1 growth makes the BASE scenario earn the framework's required return?"""
    def base_ann(g):
        s = scenarios(f, info, price, framework, pe_rng, q, g_override=g, anchor=anchor)
        return s["rows"]["Base"]["ann"] if s["ok"] else None
    lo, hi = -.30, 1.5
    a, b = base_ann(lo), base_ann(hi)
    if a is None or b is None or (a - hurdle) * (b - hurdle) > 0:
        return None, "n/a"
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if (base_ann(lo) - hurdle) * (base_ann(mid) - hurdle) <= 0 else (mid, hi)
    g = (lo + hi) / 2
    if hist_g is not None and hist_g > .02:
        r = g / hist_g
        lab = "Conservative" if r <= .5 else "Reasonable" if r <= 1.0 else "Aggressive" if r <= 1.5 else "Implausible"
    else:
        lab = "Conservative" if g <= .05 else "Reasonable" if g <= .12 else "Aggressive" if g <= .25 else "Implausible"
    return g, lab
