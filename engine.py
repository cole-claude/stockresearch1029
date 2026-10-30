"""One pipeline: raw data in -> one result out. The interface and the tests both call analyze(); no page computes its own numbers or labels."""
import re
import numpy as np
import pandas as pd
import research as R, metrics as M, valuation as V, rating as RT, qual as Q
from catalog import CATALOG

clip = lambda x, a=-1.0, b=1.0: float(np.clip(x, a, b))
PLAUS = {"Conservative": .8, "Reasonable": .3, "Aggressive": -.5, "Implausible": -1.0}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _est(df, r, c):
    try:
        return float(df.loc[r, c])
    except Exception:
        return None


def _fmt(x):
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:.3g}"


_PCT = ("margin", "growth", "return", "yield", "roe", "roa", "drawdown", "volatility", "short interest", "ownership", "cagr", "change", "upside", "price vs", "momentum", "relative", "share of")
_MULT = ("p/e", "ev /", "price /", "peg")


def _fmtm(metric, x):
    """Metric-aware number: percentages as %, multiples as x, everything else compact."""
    m = metric.lower()
    if any(k in m for k in _MULT) and "growth" not in m: return f"{x:.1f}x"
    if any(k in m for k in _PCT) and abs(x) < 20: return f"{x * 100:.0f}%" if abs(x) >= .1 else f"{x * 100:.1f}%"
    return _fmt(x)


def _usd(x):
    a, sg = abs(x), "-" if x < 0 else ""
    return f"{sg}${a / 1e12:,.2f}T" if a >= 1e12 else f"{sg}${a / 1e9:,.1f}B" if a >= 1e9 else f"{sg}${a / 1e6:,.0f}M" if a >= 1e6 else f"{sg}${a:,.0f}"


def _p0(x):
    return "n/a" if x is None else format(x, ".0%")


def _top(df, cats, n, best=True):
    d = df[df.Category.isin(cats)].sort_values("Score", ascending=not best).head(n)
    return [f"{r.Metric} {_fmtm(r.Metric, r.Value)}" for r in d.itertuples() if (r.Score > .2 if best else r.Score < -.2)]


def _sentences(text, n=2, cap=430):
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = " ".join(parts[:n])
    return out[:cap].rstrip() + ("..." if len(out) > cap else "")


def qualitative_scenarios(c):
    """Words, not numbers: used when a numerical scenario cannot be calculated reliably. Built only from reported facts and the profile."""
    f, P = c["f"], c["profile"]
    g, m0, lines = f["g"], f["m0"], ", ".join(P["business_lines"][:3])
    now = f"growth of about {g:.0%}" if g is not None else "its recent growth"
    bull = (f"Demand stays strong, so {now} is sustained or exceeded and margins" + (f" improve from {m0:.0%}" if m0 is not None else " improve") + " as the business scales; the market keeps paying today's multiple."
            + (f" The drivers would be {lines}." if lines else ""))
    base = f"{now.capitalize()} fades gradually toward a mature rate, margins" + (f" hold near {m0:.0%}" if m0 is not None else " hold near current levels") + ", and the valuation compresses modestly as growth slows."
    bear = ("Growth slows sharply, margins compress under competition or cost pressure, and the market pays a lower multiple"
            + ("; with losses and limited cash, new financing may be needed on poor terms, diluting shareholders." if c["loss_making"] else "; reinvestment needs may rise before returns arrive." if P["capital_intensity"]["level"] in ("High", "Very high") else "."))
    return dict(Bull=bull, Base=base, Bear=bear)


def _brief(c):
    """Structured summary. Every item carries its evidence class: Reported / Calculated / Forecast (assumption-based) / Judgment."""
    f, info, s, P, rt = c["f"], c["info"], c["scen"], c["profile"], c["rating"]
    B, price = {}, c["price"]
    desc = _sentences(info.get("longBusinessSummary"))
    B["business"] = [(desc, "Reported: company description")] if desc and desc != "Synthetic test company." else []
    if P["business_lines"]:
        B["business"].append(("Business lines detected from that description: " + ", ".join(P["business_lines"]) + "." + (" Their economics differ, but free structured data gives consolidated figures only, so this analysis is company-wide; segment economics are not verified." if P["diversified"] else ""), "Derived from the description, not segment data"))
    if not B["business"]:
        B["business"] = [("No company description was returned, so what the business does cannot be summarised from data.", "Unavailable")]
    facts = []
    if f["rev0"]: facts.append((f"Revenue {_usd(f['rev0'])}; blended growth {_p0(f['g'])} (analyst estimate {_p0(f['est_growth'])}, reported TTM {_p0(info.get('revenueGrowth'))}, 3-year history {_p0(f['hist_g'])}).", "Reported / calculated"))
    if f["gm"] is not None or f["m0"] is not None:
        facts.append((f"Gross margin {_p0(f['gm'])}; operating margin {_p0(f['m0'])}" + (f"; moving about {f['m_slope'] * 100:+.1f} points a year over the available history" if f.get("m_slope") is not None else "") + ".", "Reported / calculated"))
    if f["fcf0"] is not None: facts.append((f"Free cash flow {_usd(f['fcf0'])}" + (f"; capital spending about {f['capex_int']:.0%} of revenue" if f.get("capex_int") is not None else "") + ".", "Reported / calculated"))
    if f["debt0"] is not None and f["cash0"] is not None:
        nd = f["debt0"] - f["cash0"]
        facts.append((("Net cash of " + _usd(-nd) if nd < 0 else "Net debt of " + _usd(nd)) + (f"; cash covers about {f['runway']:.1f} years of burn" if f["runway"] else "") + ".", "Reported"))
    B["facts"] = facts
    val = []
    prim = [f"{r['Multiple']} {r['Value']:.1f}x" if r["Multiple"] != "FCF yield" else f"FCF yield {r['Value']:.1%}" for r in c["mult_rows"] if r["Status"] == "Meaningful" and r["Relevance"] == "Primary"]
    if prim: val.append((f"Multiples that matter for this framework: {', '.join(prim)}.", "Reported / calculated"))
    nm = [f"{r['Multiple']} ({r['Why']})" for r in c["mult_rows"] if r["Status"] == "Not meaningful"]
    if nm: val.append(("Not meaningful here: " + "; ".join(nm) + ".", "Judgment: shown as unavailable instead of a misleading number"))
    for k, v in (c["hist_stats"] or {}).items():
        val.append((f"{k} is at the {v['pct']:.0%} percentile of its own fiscal-year-end range ({v['low']:.1f}x to {v['high']:.1f}x).", "Calculated"))
    if c["reqg"] is not None: val.append((f"At today's price, the base case needs year-1 growth of about {c['reqg']:.0%} to earn the {c['hurdle']:.0%} required return; that looks {c['plaus'].lower()}.", "Calculated from assumptions"))
    B["valuation"] = val
    fc = []
    if s["ok"]:
        b, u, r_ = (s["rows"][k] for k in ("Base", "Bull", "Bear"))
        fc.append((f"Over {V.H} years the base case returns {b['ann']:+.1%} a year, the bull case {u['ann']:+.1%} and the bear case {r_['total']:+.0%} in total; the probability-weighted figure is {s['ev_ann']:+.1%} a year. Probabilities are estimates.", "Forecast: depends on the assumptions listed on the Scenarios page"))
        if s.get("anchor"): fc.append((f"The base-case exit multiple is supported by {s['anchor']['source']}." if s["anchor"]["supported"] else "The exit multiple rests on the sector rule of thumb only, so this forecast is weakly supported.", "Forecast assumption"))
        a = b["assump"]
        if s["mode"] == "earnings": succ = f"Earnings per share growing about {a['EPS growth, year 2']:.0%} a year and the multiple holding near {a['Exit forward P/E']:.0f}x."; inval = f"Growth nearer {r_['assump']['EPS growth, year 2']:.0%} a year, or the multiple falling toward {r_['assump']['Exit forward P/E']:.0f}x."
        elif s["mode"] == "revenue": succ = f"Revenue reaching {_usd(a['Revenue at horizon'])} within {V.H} years with funding that does not dilute shareholders heavily (base-case raise {_usd(a['Capital to raise'])})."; inval = f"Growth stalling toward {r_['assump']['Revenue growth, year 1']:.0%}, burn persisting, and a financing on poor terms (bear-case raise {_usd(r_['assump']['Capital to raise'])})."
        else: succ = f"Return on equity staying near {a['ROE']:.0%} and the price-to-book ratio holding near {a['Exit P/B']:.1f}x."; inval = f"Return on equity falling toward {r_['assump']['ROE']:.0%} or the price-to-book ratio de-rating to {r_['assump']['Exit P/B']:.1f}x."
        B["success"], B["invalidate"] = [(succ, "Forecast")], [(inval, "Forecast")]
    else:
        B["success"], B["invalidate"] = [("A numerical scenario could not be calculated reliably: " + (s["reason"] or "insufficient data") + ".", "Unavailable")], []
    B["invalidate"] += [(f"{x[1]}: {x[2]}", "Calculated flag") for x in c["flags"][:2]]
    unc = ["Scenario probabilities are estimates, not statistics.", "The growth and margin paths are mechanical extrapolations of reported history, not forecasts from the company."]
    if s["ok"] and s.get("anchor") and not s["anchor"]["supported"]: unc.append("The exit multiple has no company-specific history or peer support.")
    if P["diversified"]: unc.append("Segment-level economics are not available from free data.")
    if P["confidence"]["label"] != "High": unc.append("Classification confidence is " + P["confidence"]["label"].lower() + ": " + "; ".join(P["confidence"]["bad"] or ["limited evidence"]) + ".")
    if not any(v is not None for v in c["est"].values()): unc.append("No consensus estimates were available.")
    B["uncertain"] = [(u_, "Judgment") for u_ in unc]
    return B


def analyze(info, hist, inc, cf, bs, est=None, peers=None, ai=None, w_qual=.5, apply_override=False, now=None):
    est = est or {}
    sector, industry = info.get("sector"), info.get("industry")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(hist["Close"].iloc[-1]) if hist is not None and len(hist) else None)
    eg = _est(est.get("rev"), "+1y", "growth") if est.get("rev") is not None else None
    f = V.features(info, inc, cf, bs, eg)
    P = V.profile(f, info, sector, industry)
    framework, tags, why = P["framework"], P["tags"], P["why"]
    fw = RT.FRAMEWORKS[framework]
    prof = M.profile_for(info, sector)
    df = M.build_metrics(info, hist, inc, cf, bs, prof, loss_making=f["loss_making"])
    na = dict(df.attrs.get("na", {}))
    cs = df.groupby("Category")["Score"].mean().to_dict() if len(df) else {}
    quality, health = _mean([cs.get("Profitability"), cs.get("Track record")]), cs.get("Financial health")
    q = _mean([quality, health]) or 0.0
    hurdle = fw["hurdle"]

    hist_df, hist_stats = V.own_history(info, hist, inc, bs, price)
    peer_df = V.peer_compare(info, f, peers)
    anchor = V.pe_anchor(hist_stats, peer_df)
    scen = V.scenarios(f, info, price, framework, prof["exit"], q, anchor=anchor)
    reqg, plaus = V.required_growth(f, info, price, framework, prof["exit"], q, hurdle, f["hist_g"], anchor=anchor) if scen["ok"] else (None, "n/a")
    mult_rows, ev = V.multiples_table(info, f, framework, price)

    # ---- dimension scores (-1..+1, None = cannot be assessed)
    vitems = []
    if cs.get("Valuation") is not None:
        vitems.append(("Multiples vs sector rules of thumb", cs["Valuation"]))
    key = "P/S" if f["loss_making"] else "P/E"
    if hist_stats and key in hist_stats:
        vitems.append((f"{key} vs own fiscal-year history ({hist_stats[key]['pct']:.0%} percentile)", clip(1 - 2 * hist_stats[key]["pct"])))
    if peer_df is not None:
        prem = peer_df[peer_df.Metric.isin(["Fwd P/E", "EV/EBITDA", "P/S"])]["Premium"]
        if len(prem):
            vitems.append((f"Premium to peer median ({prem.mean():+.0%})", clip(-prem.mean())))
    if plaus != "n/a":
        vitems.append((f"Required growth looks {plaus.lower()}", PLAUS[plaus]))
    rr = None
    if scen["ok"]:
        ev_ann, base_ann, bear_loss = scen["ev_ann"], scen["rows"]["Base"]["ann"], max(0.0, -scen["rows"]["Bear"]["total"])
        rr = clip(.5 * clip((ev_ann - hurdle) / .15) + .3 * clip((base_ann - hurdle) / .15) + .2 * clip(1 - 2 * bear_loss / .6))
    dims = {"Business quality": quality, "Growth": cs.get("Growth"), "Financial health": health, "Valuation": _mean([s for _, s in vitems]),
            "Risk": cs.get("Risk"), "Risk-reward": rr}

    # ---- qualitative view (computed items always; AI items only fill what the data cannot assess)
    qitems = Q.assess_data(inc, cf, bs, info, dict(plaus=plaus, impl=reqg) if reqg is not None else None, loss_making=f["loss_making"], runway=f["runway"])
    if ai:
        for k, it in ai.get("items", {}).items():
            if qitems[k]["score"] is None:
                qitems[k] = it
    qscore, qcov = Q.summarize(qitems)
    override = ai["challenge"] if ai and ai.get("challenge") and apply_override else None

    # ---- coverage, confidence, rating
    have = set(df["Metric"]) if len(df) else set()
    labels = [m for _, m in CATALOG]
    unavailable = [m for m in labels if m not in have and m not in na]
    cov = dict(total=len(CATALOG), evaluated=[m for m in labels if m in have], not_meaningful={m: r for m, r in na.items() if m in labels}, unavailable=unavailable)
    cov_q = len(cov["evaluated"]) / max(len(CATALOG) - len(cov["not_meaningful"]), 1)
    have_est = any(v is not None for v in est.values())
    conf, conf_label, conf_factors = RT.confidence(framework, cov_q, qcov, scen, anchor, P["confidence"]["label"], have_est, hist_df is not None, peer_df is not None)
    raise_ratio = None
    if scen["ok"] and scen["mode"] == "revenue" and f["mc"]:
        raise_ratio = scen["rows"]["Base"]["assump"]["Capital to raise"] / f["mc"]
    rt = RT.rate(framework, dims, scen, conf, dict(score=qscore, coverage=qcov), w_qual, f["runway"], raise_ratio, override)
    trend = est.get("trend")
    rev_rev = None
    try:
        rev_rev = float(trend.loc["+1y", "current"] / trend.loc["+1y", "90daysAgo"] - 1)
    except Exception:
        pass
    nt_score, nt_label = RT.near_term_setup(cs, rev_rev)

    # ---- flags and contradictions
    flags, fm = R.forensic(inc, cf, bs)
    mv = lambda n: (lambda r: float(r.iloc[0]) if len(r) else None)(df.loc[df.Metric == n, "Value"]) if len(df) else None
    cons = R.contradictions(dict(rev_g=info.get("revenueGrowth"), margin_chg=mv("Operating margin change (YoY)"), eps_g=info.get("earningsGrowth"),
                                 oi_g=mv("Operating income growth"), share_chg=fm.get("Share count change (YoY)"), cash_conv=fm.get("Cash conversion (CFO / net income)"),
                                 fwd_pe=info.get("forwardPE"), cyclical=any(t.startswith("Cyclical") for t in tags), ps=info.get("priceToSalesTrailing12Months"),
                                 margin_peak=bool(f["m0"] and f["avg_m"] and f["m0"] > f["avg_m"] * 1.15), gross_m=f["gm"], fcf_yield=mv("Free cash flow yield"),
                                 quality=quality, exp_ret=scen.get("ev_ann") if scen["ok"] else None, hurdle=hurdle, analyst=cs.get("Analyst"), rev_trend=rev_rev))

    # ---- deterministic narrative pieces (every number below is computed above)
    name = info.get("shortName") or info.get("longName") or ""
    mc = f["mc"]
    th = [f"{name} is analysed as a {fw['label'].lower()}" + (f" ({'; '.join(tags)})" if tags else "") + f". {why[0]}" + (f" {P['sector_note']}" if P["sector_note"] else "")]
    if mc:
        th.append(f"At ${price:,.2f} the market values it at {_usd(mc)}" + (f" (enterprise value {_usd(ev)})." if ev else "."))
    if scen["ok"]:
        th.append(f"Against a {hurdle:.0%} required return, the base scenario earns {scen['rows']['Base']['ann']:+.1%} a year over {V.H} years and the probability-weighted estimate "
                  f"{scen['ev_ann']:+.1%}, with a bear-case loss of {max(0.0, -scen['rows']['Bear']['total']):.0%}.")
    if reqg is not None:
        th.append(f"Earning the required return needs about {reqg:.0%} year-1 growth, which looks {plaus.lower()}" + (f" against {f['hist_g']:.0%} historical growth." if f["hist_g"] is not None else "."))
    pos = [d for d, s in dims.items() if s is not None and s > .2]
    neg = [d for d, s in dims.items() if s is not None and s < -.1]
    own = [f"{d} ({dims[d]:+.2f}): " + ("; ".join(_top(df, {"Business quality": ["Profitability", "Track record"], "Growth": ["Growth"], "Financial health": ["Financial health"],
                                                            "Valuation": ["Valuation"], "Risk": ["Risk"]}.get(d, []), 2)) or "see scenarios") for d in pos]
    against = [f"{d} ({dims[d]:+.2f}): " + ("; ".join(_top(df, {"Business quality": ["Profitability", "Track record"], "Growth": ["Growth"], "Financial health": ["Financial health"],
                                                                "Valuation": ["Valuation"], "Risk": ["Risk"]}.get(d, []), 2, best=False)) or "see scenarios") for d in neg]
    against += [c["detail"].capitalize() for c in rt["constraints"]] + [f"{x[1]}: {x[2]}" for x in flags[:2]] + cons[:2]
    if scen["ok"]:
        against.append(f"Bear case: {scen['rows']['Bear']['total']:+.0%} versus today's price (probability {scen['rows']['Bear']['prob']:.0%}, an estimate).")
    changes = []
    if scen["ok"]:
        entry = scen["rows"]["Base"]["price_h"] / (1 + hurdle) ** V.H
        changes.append(f"A price near ${entry:,.2f} would make the base case earn the {hurdle:.0%} required return" + (" (current price is above that)." if price > entry else " (current price is already below that)."))
    if reqg is not None:
        changes.append(f"Growth running clearly below about {reqg:.0%} would undercut the base case; clearly above it would strengthen it.")
    if f["runway"] is not None:
        changes.append(f"Funding: cash covers about {f['runway']:.1f} years of burn; a financing on poor terms would raise dilution, and a path to break-even would lower it.")
    changes += [w for w in ("Estimate revisions turning negative would weaken the near-term setup." if nt_label == "Favorable" else "") if w]
    ctx = dict(info=info, hist=hist, inc=inc, cf=cf, bs=bs, est=est, price=price, name=name, f=f, framework=framework, fw=fw, tags=tags, why=why, df=df, cs=cs, dims=dims,
               vitems=vitems, scen=scen, reqg=reqg, plaus=plaus, mult_rows=mult_rows, ev=ev, hist_df=hist_df, hist_stats=hist_stats, peer_df=peer_df,
               profile=P, anchor=anchor, conf_factors=conf_factors, qitems=qitems, qscore=qscore, qcov=qcov, override=override, rating=rt, conf=conf, conf_label=conf_label, cov=cov, flags=flags, fm=fm, contradictions=cons,
               near_term=(nt_score, nt_label), thesis=" ".join(th), own=own, against=against, changes=changes, hurdle=hurdle, loss_making=f["loss_making"])
    ctx["qual_scen"] = qualitative_scenarios(ctx)
    ctx["brief"] = _brief(ctx)
    ctx["questions"] = research_questions(ctx)
    return ctx


def research_questions(c):
    """The ten business questions. Each answer is labelled with where it comes from; nothing is invented."""
    f, info, s = c["f"], c["info"], c["scen"]
    qi = c["qitems"]
    ev_ = lambda k: qi[k]["evidence"] if qi[k]["score"] is not None else None
    rows = [("What does the company do and how does it make money?", (info.get("longBusinessSummary") or "")[:700] or None, "Reported (Yahoo company description)"),
            ("Key products, customers and revenue drivers", (("Business lines detected from the description: " + ", ".join(c["profile"]["business_lines"]) + ". Revenue by line, customers and concentration are not available from free structured data.") if c["profile"]["business_lines"] else None), "Derived from the company description (AI review can summarise what it is given)"),
            ("Competitive advantage", " ".join(x for x in (ev_("pricing_power"), ev_("moat_economics")) if x) or None, "Computed (financial footprint only; moat type needs judgment)"),
            ("How large could the business become?", (f"Base scenario revenue in {V.H} years: ${s['rows']['Base']['assump']['Revenue at horizon'] / 1e9:,.2f}B vs ${f['rev0'] / 1e9:,.2f}B now. Addressable-market size is not available." if s["ok"] and s["mode"] == "revenue" and f["rev0"] else None), "Estimate (scenario assumption)"),
            ("Structural trends behind growth", None, "Not available from free data; AI review or your own view"),
            ("Evidence for the growth thesis", (f"Blended growth {f['g']:.0%} (analyst estimate {_p0(f['est_growth'])}, TTM {_p0(info.get('revenueGrowth'))}, 3-yr history {_p0(f['hist_g'])})." if f["g"] is not None else None), "Computed"),
            ("What must happen for the stock to become much more valuable?", (f"Year-1 growth of about {c['reqg']:.0%} (fading) is needed to earn the {c['hurdle']:.0%} required return; the bull case assumes {s['rows']['Bull']['assump'].get('Revenue growth, year 1', s['rows']['Bull']['assump'].get('EPS growth, year 2', 0)):.0%}." if c["reqg"] is not None and s["ok"] else None), "Estimate"),
            ("What could prevent it?", "; ".join(c["against"][:4]) or None, "Computed flags and scenario risks"),
            ("How much success is already priced in?", (f"Required growth looks {c['plaus'].lower()}" + (f"; trailing P/E sits at the {c['hist_stats']['P/E']['pct']:.0%} percentile of its own fiscal-year history." if c["hist_stats"] and "P/E" in c["hist_stats"] else ".")) if c["plaus"] != "n/a" else None, "Computed"),
            ("Return versus downside at today's price", (f"Base {s['rows']['Base']['ann']:+.1%}/yr, probability-weighted {s['ev_ann']:+.1%}/yr (estimated probabilities), bear case {s['rows']['Bear']['total']:+.0%} in total." if s["ok"] else None), "Estimate")]
    return [dict(Question=q, Answer=a or "Not available: " + (s.get("reason") or "insufficient free data"), Source=src if a else "Not available") for q, a, src in rows]


def facts_for_ai(c):
    """Everything the AI review may cite: deterministic outputs only."""
    s, rt, f = c["scen"], c["rating"], c["f"]
    df = c["df"]
    out = dict(ticker=c["info"].get("symbol"), name=c["name"], sector=c["info"].get("sector"), industry=c["info"].get("industry"), framework=c["fw"]["label"],
               framework_tags=c["tags"], framework_reason=c["why"][0], price=c["price"], market_cap=f["mc"], enterprise_value=c["ev"],
               model_rating=rt["rating"], model_score=rt["score"], model_constraints=[f"{x['rule']}: {x['detail']}" for x in rt["constraints"]],
               confidence=c["conf_label"], confidence_factors=[f"{n}: {w}" for n, _, w in c["conf_factors"]], classification={k: v for k, v in c["profile"].items() if k != "confidence"} | {"classification_confidence": c["profile"]["confidence"]["label"]}, horizon=rt["horizon"], required_return=c["hurdle"], dimension_scores=c["dims"], near_term_setup=c["near_term"][1],
               loss_making=c["loss_making"], runway_years=f["runway"], dilution_per_year=f["dilution"], revenue=f["rev0"], growth_blended=f["g"], gross_margin=f["gm"],
               operating_margin=f["m0"], free_cash_flow=f["fcf0"], forensic_flags=[f"{x[1]}: {x[2]}" for x in c["flags"]], contradictions=c["contradictions"],
               metrics=[dict(metric=r.Metric, category=r.Category, value=r.Value, score=r.Score) for r in df.itertuples()] if len(df) else [],
               multiples=[{k: v for k, v in r.items() if k != "Why"} | {"note": r["Why"]} for r in c["mult_rows"]],
               own_history=c["hist_stats"], peers=None if c["peer_df"] is None else c["peer_df"].to_dict("records"),
               required_growth_year1=c["reqg"], required_growth_assessment=c["plaus"],
               scenario_method=s.get("mode"), scenario_horizon_years=V.H, scenario_limitations=s.get("limitations"),
               scenarios=None if not s["ok"] else {k: dict(probability_estimate=v["prob"], price_at_horizon=v["price_h"], annualized_return=v["ann"], total_return=v["total"], assumptions=v["assump"]) for k, v in s["rows"].items()},
               qualitative_rubric={k: dict(score=i["score"], evidence=i["evidence"], source=i["source"]) for k, i in c["qitems"].items()},
               data_not_available=dict(unavailable_metrics=c["cov"]["unavailable"], not_meaningful_metrics=c["cov"]["not_meaningful"], scenario_reason=s.get("reason")))
    return Q.clean(out)
