"""The one place where a rating is decided. Every page shows the output of rate(); nothing else may produce a Buy/Hold/Sell label.

RATINGS (same definitions everywhere):
  Strong Buy  unusually attractive: strong evidence AND probability-weighted return well above the required return AND limited downside
  Buy         expected return clearly exceeds the required return for the risk taken
  Hold        roughly fair: expected return near the required return, or an attractive idea held back by a stated constraint
  Sell        expected return below the required return; holders are better off elsewhere
  Avoid       unattractive at this price: poor risk/reward, not merely weak current financials

METHOD: six dimensions (business quality, growth, financial health, valuation, risk, risk-reward) scored -1..+1, weighted by the company's
framework. Profit-based measures are skipped (not penalized) for loss-making companies, and cash burn is scored once (financial health),
so one underlying fact is never penalized in several dimensions. The qualitative view is blended in with a weight scaled by how much of it
has evidence. Constraints can cap a rating; each is shown with its reason and effect."""
import numpy as np

RATINGS = ["Avoid", "Sell", "Hold", "Buy", "Strong Buy"]
DEFS = {"Strong Buy": "Unusually attractive opportunity.", "Buy": "Expected return clearly exceeds the required return for the risk.",
        "Hold": "Roughly fair at this price, or held back by a stated constraint.", "Sell": "Expected return below the required return.",
        "Avoid": "Poor risk/reward at this price (not merely weak current financials)."}
DIMS = ("Business quality", "Growth", "Financial health", "Valuation", "Risk", "Risk-reward")
FRAMEWORKS = {
    "mature": dict(label="Mature profitable compounder", w=(.25, .10, .15, .20, .10, .20), hurdle=.09, horizon="3-5 years",
                   emphasis="earnings quality, cash conversion, returns on capital, price paid"),
    "established_growth": dict(label="Profitable growth compounder", w=(.22, .18, .12, .20, .08, .20), hurdle=.10, horizon="3-5 years",
                               emphasis="growth durability, returns on reinvestment, operating leverage, price paid"),
    "growth_profitable": dict(label="High-growth profitable company", w=(.20, .25, .10, .20, .05, .20), hurdle=.10, horizon="3-5 years",
                              emphasis="growth durability, operating leverage, what is priced in"),
    "emerging_growth": dict(label="Emerging high-growth company (loss-making)", w=(.20, .25, .20, .10, .05, .20), hurdle=.12, horizon="3-5 years",
                            emphasis="growth, unit economics, funding runway, dilution; profits are not required"),
    "speculative": dict(label="Speculative growth company", w=(.15, .20, .20, .10, .05, .30), hurdle=.15, horizon="3-5 years, high uncertainty",
                        emphasis="probability of failure vs upside, financing risk, dilution"),
    "turnaround": dict(label="Turnaround / restructuring", w=(.15, .10, .25, .20, .10, .20), hurdle=.12, horizon="2-4 years",
                       emphasis="solvency, evidence the decline is stopping, price vs recovery"),
    "cyclical": dict(label="Cyclical company", w=(.15, .10, .15, .25, .10, .25), hurdle=.10, horizon="3-5 years (one cycle)",
                     emphasis="mid-cycle earnings, balance sheet through a downturn, price vs normalized earnings"),
    "semis_ai": dict(label="Semiconductor / AI infrastructure", w=(.20, .20, .10, .20, .10, .20), hurdle=.11, horizon="3-5 years",
                     emphasis="normalized earnings, customer concentration, supply catching up with demand, capital intensity"),
    "capital_intensive": dict(label="Capital-intensive / asset-heavy", w=(.15, .10, .20, .25, .10, .20), hurdle=.08, horizon="3-5 years",
                              emphasis="returns on capital, leverage, asset valuation"),
    "financial": dict(label="Financial services", w=(.25, .10, .15, .25, .10, .15), hurdle=.10, horizon="3-5 years",
                      emphasis="returns on equity, price/book, credit and rate sensitivity"),
}
clip = lambda x, a=-1.0, b=1.0: float(np.clip(x, a, b))


def label(s):
    return "Strong Buy" if s > .45 else "Buy" if s > .18 else "Hold" if s > -.12 else "Sell" if s > -.35 else "Avoid"


def confidence(framework, cov_q, qcov, scen, anchor, class_label, have_est, have_hist=False, have_peers=False):
    """Evidence quality, NOT upside. Returns (coarse proxy, label, factors). Percentages are deliberately not shown: they would imply precision the inputs do not have."""
    if scen and scen["ok"]:
        base = scen["rows"]["Base"]["price_h"]
        spread = (scen["rows"]["Bull"]["price_h"] - scen["rows"]["Bear"]["price_h"]) / base if base > 0 else 3.0
        disp = clip(1 - spread / 3, 0, 1)
    else:
        disp = 0.0
    support = 1.0 if (have_hist and have_peers) else .6 if (have_hist or have_peers or (anchor and anchor.get("supported", True))) else .2
    cl = {"High": 1.0, "Medium": .6, "Low": .2}[class_label]
    fac = [("Data completeness", clip(cov_q, 0, 1), f"{cov_q:.0%} of applicable quantitative metrics available"),
           ("Consensus estimates", 1.0 if have_est else 0.0, "available" if have_est else "none from free data; growth is inferred from reported figures"),
           ("Support for the exit multiple", support, "own history and peers" if support == 1.0 else "own history or peers" if support == .6 else "sector rule of thumb only"),
           ("Classification", cl, f"{class_label} confidence in the business-model classification"),
           ("Scenario dispersion", disp, "narrow range of outcomes" if disp > .6 else "wide range of outcomes" if disp < .3 else "moderate range of outcomes"),
           ("Qualitative evidence", clip(qcov / .6, 0, 1), f"{qcov:.0%} of the qualitative rubric has evidence")]
    wts = [.25, .10, .25, .15, .15, .10]
    sc = sum(w * v for w, (_, v, _) in zip(wts, fac))
    label = "High" if sc >= .70 and support >= .6 and class_label != "Low" and cov_q >= .7 else "Medium" if sc >= .45 else "Low"
    if scen is None or not scen["ok"]: label = "Low"
    if framework in ("speculative",) and label == "High": label = "Medium"
    if framework in ("emerging_growth", "turnaround") and label == "High": label = "Medium"
    return {"High": .75, "Medium": .50, "Low": .30}[label], label, fac


def rate(framework, dims, scen, conf, qual=None, w_qual=.5, runway=None, raise_ratio=None, override=None):
    fw = FRAMEWORKS[framework]
    avail = [(d, w) for d, w in zip(DIMS, fw["w"]) if dims.get(d) is not None]
    tot = sum(w for _, w in avail) or 1.0
    contrib = {d: dims[d] * w / tot for d, w in avail}
    quant = sum(contrib.values())
    qw = w_qual * qual["coverage"] if qual and qual.get("coverage", 0) > 0 else 0.0
    score = (1 - qw) * quant + qw * (qual["score"] if qw else 0.0)
    if override:
        score += override["adjustment"]
    score = clip(score)
    cons, s = [], score

    def cap(rule, detail, to):
        nonlocal s
        cons.append({"rule": rule, "detail": detail, "from": label(s), "to": label(to)}); s = to
    ok = bool(scen and scen["ok"])
    hurdle = fw["hurdle"]
    if ok:
        base_ann, ev_ann = scen["rows"]["Base"]["ann"], scen["ev_ann"]
        bear_loss = max(0.0, -scen["rows"]["Bear"]["total"])  # loss vs today's price in the bear case
        if s > .18 and base_ann < hurdle and ev_ann < hurdle:
            cap("Required-return floor", f"modeled base-case return {base_ann:+.1%}/yr and probability-weighted {ev_ann:+.1%}/yr are both below the {hurdle:.0%} required return", .18)
        if s > -.12 and ev_ann < hurdle - .08 and base_ann < hurdle - .05:
            cap("Expected-return ceiling", f"base case {base_ann:+.1%}/yr and probability-weighted {ev_ann:+.1%}/yr are well below the {hurdle:.0%} required return, so Hold (roughly fair) would overstate the case", -.12)
        if s > -.35 and ev_ann < 0 and bear_loss >= .5:
            if conf >= .45:
                cap("Poor risk/reward", f"probability-weighted return is negative ({ev_ann:+.1%}/yr) and the bear case loses {bear_loss:.0%}", -.35)
        if s <= -.12 and ev_ann >= hurdle and base_ann >= hurdle:
            cap("Metrics alone do not justify Sell/Avoid", f"modeled returns ({base_ann:+.1%}/yr base, {ev_ann:+.1%}/yr probability-weighted) meet the {hurdle:.0%} requirement; weak current financials are not enough", -.12 + 1e-9)
        lim = .75 if framework == "speculative" else .60
        if s > .45 and (ev_ann < hurdle + .08 or bear_loss > lim or conf < .55):
            cap("Strong Buy gate", f"needs probability-weighted return ≥ {hurdle + .08:.0%}/yr (got {ev_ann:+.1%}), bear-case loss ≤ {lim:.0%} (got {bear_loss:.0%}) and confidence ≥ 55% (got {conf:.0%})", .45)
    else:
        if s > .45:
            cap("Strong Buy gate", "no scenario analysis is available, so the expected return cannot be assessed", .45)
        if s <= -.35:
            cap("Avoid requires valuation evidence", "Avoid means poor risk/reward at the current price, which cannot be shown without scenarios", -.35 + 1e-9)
    if framework in ("emerging_growth", "speculative", "turnaround") and s > .18:
        if runway is not None and runway < 1.5:
            cap("Funding constraint", f"cash covers only {runway:.1f} years at the current burn; near-term financing risk limits the rating", .18)
        elif raise_ratio is not None and raise_ratio > .5:
            cap("Funding constraint", f"the base case needs new capital worth {raise_ratio:.0%} of today's market cap; dilution risk limits the rating", .18)
    if s <= -.35 and conf < .45:
        cap("Avoid needs evidence quality", f"confidence is only {conf:.0%}; a low-confidence analysis can support Sell but not Avoid", -.35 + 1e-9)
    rating = label(s)
    return dict(score=s, raw_score=score, quant_only=quant, rating=rating, raw_rating=label(score), constraints=cons, contrib=contrib,
                qual_weight=qw, hurdle=hurdle, framework=framework, label_text=fw["label"], horizon=fw["horizon"])


def near_term_setup(cs, rev_revision=None):
    """Not a rating: a 3-12 month read on momentum, analyst sentiment and estimate revisions."""
    parts = [cs[k] for k in ("Technicals", "Analyst") if k in cs]
    if rev_revision is not None:
        parts.append(clip(rev_revision / .05))
    if not parts:
        return None, "Unavailable"
    s = float(np.mean(parts))
    return s, ("Favorable" if s > .25 else "Unfavorable" if s < -.25 else "Neutral")
