"""Equity research workstation. Every number and label on screen comes from engine.analyze(); pages only render it."""
import html
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st


class _CaseInsensitiveFinder:
    """Safety net: files uploaded from a phone often get capitalised names (Data.py). Linux imports are case-sensitive, so resolve those here.
    It is appended last, so exact-case modules are always found first."""
    @staticmethod
    def find_spec(name, path=None, target=None):
        if "." in name:
            return None
        here = os.path.dirname(os.path.abspath(__file__))
        for f in os.listdir(here):
            if f.lower() == name.lower() + ".py":
                return importlib.util.spec_from_file_location(name, os.path.join(here, f))
        return None


sys.meta_path.append(_CaseInsensitiveFinder)

import data as D
import engine as E
import metrics as M
import qual as Q
import rating as RT
import valuation as V

st.set_page_config(page_title="Equity Research", layout="wide", initial_sidebar_state="auto")   # "auto": open on desktop, collapsed on phones
POS, NEG, MUTED = "#12724A", "#B42318", "#5B6676"
RCOL = {"Strong Buy": POS, "Buy": POS, "Hold": MUTED, "Sell": NEG, "Avoid": NEG}
PAGES = ["Overview", "Financials", "Valuation", "Scenarios", "Research", "Data & method"]
EXAMPLES = ["AAPL", "MSFT", "AMZN", "NVDA", "JPM", "KO"]
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.stApp { background: #F6F7F9; }
.block-container { padding: 3.4rem 1rem 3rem; max-width: 1280px; }
@media (min-width: 900px) { .block-container { padding: 2.8rem 2rem 3rem; } }
/* Type: scoped to our own content. Streamlit's icon elements are never matched, so their icon font keeps working. */
.block-container [data-testid="stMarkdownContainer"] p, .block-container [data-testid="stMarkdownContainer"] li, .block-container table.t, .topbar, .sec, .na, .note, .bl,
.block-container label p, .block-container input, .block-container button p { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
.block-container [data-testid="stMarkdownContainer"] p, .block-container [data-testid="stMarkdownContainer"] li { font-size: .9rem; line-height: 1.5; color: #1B2430; max-width: 74ch; }
.block-container [data-testid="stMarkdownContainer"] p { margin: .15rem 0 .5rem; }
.sec { font-size: .78rem; font-weight: 600; color: #0F1B2D; letter-spacing: .02em; margin: 1.25rem 0 .45rem; padding-bottom: .25rem; border-bottom: 1px solid #D9DEE5; }
.sec:first-child { margin-top: .2rem; }
.h1 { font-size: 1.15rem; font-weight: 600; color: #0F1B2D; margin: 0 0 .3rem; }
/* The page is always light. In a dark system theme Streamlit turns inherited text near-white, so set the colors that were inherited. */
[data-testid="stMain"] [data-testid="stMarkdownContainer"] { color: #1B2430; }
[data-testid="stMain"] [data-testid="stExpander"] summary, [data-testid="stMain"] [data-testid="stExpander"] summary * { color: #1B2430; }
[data-testid="stMain"] [data-testid="InputInstructions"] { color: #5B6676 !important; }
[data-testid="stMain"] [data-testid="stTextInputRootElement"] { background: #fff; border: 1px solid #D9DEE5; }
[data-testid="stMain"] [data-testid="stTextInput"] input { color: #1B2430; background: #fff; }
/* sidebar: compact navy rail; bounded drawer on phones */
[data-testid="stSidebar"] { background: #0F1B2D; }
@media (max-width: 767px) { [data-testid="stSidebar"] { max-width: min(290px, 80vw) !important; min-width: 0 !important; } }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p, [data-testid="stSidebar"] label p { color: #D5DBE5; font-size: .82rem; }
[data-testid="stSidebar"] .brand { font-weight: 600; letter-spacing: .14em; font-size: .72rem; color: #fff; margin: .2rem 0 .8rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .1rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .45rem .7rem; border-left: 2px solid transparent; width: 100%; margin: 0; }
[data-testid="stSidebar"] [role="radiogroup"] label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { border-left-color: #fff; background: rgba(255,255,255,.08); }
/* horizontal page navigation for phones and tablets; the sidebar rail takes over on wide screens */
.st-key-nav_top { margin: .2rem 0 .8rem; }
.st-key-nav_top [role="radiogroup"] { flex-wrap: nowrap; overflow-x: auto; gap: 0; border-bottom: 1px solid #D9DEE5; -webkit-overflow-scrolling: touch; scrollbar-width: none; }
.st-key-nav_top [role="radiogroup"]::-webkit-scrollbar { display: none; }
.st-key-nav_top [role="radiogroup"] label { padding: .6rem .8rem; margin: 0; white-space: nowrap; min-height: 44px; align-items: center; border-bottom: 2px solid transparent; }
.st-key-nav_top [role="radiogroup"] label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-nav_top [role="radiogroup"] label:has(input:checked) { border-bottom-color: #0F1B2D; }
.st-key-nav_top [role="radiogroup"] label:has(input:checked) p { font-weight: 600; color: #0F1B2D; }
@media (min-width: 900px) { .st-key-nav_top { display: none; } }
/* below 1000px (phones and tablets) page columns stack; there is not room for two readable columns of tables */
@media (max-width: 999px) { .block-container [data-testid="stHorizontalBlock"] { flex-wrap: wrap; } .block-container [data-testid="stColumn"] { min-width: 100% !important; } }
[data-testid="stBaseButton-pills"], [data-testid="stBaseButton-pillsActive"] { min-height: 40px; }
/* search row stays on one line, with touch-sized controls (Streamlit forms carry no key class, so target the form itself) */
[data-testid="stForm"] { padding: 0; border: none; max-width: 560px; }
[data-testid="stForm"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .5rem; align-items: stretch; }
[data-testid="stForm"] [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 auto !important; }
[data-testid="stForm"] [data-testid="stColumn"]:last-child { flex: 0 0 5rem !important; width: 5rem !important; }
[data-testid="stTextInput"] input { min-height: 44px; box-sizing: border-box; }
[data-testid="stTextInputRootElement"] { min-height: 44px; }
[data-testid="stFormSubmitButton"] > button, .stButton > button { min-height: 44px; border-radius: 2px; border: 1px solid #1F3A5F; background: #1F3A5F; }
[data-testid="stFormSubmitButton"] button p, .stButton button p { color: #fff !important; font-weight: 500; }
[data-testid="stExpandSidebarButton"], [data-testid="stBaseButton-headerNoPadding"] { min-width: 44px; min-height: 44px; display: flex; align-items: center; justify-content: center; }
[data-testid="stElementToolbar"] { display: none; }
/* top bar: grid that stacks sensibly */
.topbar { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: .55rem 1rem; align-items: end; border-bottom: 1px solid #D9DEE5; padding-bottom: .7rem; margin: .5rem 0 .4rem; }
.tb-name { grid-column: 1 / -1; font-size: 1.1rem; font-weight: 600; color: #0F1B2D; line-height: 1.2; }
.tb-name span { font-size: .82rem; color: #5B6676; font-weight: 500; margin-left: .4rem; }
.kv { font-size: .66rem; color: #5B6676; text-transform: uppercase; letter-spacing: .05em; line-height: 1.15; min-width: 0; }
.kv.wide { grid-column: span 2; }
.kv b { display: block; font-size: .92rem; color: #1B2430; font-weight: 600; font-variant-numeric: tabular-nums; text-transform: none; letter-spacing: 0; overflow-wrap: anywhere; }
.chip { display: inline-block; padding: 1px 8px; border: 1px solid; border-radius: 2px; font-weight: 600; font-size: .82rem; }
/* tables: tabular numerals, controlled horizontal scroll */
.tw { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: 0 0 .6rem; max-width: 100%; }
table.t { width: 100%; border-collapse: collapse; font-size: .8rem; font-variant-numeric: tabular-nums; }
.t th, .t td { border-left: none; border-right: none; }
.t th { text-align: right; font-weight: 600; color: #5B6676; border-bottom: 1px solid #B8C0CC; padding: .3rem .5rem; font-size: .66rem; text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }
.t td { padding: .32rem .5rem; border-bottom: 1px solid #EDF0F4; text-align: right; vertical-align: top; }
.t th.l, .t td.l { text-align: left; }
.muted { color: #5B6676; } .pos { color: #12724A; } .neg { color: #B42318; }
.na { border: 1px solid #D9DEE5; border-left: 3px solid #8A94A3; background: #fff; padding: .65rem .85rem; font-size: .84rem; color: #2B3544; margin: .4rem 0; }
.note { font-size: .74rem; color: #5B6676; line-height: 1.45; margin: .15rem 0 .5rem; }
ul.bl { margin: .1rem 0 .6rem 1.1rem; padding: 0; } ul.bl li { font-size: .88rem; line-height: 1.5; margin-bottom: .25rem; color: #1B2430; }
.tag { font-size: .66rem; color: #5B6676; border: 1px solid #D9DEE5; padding: 0 .35rem; margin-left: .3rem; white-space: nowrap; }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ formatting and small components
esc = lambda x: html.escape(str(x))
ok = lambda x: x is not None and isinstance(x, (int, float, np.floating)) and np.isfinite(x)


def md(body, unsafe_allow_html=False, **kw):
    """Markdown with '$' made literal: unescaped dollar pairs are parsed as math and swallow the text between them."""
    body = str(body)
    st.markdown(body.replace("$", "&#36;") if unsafe_allow_html else body.replace("$", "\\$"), unsafe_allow_html=unsafe_allow_html, **kw)


def section(title):
    md(f'<div class="sec">{esc(title)}</div>', True)


def money(x):
    if not ok(x): return "n/a"
    a, s = abs(x), "-" if x < 0 else ""
    return f"{s}${a / 1e12:,.2f}T" if a >= 1e12 else f"{s}${a / 1e9:,.2f}B" if a >= 1e9 else f"{s}${a / 1e6:,.0f}M" if a >= 1e6 else f"{s}${a:,.0f}"


pct = lambda x, d=1: f"{x * 100:+.{d}f}%" if ok(x) else "n/a"
pct0 = lambda x, d=1: f"{x * 100:.{d}f}%" if ok(x) else "n/a"
mx = lambda x: f"{x:.1f}x" if ok(x) else "n/a"
tone = lambda x, txt: f'<span class="{"pos" if x > 0.0005 else "neg" if x < -0.0005 else "muted"}">{esc(txt)}</span>' if ok(x) else '<span class="muted">n/a</span>'
chip = lambda r: f'<span class="chip" style="color:{RCOL[r]};border-color:{RCOL[r]}">{esc(r)}</span>'


def table(cols, rows, left=(0,)):
    """HTML table (tabular numerals) inside a horizontally scrollable wrapper, so wide tables never push the page sideways."""
    th = "".join(f'<th class="{"l" if i in left else ""}">{esc(c)}</th>' for i, c in enumerate(cols)) if any(cols) else ""
    body = "".join("<tr>" + "".join(f'<td class="{"l" if i in left else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>" for r in rows)
    md(f'<div class="tw"><table class="t" style="min-width:{max(280, 76 * len(cols))}px">{f"<thead><tr>{th}</tr></thead>" if th else ""}<tbody>{body}</tbody></table></div>', True)


def na(title, reasons, hint=""):
    """Unavailable-data state: says what is missing and why. Never a blank page."""
    li = "".join(f"<li>{esc(r)}</li>" for r in reasons)
    md(f'<div class="na"><b>{esc(title)}</b><ul style="margin:.3rem 0 .2rem 1rem;padding:0">{li}</ul><span class="note">{esc(hint)}</span></div>', True)


def bullets(items):
    for i in items:
        md(f"- {i}")


def tagged(items):
    """List whose every item states what kind of statement it is (reported, calculated, forecast, judgment)."""
    if items:
        md('<ul class="bl">' + "".join(f'<li>{esc(t)} <span class="tag">{esc(g)}</span></li>' for t, g in items) + "</ul>", True)


def fa(k, v):
    if isinstance(v, str): return esc(v)
    kl = k.lower()
    if any(w in kl for w in ("growth", "roe", "dilution", "change")): return pct(v)
    if "exit" in kl: return mx(v)
    if any(w in kl for w in ("revenue at", "burn", "raise", "dividends")): return money(v)
    if "book value" in kl or "eps" in kl: return f"${v:,.2f}"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


# ------------------------------------------------------------------ shell
def _sync(key):
    st.session_state["page"] = st.session_state[key]


def sidebar():
    """Desktop rail (collapsed by default on phones). Page navigation and settings only; search lives in the page."""
    st.session_state.setdefault("page", PAGES[0])
    st.session_state["nav_side"] = st.session_state["page"]
    st.sidebar.markdown('<div class="brand">EQUITY RESEARCH</div>', unsafe_allow_html=True)
    st.sidebar.radio("Navigation", PAGES, key="nav_side", label_visibility="collapsed", on_change=_sync, args=("nav_side",))
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    wq = st.sidebar.slider("Qualitative weight (%)", 0, 100, 50, key="wq")
    st.sidebar.markdown('<div class="note" style="color:#8A94A3">Qualitative weight is the share of the rating from the qualitative view, scaled by how much of it has evidence. Free data: Yahoo Finance and SEC filings. Educational tool, not investment advice.</div>', unsafe_allow_html=True)
    return wq / 100


def search_bar():
    try:
        form = st.form("search", border=False)
    except TypeError:
        form = st.form("search")
    with form:
        c1, c2 = st.columns([5, 1])
        t = c1.text_input("Ticker", value=st.session_state.get("sym", ""), placeholder="Ticker symbol, e.g. MSFT", label_visibility="collapsed")
        go = c2.form_submit_button("Load", width="stretch")
    if go and t.strip():
        st.session_state["sym"] = t.strip().upper()
    return st.session_state.get("sym")


def nav_top():
    st.session_state["nav_top"] = st.session_state["page"]
    st.radio("Navigation", PAGES, key="nav_top", horizontal=True, label_visibility="collapsed", on_change=_sync, args=("nav_top",))


def topbar(c):
    f, rt, i = c["f"], c["rating"], c["info"]
    kv = lambda k, v: f'<div class="kv">{esc(k)}<b>{v}</b></div>'
    md('<div class="topbar">' + f'<div class="tb-name">{esc(c["name"] or i["symbol"])}<span>{esc(i["symbol"])}</span></div>' + kv("Price", f"${c['price']:,.2f}") +
       kv("Market cap", money(f["mc"])) + kv("Enterprise value", money(c["ev"])) + kv("Rating", chip(rt["rating"])) + kv("Horizon", esc(rt["horizon"])) +
       kv("Confidence", esc(c["conf_label"])) + kv("Framework", esc(c["fw"]["label"])).replace('class="kv"', 'class="kv wide"') + "</div>", True)


def landing():
    section("Search a company")
    md("Enter a ticker symbol to load ratings, valuation and scenarios. Data comes from Yahoo Finance and SEC filings; nothing is estimated where data is missing.")
    st.markdown('<div class="note">Examples</div>', unsafe_allow_html=True)
    pick = st.pills("Examples", EXAMPLES, selection_mode="single", key="examples", label_visibility="collapsed") if hasattr(st, "pills") else None
    if pick:
        st.session_state["sym"] = pick
        st.session_state["examples"] = None
        st.rerun()


# ------------------------------------------------------------------ pages
def page_overview(c, extra):
    rt, s, P = c["rating"], c["scen"], c["profile"]
    B = c["brief"]
    L, R = st.columns([3, 2], gap="large")
    with L:
        section("Business")
        tagged(B["business"])
        section("Verified facts")
        tagged(B["facts"] or [("Reported statements were not available for this ticker.", "Unavailable")])
        section("Valuation considerations")
        tagged(B["valuation"] or [("No meaningful multiples could be calculated from the available data.", "Unavailable")])
        a, b = st.columns(2)
        with a:
            section("Reasons to own")
            bullets(c["own"] or ["No dimension scores clearly favorable; see scenarios."])
        with b:
            section("Reasons not to own")
            bullets(c["against"] or ["No material negatives identified from the available data."])
        section("What must happen for the thesis to work")
        tagged(B["success"])
        section("What would invalidate it")
        tagged(B["invalidate"] or [("No specific invalidation triggers could be quantified.", "Unavailable")])
        section("Uncertain assumptions")
        tagged(B["uncertain"])
    with R:
        section("Rating")
        nt, ntl = c["near_term"]
        table(["", ""], [["Rating", chip(rt["rating"])], ["Score", f"{rt['score']:+.2f}"], ["Expected horizon", esc(rt["horizon"])], ["Confidence", esc(c["conf_label"])],
                         ["Near-term setup (not a rating)", esc(ntl)], ["Required return", pct0(c["hurdle"], 0)]], left=(0, 1))
        for k in rt["constraints"]:
            md(f'<div class="na"><b>{esc(k["rule"])}</b>: {esc(k["detail"])}. Rating moved {esc(k["from"])} to {esc(k["to"])}.</div>', True)
        with st.expander(f"Why confidence is {c['conf_label'].lower()}"):
            table(["Evidence", "Strength", "Detail"], [[esc(n), "Strong" if v >= .75 else "Partial" if v >= .4 else "Weak", esc(w)] for n, v, w in c["conf_factors"]], left=(0, 2))
            md('<div class="note">Confidence reflects evidence quality and how well the assumptions are supported, not how large the upside is. No percentage is shown because the inputs do not support that precision.</div>', True)
        section("Classification")
        cy, ci = P["cyclicality"], P["capital_intensity"]
        table(["Attribute", "Assessment"], [["Sector / industry (Yahoo)", esc(f"{P['sector'] or 'n/a'} / {P['industry'] or 'n/a'}")], ["Business lines", esc(", ".join(P["business_lines"]) or "not identified")],
                                          ["Maturity", esc(P["maturity"])], ["Growth", esc(P["growth_profile"])], ["Profitability", esc(P["profitability"])],
                                          ["Cyclicality", esc(f"{cy['level']} ({'; '.join(cy['evidence'])})")], ["Capital intensity", esc(ci["level"] + (f" ({ci['value']:.0%} of revenue)" if ci["value"] is not None else ""))],
                                          ["Risk category", esc(P["risk_category"])], ["Investment framework", esc(c["fw"]["label"])],
                                          ["Classification confidence", esc(P["confidence"]["label"] + (": " + "; ".join(P["confidence"]["bad"]) if P["confidence"]["bad"] else ""))]], left=(0, 1))
        if P["sector_note"]:
            md(f'<div class="note">{esc(P["sector_note"])}</div>', True)
        section("Dimensions")
        table(["Dimension", "Weight", "Score"], [[esc(d), f"{w:.0%}", tone(c["dims"][d], f"{c['dims'][d]:+.2f}") if c["dims"][d] is not None else '<span class="muted">not assessed</span>'] for d, w in zip(RT.DIMS, c["fw"]["w"])])
        md(f'<div class="note">Weights depend on the framework. Qualitative view blended in at {rt["qual_weight"]:.0%} (scaled by evidence coverage {c["qcov"]:.0%}). Profit measures are skipped, not penalized, for loss-making companies.</div>', True)
        section("Scenarios at a glance")
        if s["ok"]:
            table(["Case", "Total", "Per year", "Probability*"], [[k, tone(v["total"], pct(v["total"], 0)), tone(v["ann"], pct(v["ann"])), pct0(v["prob"], 0)] for k, v in s["rows"].items()])
            md(f'<div class="note">*Estimates, not statistics. Probability-weighted: {pct(s["ev_ann"])} per year over {V.H} years. Assumptions are on the Scenarios page.</div>', True)
        else:
            na("Numerical scenarios unavailable", [s["reason"] or "insufficient data"], "See the Scenarios page for a qualitative view.")
        section("What could change the rating")
        bullets(c["changes"] or ["Not enough data to identify specific triggers."])
    section("Price, five years")
    st.line_chart(px_series(c), height=200)
    fr = stmt_frame(c)
    if fr is not None and "Revenue" in fr.index:
        section("Revenue and margins")
        x, y = st.columns(2)
        x.bar_chart(fr.loc["Revenue"].iloc[::-1] / 1e9, height=180)
        m = pd.DataFrame({k: (fr.loc[v] / fr.loc["Revenue"]).iloc[::-1] for k, v in (("Gross margin", "Gross profit"), ("Operating margin", "Operating income")) if v in fr.index})
        if len(m.columns): y.line_chart(m, height=180)
        md('<div class="note">Revenue in $ billions by fiscal year; margins as a fraction of revenue. Source: reported statements.</div>', True)


def px_series(c, years=5):
    h = c["hist"]["Close"].dropna()
    h.index = pd.to_datetime(h.index).tz_localize(None) if getattr(h.index, "tz", None) is not None else pd.to_datetime(h.index)
    return h.tail(252 * years)


def stmt_frame(c):
    inc, cf, bs = c["inc"], c["cf"], c["bs"]
    if inc is None or inc.empty: return None
    cols = list(inc.columns)
    g = lambda df, names: next((df.loc[n] for n in names if df is not None and not df.empty and n in df.index), None)
    rows = {"Revenue": g(inc, ["Total Revenue", "Operating Revenue"]), "Gross profit": g(inc, ["Gross Profit"]), "Operating income": g(inc, ["Operating Income"]),
            "Net income": g(inc, ["Net Income"]), "Operating cash flow": g(cf, ["Operating Cash Flow"]), "Free cash flow": g(cf, ["Free Cash Flow"]),
            "Stock-based compensation": g(cf, ["Stock Based Compensation"]), "Cash and short-term investments": g(bs, ["Cash Cash Equivalents And Short Term Investments"]),
            "Total debt": g(bs, ["Total Debt"]), "Shareholders' equity": g(bs, ["Stockholders Equity", "Common Stock Equity"]), "Shares outstanding": g(bs, ["Ordinary Shares Number", "Share Issued"])}
    df = pd.DataFrame({k: v.reindex(cols) for k, v in rows.items() if v is not None}).T
    df.columns = [str(pd.to_datetime(x).year) for x in df.columns]
    return df


def page_financials(c, extra):
    fr = stmt_frame(c)
    section("Annual financial statements")
    if fr is None:
        na("Financial statements unavailable", ["Neither Yahoo Finance nor SEC filings returned annual statements for this ticker.", f"SEC status: {extra['sec_status']}"],
           "SEC covers US 10-K filers. ETFs, foreign issuers and very new listings often have no statements.")
        return
    def fmt(k, v):
        if not ok(v): return '<span class="muted">n/a</span>'
        return f"{v / 1e6:,.0f}" if k == "Shares outstanding" else f"{v / 1e6:,.0f}"
    table(["$ millions (shares in millions)"] + list(fr.columns), [[esc(k)] + [fmt(k, v) for v in fr.loc[k]] for k in fr.index])
    section("Margins and growth")
    rev = fr.loc["Revenue"] if "Revenue" in fr.index else None
    rows = []
    for lab, num in (("Gross margin", "Gross profit"), ("Operating margin", "Operating income"), ("Net margin", "Net income"), ("Free-cash-flow margin", "Free cash flow")):
        if rev is not None and num in fr.index: rows.append([lab] + [tone(0, pct0(n / r)) if ok(n) and ok(r) and r else '<span class="muted">n/a</span>' for n, r in zip(fr.loc[num], rev)])
    if rev is not None:
        gr = [(rev.iloc[i] / rev.iloc[i + 1] - 1) if i + 1 < len(rev) and ok(rev.iloc[i]) and ok(rev.iloc[i + 1]) and rev.iloc[i + 1] > 0 else None for i in range(len(rev))]
        rows.append(["Revenue growth"] + [tone(g, pct(g)) if g is not None else '<span class="muted">n/a</span>' for g in gr])
    table([""] + list(fr.columns), rows)
    src = next((d for k, v, d in extra["sources"] if k == "Financial statements"), "")
    md(f'<div class="note">Source: {esc(next((v for k, v, d in extra["sources"] if k == "Financial statements"), ""))}. {esc(src)}. Annual figures; the latest quarters are not included.</div>', unsafe_allow_html=True)


def page_valuation(c, extra):
    f, info = c["f"], c["info"]
    prof = M.profile_for(info, info.get("sector"))
    section("What investors are paying")
    nd = (f["debt0"] or 0) - (f["cash0"] or 0) if f["debt0"] is not None and f["cash0"] is not None else None
    table(["", "Value", "Source"], [["Market capitalization", money(f["mc"]), "price x shares (Yahoo / SEC cover page)"], ["+ Debt (incl. minority interest)", money(f["debt0"]), "latest balance sheet"],
                                   ["- Cash and short-term investments", money(f["cash0"]), "latest balance sheet"], ["= Enterprise value", money(c["ev"]), "computed; Yahoo reports " + money(info.get("enterpriseValue"))]], left=(0, 2))
    md(f'<div class="note">Framework: {esc(c["fw"]["label"])}. Emphasis: {esc(c["fw"]["emphasis"])}. Multiples are judged against sector rules of thumb, the company\'s own history and peers where available; not every multiple applies to every business.</div>', unsafe_allow_html=True)
    section("Multiples and whether they mean anything here")
    ref = {"Forward P/E": "Forward P/E", "Trailing P/E": "Trailing P/E", "EV / EBITDA": "EV / EBITDA", "EV / Sales": "EV / Revenue", "Price / Sales": "Price / Sales", "Price / Book": "Price / Book"}
    rows = []
    for r in c["mult_rows"]:
        band = prof["over"].get(ref.get(r["Multiple"], ""))
        refs = f"{band[0]:g}x to {band[1]:g}x" if band and r["Status"] == "Meaningful" else "n/a"
        val = (pct0(r["Value"], 1) if r["Multiple"] == "FCF yield" else mx(r["Value"])) if r["Status"] == "Meaningful" else '<span class="muted">not meaningful</span>' if r["Status"] == "Not meaningful" else '<span class="muted">unavailable</span>'
        rows.append([esc(r["Multiple"]), val, esc(r["Relevance"]), refs, esc(r["Why"])])
    table(["Multiple", "Value", "Relevance", "Sector range", "Note"], rows, left=(0, 2, 4))
    md('<div class="note">Negative earnings, EBITDA or cash flow are marked not meaningful instead of being shown as a negative multiple. Ranges are rules of thumb, not live peer data.</div>', unsafe_allow_html=True)
    section("Valuation against the company's own history")
    if c["hist_df"] is not None:
        table(["Fiscal year end", "Price", "P/E", "P/S"], [[esc(r["FY end"]), f"${r['Price']:,.2f}", mx(r["P/E"]) if ok(r["P/E"]) else '<span class="muted">n/a</span>', mx(r["P/S"])] for _, r in c["hist_df"].iterrows()])
        for k, v in (c["hist_stats"] or {}).items():
            md(f'<div class="note">{k}: current {mx(v["current"])} vs fiscal-year-end range {mx(v["low"])} to {mx(v["high"])}, median {mx(v["median"])}; sits at the {v["pct"]:.0%} percentile of {v["n"]} observations.</div>', unsafe_allow_html=True)
    else:
        na("Own-history multiples unavailable", ["Fewer than three fiscal years with price, share count and results."])
    section("Peers")
    if c["peer_df"] is not None:
        table(["Metric", "Company", "Peer median", "Difference"], [[esc(r["Metric"]), (pct0(r["Company"], 1) if r["Metric"] in ("Op margin", "Rev growth") else mx(r["Company"])), (pct0(r["Peer median"], 1) if r["Metric"] in ("Op margin", "Rev growth") else mx(r["Peer median"])),
                                                              (pct(r["Premium"], 1) if r["Metric"] not in ("Op margin", "Rev growth") else f"{r['Premium'] * 100:+.1f} pts")] for _, r in c["peer_df"].iterrows()])
        md('<div class="note">Peers are the top companies Yahoo lists for the same industry; they differ in size and mix, so treat the comparison as context.</div>', unsafe_allow_html=True)
    else:
        na("Peer comparison unavailable", ["Yahoo did not return an industry peer list for this ticker.", "No peer multiples were estimated."])
    section("What the current price requires")
    if c["reqg"] is not None:
        md(f"To earn the {c['hurdle']:.0%} required return over {V.H} years in the base scenario, year-1 growth of about {c['reqg']:.0%} (fading afterwards) is needed. That looks **{c['plaus'].lower()}**"
                 + (f" against {f['hist_g']:.0%} annual growth over the last three fiscal years." if f["hist_g"] is not None else "."))
        md('<div class="note">Computed with the same scenario machinery as the Scenarios page: the growth rate that makes the base case earn the required return.</div>', unsafe_allow_html=True)
    else:
        na("Required growth unavailable", [c["scen"]["reason"] or "scenario analysis unavailable"])
    if c["vitems"]:
        section("How valuation was scored")
        table(["Component", "Score"], [[esc(n), tone(sc, f"{sc:+.2f}")] for n, sc in c["vitems"]])


def ai_panel(c, sym):
    section("AI-written review and independent opinion")
    key, pwd = D._secret("ANTHROPIC_API_KEY"), D._secret("APP_PASSWORD")
    ai = st.session_state.get(f"ai_{sym}")
    if not key:
        na("AI review not enabled", ["Add ANTHROPIC_API_KEY to this app's secrets to have Claude write a review from every metric shown here and give its own opinion."],
           "The API is a separate paid product from a Claude subscription. Everything else in the app is free.")
    else:
        ok_pw = True
        if pwd:
            st.text_input("Access password (protects your API credits)", type="password", key="pw")
            ok_pw = st.session_state.get("pw") == pwd
        else:
            md('<div class="note">No APP_PASSWORD set: anyone with this app\'s link can trigger paid AI calls. Set APP_PASSWORD in secrets and a spend limit in the Anthropic console.</div>', unsafe_allow_html=True)
        if st.button("Generate AI review", key=f"runai{sym}", disabled=not ok_pw or st.session_state.get("ai_calls", 0) >= 8):
            facts = E.facts_for_ai(c)
            try:
                with st.spinner("Writing review..."):
                    text = ai_call(Q.review_prompt(facts, c["info"].get("longBusinessSummary")), D._secret("ANTHROPIC_MODEL") or "claude-sonnet-5-5", key)
                p = Q.parse_review(text, facts)
                st.session_state[f"ai_{sym}"] = dict(items=p["items"], challenge=p["challenge"], review=p["review"], errors=p["errors"])
                st.session_state["ai_calls"] = st.session_state.get("ai_calls", 0) + 1
                st.rerun()
            except RuntimeError as e:
                st.error(f"AI review unavailable: {e}")
    if not ai or not ai.get("review"):
        if ai:
            for e_ in ai["errors"]: st.caption(e_)
        return
    r, op = ai["review"], ai["review"]["opinion"]
    md('<div class="note">Written by an AI model from the deterministic outputs on this site plus its general knowledge (labelled where used). It is unverified. Its opinion never changes the calculated rating; the judgment questions it scored feed the qualitative view (shown on this page), and a challenge applies only if you tick the box below.</div>', unsafe_allow_html=True)
    table(["", "Rating", "Notes"], [["Model (calculated)", chip(c["rating"]["rating"]), esc(f"score {c['rating']['score']:+.2f}, confidence {c['conf_label']}")],
                                    ["AI opinion", chip(op["rating"]) if op["rating"] else "n/a", esc(("agrees with the model" if op["agrees"] else "disagrees with the model" if op["agrees"] is False else "") + f"; confidence {op['confidence']:.0%}")]], left=(0, 2))
    md(op["reasoning"])
    for title, txt in (("Thesis", r["thesis"]), ("Business", r["business"]), ("Competitive position", r["moat"]["assessment"]), ("Industry and AI", r["industry_and_ai"]),
                       ("Management and capital allocation", r["management"]), ("What the price implies", r["expectations"]), ("What the market may be missing", r["missing"])):
        if txt: md(f"**{title}.** {esc(txt)}")
    a, b = st.columns(2)
    with a:
        md("**For**"); bullets([esc(x) for x in r["for_"]])
    with b:
        md("**Against**"); bullets([esc(x) for x in r["against"]])
    for k in ("bull", "base", "bear"):
        if r["scenarios"].get(k): md(f"**{k.title()} case.** {esc(r['scenarios'][k])}")
    if r["risks"]: md("**Thesis-breaking risks**"); bullets([esc(x) for x in r["risks"]])
    if r["monitor"]: md("**Monitor next quarter**"); bullets([esc(x) for x in r["monitor"]])
    if r["unknowns"]: md(f'<div class="note">Unknown to the reviewer: {esc("; ".join(r["unknowns"]))}</div>', unsafe_allow_html=True)
    if ai.get("challenge"):
        ch = ai["challenge"]
        md(f"**Challenge to the numbers ({ch['adjustment']:+.2f}):** {esc(ch['claim'])}  \nEvidence: {esc(ch['evidence'])}")
        st.checkbox("Apply this challenge to the rating (explicit override, at most 0.25)", key=f"ai_apply_{sym}")
    for e_ in ai["errors"]: st.caption(f"Response check: {e_}")


@st.cache_data(ttl=21600, show_spinner=False)
def ai_call(prompt, model, _key):
    text, err = Q.call_llm(prompt, _key, model, timeout=150, max_tokens=4500)
    if err: raise RuntimeError(err)
    return text


def page_research(c, extra):
    sym = c["info"]["symbol"]
    section("Ten business questions")
    table(["Question", "Answer", "Source"], [[esc(q["Question"]), esc(q["Answer"]), esc(q["Source"])] for q in c["questions"]], left=(0, 1, 2))
    section("Qualitative assessment")
    md(f'<div class="note">{sum(1 for i in c["qitems"].values() if i["score"] is not None)} of {len(c["qitems"])} questions have evidence (coverage {c["qcov"]:.0%}). Computed items come from reported statements; the rest need business knowledge and stay unassessed unless the AI review or your own judgment supplies them.</div>', unsafe_allow_html=True)
    table(["Question", "Score", "Confidence", "Evidence", "Source"], [[esc(i["label"]), tone(i["score"], f"{i['score']:+.2f}") if i["score"] is not None else '<span class="muted">not assessed</span>', f"{i['conf']:.0%}", esc(i["evidence"]), esc(i["source"])]
                                                                      for i in c["qitems"].values()], left=(0, 3, 4))
    if c["contradictions"] or c["flags"]:
        section("Flags")
        bullets([esc(f"{x[1]}: {x[2]}") for x in c["flags"]] + [esc(x) for x in c["contradictions"]])
    ai_panel(c, sym)


def page_scenarios(c, extra):
    s = c["scen"]
    section("Bull, base and bear scenarios")
    if not s["ok"]:
        na("A numerical scenario cannot be calculated reliably", [s["reason"] or "insufficient data"] + [f"Missing: {m}" for m in s.get("missing", [])], "Nothing was estimated. A qualitative view follows instead.")
        section("Qualitative scenarios")
        for k, v in c["qual_scen"].items():
            md(f"**{k}.** {v}")
        md('<div class="note">Qualitative judgment built from reported facts; no numbers are estimated.</div>', True)
        return
    mode = {"earnings": "Forward earnings x exit P/E", "revenue": "Revenue x exit EV/Sales (company is loss-making)", "book": "Book value x exit P/B"}[s["mode"]]
    md(f"Method: **{mode}**, {V.H}-year horizon, required return {c['hurdle']:.0%} per year.")
    if s.get("anchor"):
        a = s["anchor"]
        md(f'<div class="note">Exit multiple support: today\'s forward P/E {a["current"]:.1f}x; anchor {a["pe"]:.1f}x from {esc(a["source"])}. '
           + ("Company-specific evidence supports this assumption." if a["supported"] else "No company-specific evidence supports it, so treat the numbers as weakly supported.") + "</div>", True)
    keys = list(dict.fromkeys(k for v in s["rows"].values() for k in v["assump"]))
    section("Assumptions")
    table(["Assumption", "Bear", "Base", "Bull"], [[esc(k)] + [fa(k, s["rows"][case]["assump"][k]) if k in s["rows"][case]["assump"] else "n/a" for case in ("Bear", "Base", "Bull")] for k in keys])
    section("Outcomes")
    table(["", "Bear", "Base", "Bull"], [["Price at horizon"] + [f"${s['rows'][k]['price_h']:,.2f}" for k in ("Bear", "Base", "Bull")],
                                          ["Total return vs today"] + [tone(s["rows"][k]["total"], pct(s["rows"][k]["total"], 0)) for k in ("Bear", "Base", "Bull")],
                                          ["Annualized return"] + [tone(s["rows"][k]["ann"], pct(s["rows"][k]["ann"])) for k in ("Bear", "Base", "Bull")],
                                          ["Probability (estimate)"] + [pct0(s["rows"][k]["prob"], 0) for k in ("Bear", "Base", "Bull")]])
    md(f"Probability-weighted outcome: **{pct(s['ev_ann'])} per year** (${s['ev_price']:,.2f} at the horizon), against a {c['hurdle']:.0%} required return.")
    md(f'<div class="note">{esc(s["probs_note"])} The weighted figure is only as good as these estimates; the base case, not the average, is the central outcome. Current price ${c["price"]:,.2f}.</div>', True)
    st.bar_chart(pd.DataFrame({"Price at horizon": [s["rows"][k]["price_h"] for k in ("Bear", "Base", "Bull")]}, index=["Bear", "Base", "Bull"]), height=170)
    section("Limitations")
    bullets(s["limitations"])


def page_data(c, extra):
    section("Data sources")
    table(["Item", "Source", "Detail"], [[esc(a), esc(b), esc(d)] for a, b, d in extra["sources"]], left=(0, 1, 2))
    for x in extra["conflicts"]:
        md(f'<div class="na">{esc(x)}</div>', True)
    md(f'<div class="note">Retrieved {extra["fetched"]:%Y-%m-%d %H:%M} UTC. Yahoo Finance data is unofficial and delayed. SEC data covers US 10-K filers and is annual.</div>', True)
    cv = c["cov"]
    section("Coverage")
    md(f"**{len(cv['evaluated'])} of {cv['total']}** quantitative metrics evaluated; {len(cv['not_meaningful'])} not meaningful or not applicable for this company; {len(cv['unavailable'])} unavailable (not reported or too little history). "
       f"Qualitative rubric: {sum(1 for i in c['qitems'].values() if i['score'] is not None)} of {len(c['qitems'])} assessed.")
    with st.expander("Metric detail"):
        md("Not meaningful / not applicable: " + ("; ".join(f"{k} ({v})" for k, v in cv["not_meaningful"].items()) or "none"))
        md("Unavailable: " + (", ".join(cv["unavailable"]) or "none"))
    section("How companies are classified")
    md('<div class="note">Sector, business lines, maturity, growth, cyclicality, capital intensity, profitability and risk are assessed separately. The framework follows from profitability, growth, funding and business lines; a sector label is shown but does not select it. '
       'Cyclicality is measured from the company\'s own revenue and margin history (with the sector as a prior only where history is short) and is treated as a risk characteristic, not a verdict. '
       'Business lines are detected from the company description; segment financials are not available from free structured data, so diversified companies are analysed on consolidated figures and flagged.</div>', True)
    section("Rating methodology")
    table(["Rating", "Meaning"], [[chip(k), esc(RT.DEFS[k])] for k in ("Strong Buy", "Buy", "Hold", "Sell", "Avoid")], left=(0, 1))
    md('<div class="note">Score thresholds: above +0.45 Strong Buy, above +0.18 Buy, above -0.12 Hold, above -0.35 Sell, otherwise Avoid. One rating is computed once per analysis and every page shows it.</div>', True)
    table(["Framework"] + list(RT.DIMS), [[esc(v["label"])] + [f"{w:.0%}" for w in v["w"]] for v in RT.FRAMEWORKS.values()], left=(0,))
    md('<div class="note">Constraints that can cap a rating, each shown with its reason: required-return floor, expected-return ceiling, poor risk/reward, metrics-alone guard, Strong Buy gate, funding constraint, evidence-quality gate. '
       'Confidence is a label (High, Medium, Low) derived from data completeness, estimate availability, support for the exit multiple, classification confidence, scenario dispersion and qualitative evidence; a rating of Avoid or Strong Buy needs at least Medium. '
       'Losses are handled once (financial health and funding), not again in profitability, valuation and the verdict.</div>', True)
    section("Not evaluated")
    bullets(["Segment-level revenue, margins and growth (not in free structured data)", "Customer concentration, backlog, contracts (need filing text)", "Addressable-market size", "Management quality and moat type unless the AI review or your own view supplies them",
             "Consensus estimates and revisions beyond what Yahoo returns", "Options-implied volatility, credit ratings, insider-transaction analysis"])


def main():
    wq = sidebar()
    sym = search_bar()
    if not sym:
        landing(); return
    with st.spinner(f"Loading {sym}..."):
        info, hist, inc, cf, bs, extra = D.prepare(sym)
    if hist is None or hist.empty:
        section(sym)
        na(f"No data found for '{sym}'", ["The symbol may be misspelled or not listed (use BRK-B, not BRK.B).", "Yahoo Finance may be throttling requests; try again in a minute.", f"SEC status: {extra['sec_status']}"],
           "Check the ticker and choose Load again. Nothing was estimated.")
        return
    info["symbol"] = sym
    peers = D.load_peers(sym, info.get("industryKey")) if info.get("industryKey") else None
    try:
        c = E.analyze(info, hist, inc, cf, bs, est=extra["est"], peers=peers, ai=st.session_state.get(f"ai_{sym}"), w_qual=wq, apply_override=bool(st.session_state.get(f"ai_apply_{sym}")))
    except Exception as e:
        section(sym)
        na("The analysis could not be completed", [f"{type(e).__name__}: {e}"], "This is a bug or an unusual data shape; nothing was estimated.")
        return
    topbar(c)
    nav_top()
    {"Overview": page_overview, "Financials": page_financials, "Valuation": page_valuation, "Scenarios": page_scenarios, "Research": page_research, "Data & method": page_data}[st.session_state["page"]](c, extra)


main()
