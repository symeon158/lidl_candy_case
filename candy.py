"""
Lidl · Confectionery Range Analytics — Streamlit app
====================================================
Interactive dashboard for the Data & AI case study:
  • Plotly visuals (drivers, scatter, simulator)
  • Live predicted-preference + ROI calculator
  • Two OpenAI-powered features: "Ask the data" Q&A and an AI concept generator
  • Password-gated AI with a per-session call cap (protects the API bill)

Run locally:  streamlit run app.py
Deploy:       see README.md (Streamlit Community Cloud, free)
"""
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------- #
# Config & light theming
# ---------------------------------------------------------------- #
st.set_page_config(page_title="Lidl · Candy Analytics", page_icon="🍫", layout="wide")

BLUE, RED, YELLOW, GREY, INK = "#0050AA", "#E60A14", "#FFD400", "#B8B8B8", "#1A1A1A"
FAMCOL = {"Chocolate": BLUE, "Fruity": "#D98A00", "Other": GREY}

st.markdown("""
<style>
  .block-container {padding-top: 2.6rem; max-width: 1250px;}
  h1, h2, h3 {font-family: Georgia, 'Times New Roman', serif;}

  /* header bar — fills its container, can't overflow */
  .lidl-bar {background:linear-gradient(90deg,#0050AA,#0066cc); color:#fff;
             padding:15px 22px; border-radius:12px; display:flex; align-items:center;
             gap:13px; margin:0 0 6px 0; box-sizing:border-box; max-width:100%;
             box-shadow:0 2px 10px rgba(0,80,170,.18);}
  .lidl-bar .m {width:26px;height:26px;border-radius:6px;background:#FFD400;position:relative;flex:none;}
  .lidl-bar .m::after{content:"";position:absolute;inset:6px;border-radius:50%;
             background:#E60A14;box-shadow:0 0 0 2px #0050AA inset;}
  .sub {color:#5b6573; font-size:0.97rem; margin-top:2px;}

  /* metric cards */
  [data-testid="stMetric"]{
     background:#fff; border:1px solid #e8e8ec; border-left:4px solid #0050AA;
     border-radius:12px; padding:14px 16px 12px; min-height:122px;
     display:flex; flex-direction:column; gap:3px;
     box-shadow:0 1px 4px rgba(20,33,61,.07); transition:box-shadow .15s, transform .15s;}
  [data-testid="stMetric"]:hover{box-shadow:0 4px 14px rgba(20,33,61,.13); transform:translateY(-1px);}
  [data-testid="stMetricLabel"] p{font-size:.78rem; color:#5b6573; font-weight:600;
     text-transform:uppercase; letter-spacing:.03em;}
  [data-testid="stMetricValue"]{color:#0050AA; font-weight:700; font-size:2rem;}
  [data-testid="stMetricDelta"]{font-size:.8rem;}
</style>
""", unsafe_allow_html=True)

FLAGS = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat",
         "crispedricewafer", "hard", "bar", "pluribus"]
NUM = ["sugarpercent", "pricepercent"]
FEATURES = FLAGS + NUM
NICE = {"chocolate": "Chocolate", "fruity": "Fruity (gummy)", "caramel": "Caramel",
        "peanutyalmondy": "Peanut / almond", "nougat": "Nougat",
        "crispedricewafer": "Crispy wafer", "hard": "Hard candy", "bar": "Bar format",
        "pluribus": "Multi-pack", "sugarpercent": "Sweetness", "pricepercent": "Price"}
DATA_URL = ("https://raw.githubusercontent.com/fivethirtyeight/data/master/"
            "candy-power-ranking/candy-data.csv")


# ---------------------------------------------------------------- #
# Data + model (cached)
# ---------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_URL)
    df = df[df[FLAGS].sum(axis=1) > 0].reset_index(drop=True)   # drop the two coins
    df["family"] = np.where(df["chocolate"] == 1, "Chocolate",
                     np.where(df["fruity"] == 1, "Fruity", "Other"))
    df["format"] = np.where(df["bar"] == 1, "Bar",
                     np.where(df["pluribus"] == 1, "Multi-pack", "Single"))
    return df


@st.cache_data(show_spinner=False)
def fit_model():
    df = load_data()
    res = sm.OLS(df["winpercent"], sm.add_constant(df[FEATURES])).fit()
    return {"params": res.params.to_dict(),
            "pvalues": res.pvalues.to_dict(),
            "adj_r2": float(res.rsquared_adj)}


df = load_data()
MODEL = fit_model()
MKT = float(df["winpercent"].mean())
OBS_MAX = float(df["winpercent"].max())


@st.cache_data(show_spinner=False)
def load_artifacts():
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


ART = load_artifacts()



def predict(profile: dict) -> float:
    c = MODEL["params"]
    v = c["const"] + sum(c[k] * profile.get(k, 0) for k in FEATURES)
    return max(0.0, min(100.0, v))


def pct_of_market(v: float) -> int:
    return int(round((df["winpercent"] < v).mean() * 100))


# ---------------------------------------------------------------- #
# LLM helpers (OpenAI) — gated + capped
# ---------------------------------------------------------------- #
def get_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None


CALL_CAP = 20

def call_llm(messages, max_tokens=450, temperature=0.7, want_json=False):
    if st.session_state.get("llm_calls", 0) >= CALL_CAP:
        return None, f"Demo limit reached ({CALL_CAP} AI calls per session)."
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return None, "No OPENAI_API_KEY set. Add it in the app's Secrets."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        kwargs = dict(model="gpt-4o-mini", messages=messages,
                      max_tokens=max_tokens, temperature=temperature)
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        st.session_state["llm_calls"] = st.session_state.get("llm_calls", 0) + 1
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"AI error: {e}"


def summarize_chart(title, facts):
    sysmsg = ("You explain analytics charts to a non-technical Lidl Controlling audience in 2-3 short "
              "sentences. Be concrete, reference the numbers, and end with why it matters for the decision. "
              "No preamble, no bullet points.")
    user = f"Chart: {title}\nUnderlying numbers:\n{facts}\nSummarise what it shows and why it matters."
    return call_llm([{"role": "system", "content": sysmsg},
                     {"role": "user", "content": user}], temperature=0.3, max_tokens=220)


def ai_summary_button(key, title, facts):
    if not st.session_state.get("unlocked", False):
        st.caption("🔒 Enter the access code in the sidebar to summarise this chart with AI.")
        return
    if st.button("🤖 Summarise with AI", key=f"sum_{key}"):
        with st.spinner("Summarising…"):
            out, err = summarize_chart(title, facts)
        st.session_state[f"sumres_{key}"] = err and f"⚠️ {err}" or out
    if st.session_state.get(f"sumres_{key}"):
        st.info(st.session_state[f"sumres_{key}"])


def summarize_view(title, facts):
    sysmsg = ("You are a data analyst writing a short briefing for Lidl's Controlling team. From the "
              "numbers, write ONE tight paragraph (3-5 sentences) a manager could paste into a report: "
              "lead with the key takeaway, cite the most important figures, and end with one implication "
              "or caveat. No headings, no bullet points, no preamble.")
    user = f"Topic: {title}\nNumbers:\n{facts}\nWrite the briefing paragraph."
    return call_llm([{"role": "system", "content": sysmsg},
                     {"role": "user", "content": user}], temperature=0.4, max_tokens=320)


def ai_view_summary(key, title, facts):
    if not st.session_state.get("unlocked", False):
        st.caption("🔒 Enter the access code in the sidebar to generate an AI summary of this view.")
        return
    if st.button("🤖 Generate AI summary of this view", key=f"vsum_{key}"):
        with st.spinner("Writing summary…"):
            out, err = summarize_view(title, facts)
        st.session_state[f"vsumres_{key}"] = err and f"⚠️ {err}" or out
    if st.session_state.get(f"vsumres_{key}"):
        st.info(st.session_state[f"vsumres_{key}"])


def data_context() -> str:
    """Grounding facts so the LLM answers from THIS analysis, not its priors."""
    c, p = MODEL["params"], MODEL["pvalues"]
    drivers = [f"{NICE[k]} {c[k]:+.1f} pts (p={p[k]:.3f}{'*' if p[k] < 0.05 else ''})"
               for k in FEATURES]
    top = df.sort_values("winpercent", ascending=False).head(6)
    top_s = "; ".join(f"{r.competitorname} {r.winpercent:.0f}" for r in top.itertuples())
    return (
        f"Dataset: {len(df)} candies from a head-to-head preference survey. "
        f"Target = winpercent (share of taste match-ups won). Market average = {MKT:.1f}.\n"
        f"OLS adj R² = {MODEL['adj_r2']:.2f}. Coefficients (win-% points, *=significant p<0.05): "
        f"{', '.join(drivers)}.\n"
        f"Top products: {top_s}.\n"
        "Recommendation: a chocolate + peanut + crispy-wafer bar (predicted ~78, beats ~96% of market); "
        "a gummy predicts ~47, below average. Chocolate beats non-chocolate gummies by ~17 pts on average. "
        "Note: with only 83 rows the linear model is the headline; trees were tested but tie within noise. "
        "ROI figures are illustrative (assumed stores/price/margin), the logic is the deliverable."
    )


# ---------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🍫 Lidl Candy Analytics")
    st.caption("Data & AI · Controlling — case study demo")
    st.divider()
    st.markdown("**Filters** (Overview & Drivers)")
    fam_sel = st.multiselect("Flavour family", ["Chocolate", "Fruity", "Other"], default=[])
    fmt_sel = st.multiselect("Format", ["Bar", "Multi-pack", "Single"], default=[])
    st.divider()
    st.markdown("**🔓 AI features**")
    app_pw = get_secret("APP_PASSWORD")
    if not app_pw:
        st.session_state["unlocked"] = True
        st.caption("No password set — AI features open.")
    else:
        code = st.text_input("Access code", type="password",
                             help="Unlocks the AI features (protects the API budget).")
        st.session_state["unlocked"] = (code == app_pw)
        st.caption("✅ Unlocked" if st.session_state["unlocked"] else "Enter code to enable AI.")
    used = st.session_state.get("llm_calls", 0)
    st.progress(min(used / CALL_CAP, 1.0), text=f"AI calls used: {used}/{CALL_CAP}")

dff = df.copy()
if fam_sel:
    dff = dff[dff["family"].isin(fam_sel)]
if fmt_sel:
    dff = dff[dff["format"].isin(fmt_sel)]


def ai_enabled():
    return st.session_state.get("unlocked", False)


# ---------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------- #
st.markdown(
    '<div style="display:flex;align-items:center;gap:12px;margin:6px 0 0;line-height:1.4;">'
    '<span style="width:26px;height:26px;border-radius:6px;background:#FFD400;'
    'position:relative;display:inline-block;flex:none;">'
    '<span style="position:absolute;inset:6px;border-radius:50%;background:#E60A14;'
    'box-shadow:0 0 0 2px #0050AA inset;display:block;"></span></span>'
    '<span style="font-family:Georgia,serif;font-size:1.5rem;font-weight:700;color:#0050AA;'
    'line-height:1.5;">Confectionery Range Analytics</span></div>',
    unsafe_allow_html=True)
st.markdown('<div style="height:4px;border-radius:2px;margin:9px 0 2px;'
            'background:linear-gradient(90deg,#0050AA 0%,#0050AA 55%,#E60A14 80%,#FFD400 100%);">'
            '</div>', unsafe_allow_html=True)
st.markdown('<p class="sub">Which new own-brand candy should we launch — and what is it worth? '
            'A data-driven answer, from raw preferences to euros.</p>', unsafe_allow_html=True)

tab1, tab5, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview","📈 Insights", "🔍 Driver Analysis", "🧪 Concept Simulator & ROI",
      "💬 Ask the Data"])

# =================================================================
# TAB 1 — OVERVIEW
# =================================================================
with tab1:
    if len(dff) == 0:
        st.warning("No products match the current filters.")
    else:
        avg = dff["winpercent"].mean()
        choc = dff[dff["chocolate"] == 1]["winpercent"].mean()
        gum = dff[(dff["fruity"] == 1) & (dff["chocolate"] == 0)]["winpercent"].mean()
        best = dff.sort_values("winpercent", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Products in view", len(dff))
        c2.metric("Avg win-%", f"{avg:.1f}", f"{avg - MKT:+.1f} vs market")
        c3.metric("Chocolate − gummy gap",
                  f"{(choc - gum):+.1f}" if pd.notna(choc) and pd.notna(gum) else "—")
        c4.metric(f"🏆 {best['competitorname'][:20]}", f"{best['winpercent']:.0f}", "win-%")

        col_l, col_r = st.columns([1.15, 0.85])
        with col_l:
            st.markdown("**Strongest preference drivers (in current view)**")
            gaps = []
            for k in FLAGS:
                w = dff.loc[dff[k] == 1, "winpercent"]
                wo = dff.loc[dff[k] == 0, "winpercent"]
                if 0 < len(w) < len(dff):
                    gaps.append((NICE[k], w.mean() - wo.mean()))
            g = pd.DataFrame(gaps, columns=["feature", "gap"]).sort_values("gap")
            fig = go.Figure(go.Bar(
                x=g["gap"], y=g["feature"], orientation="h",
                marker_color=[RED if v < 0 else BLUE for v in g["gap"]],
                text=[f"{v:+.1f}" for v in g["gap"]], textposition="outside"))
            fig.update_layout(height=330, margin=dict(l=10, r=20, t=10, b=10),
                              xaxis_title="avg win-% (with − without)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with col_r:
            st.markdown("**Average win-% by flavour family**")
            fa = dff.groupby("family")["winpercent"].mean().reindex(["Chocolate", "Fruity", "Other"]).dropna()
            fig = go.Figure(go.Bar(x=fa.index, y=fa.values,
                                   marker_color=[FAMCOL[i] for i in fa.index],
                                   text=[f"{v:.0f}" for v in fa.values], textposition="outside"))
            fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                              yaxis_title="win-%", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Top products by customer preference**")
        top = dff.sort_values("winpercent", ascending=False).head(10)
        fig = go.Figure(go.Bar(x=top["winpercent"], y=top["competitorname"], orientation="h",
                               marker_color=[FAMCOL[f] for f in top["family"]],
                               text=[f"{v:.0f}" for v in top["winpercent"]], textposition="outside"))
        fig.update_layout(height=360, margin=dict(l=10, r=20, t=10, b=10),
                          xaxis_title="win-%", template="plotly_white",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.markdown("##### 🤖 AI summary of this view")
        _drv = g.sort_values("gap", ascending=False)
        _gap = (f"{(choc - gum):+.1f} pts" if pd.notna(choc) and pd.notna(gum)
                else "n/a (one group absent in view)")
        _facts = (
            f"Filters — flavour: {', '.join(fam_sel) or 'all'}; format: {', '.join(fmt_sel) or 'all'}. "
            f"Products in view: {len(dff)}. Average win-% {avg:.1f} ({avg - MKT:+.1f} vs market {MKT:.0f}). "
            f"Chocolate vs non-chocolate-gummy gap: {_gap}. "
            "Feature lift (avg win-% with minus without): "
            + "; ".join(f"{r.feature} {r.gap:+.1f}" for r in _drv.itertuples()) + ". "
            "Avg win-% by family: " + "; ".join(f"{i} {v:.0f}" for i, v in fa.items()) + ". "
            "Top products: "
            + "; ".join(f"{r.competitorname} {r.winpercent:.0f}" for r in top.head(3).itertuples()) + "."
        )
        ai_view_summary(f"ov_{'-'.join(sorted(fam_sel))}_{'-'.join(sorted(fmt_sel))}",
                        "Overview of the candy range for the selected filters", _facts)

# =================================================================
# TAB 2 — DRIVER ANALYSIS
# =================================================================
with tab2:
    st.markdown("**Sweetness vs. preference** — bubble size = price percentile, colour = flavour family")
    fig = px.scatter(dff, x="sugarpercent", y="winpercent", size="pricepercent",
                     color="family", color_discrete_map=FAMCOL,
                     hover_name="competitorname", size_max=22,
                     labels={"sugarpercent": "sweetness percentile", "winpercent": "win-%"})
    fig.add_hline(y=MKT, line_dash="dash", line_color="grey",
                  annotation_text=f"market avg {MKT:.0f}", annotation_position="top right")
    fig.update_layout(height=430, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Model driver strength** — win-% points per feature, all else equal (OLS). "
                f"Adjusted R² = **{MODEL['adj_r2']:.2f}**.")
    c, p = MODEL["params"], MODEL["pvalues"]
    rows = [(NICE[k], c[k], p[k]) for k in FEATURES]
    cd = pd.DataFrame(rows, columns=["feature", "coef", "p"]).sort_values("coef")
    fig = go.Figure(go.Bar(
        x=cd["coef"], y=cd["feature"], orientation="h",
        marker_color=[RED if v < 0 else BLUE for v in cd["coef"]],
        text=[f"{v:+.1f}{'*' if pp < 0.05 else ''}" for v, pp in zip(cd["coef"], cd["p"])],
        textposition="outside",
        customdata=cd["p"], hovertemplate="%{y}: %{x:+.1f} pts<br>p=%{customdata:.3f}<extra></extra>"))
    fig.update_layout(height=380, template="plotly_white", margin=dict(l=10, r=20, t=10, b=10),
                      xaxis_title="win-% points  (* = significant, p<0.05)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This visual is model output and does not respond to the sidebar filters.")

    st.divider()
    st.markdown("##### 🤖 AI summary")
    _pos = "; ".join(f"{r.feature} {r.coef:+.1f}" for r in cd.sort_values("coef", ascending=False).itertuples()
                     if r.p < 0.05 and r.coef > 0)
    _negns = "; ".join(f"{r.feature} {r.coef:+.1f} (p={r.p:.2f})" for r in cd.itertuples() if r.coef < 0)
    _facts = (
        f"OLS regression, adjusted R²={MODEL['adj_r2']:.2f} on {len(df)} candies. "
        f"Statistically significant positive drivers (p<0.05): {_pos}. "
        f"Negative coefficients: {_negns}. "
        "Scatter (sweetness vs win-%): chocolate-based products sit mostly above the market average, "
        "non-chocolate below; sweeter products trend higher; bubble size (price) shows price is not a clear driver. "
        "Caveat: with 83 rows, marginal features (p≈0.10, e.g. crispy wafer, hard) are suggestive, not proven."
    )
    ai_view_summary("drivers", "What drives candy preference (regression model)", _facts)

# =================================================================
# TAB 3 — CONCEPT SIMULATOR & ROI
# =================================================================
with tab3:
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Build a product")
        cc = st.columns(2)
        prof = {}
        labels_pos = ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat", "crispedricewafer"]
        defaults = {"chocolate": True, "peanutyalmondy": True, "crispedricewafer": True, "bar": True}
        for i, k in enumerate(labels_pos):
            prof[k] = 1 if cc[i % 2].checkbox(NICE[k], value=defaults.get(k, False), key=f"t_{k}") else 0
        cc2 = st.columns(3)
        prof["bar"] = 1 if cc2[0].checkbox("Bar", value=True, key="t_bar") else 0
        prof["pluribus"] = 1 if cc2[1].checkbox("Multi-pack", value=False, key="t_pl") else 0
        prof["hard"] = 1 if cc2[2].checkbox("Hard", value=False, key="t_hard") else 0
        prof["sugarpercent"] = st.slider("Sweetness percentile", 0, 100, 70) / 100
        prof["pricepercent"] = st.slider("Price percentile", 0, 100, 45) / 100

    pred = predict(prof)
    beats = pct_of_market(pred)
    with right:
        st.markdown("#### Predicted customer preference")
        gcol = RED if pred >= 70 else (BLUE if pred >= MKT else "#8a8886")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=round(pred, 1),
            number={"suffix": " win-%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": gcol},
                   "threshold": {"line": {"color": INK, "width": 3}, "value": MKT}}))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        verdict = ("🏆 Top-tier — launch candidate" if pred >= 70
                   else "👍 Above market average" if pred >= MKT else "👎 Below average")
        n_major = sum(prof.get(k, 0) for k in
                      ["chocolate", "fruity", "caramel", "peanutyalmondy", "nougat", "crispedricewafer"])
        both_cf = prof.get("chocolate") and prof.get("fruity")
        extrapolating = (pred > OBS_MAX) or (n_major >= 4) or both_cf
        if extrapolating:
            msg = ("⚠️ **Unusual combination.** This mixes features that rarely co-occur in real products"
                   + (" — almost no candy is both chocolate **and** fruity" if both_cf else "")
                   + f". The model is extrapolating beyond the {len(df)} real candies "
                   f"(highest observed is {OBS_MAX:.0f}), so treat this number as **indicative only**, "
                   "not a reliable prediction.")
            st.warning(msg)
            st.caption("Tip: realistic 3–4 ingredient bars (e.g. chocolate + peanut + wafer) stay within "
                       "the data and give trustworthy predictions.")
        else:
            st.markdown(f"**{verdict}** · beats ~**{beats}%** of products on the market "
                        f"(dashed line = market avg {MKT:.0f}).")

    st.divider()
    st.markdown("#### Business case  *(illustrative — swap in Lidl category data)*")
    a = st.columns(5)
    stores = a[0].slider("Stores", 50, 400, 230, 10)
    price = a[1].slider("Retail € ", 0.5, 3.0, 1.29, 0.05)
    margin = a[2].slider("Margin %", 10, 45, 32) / 100
    base = a[3].slider("Units/store/wk", 10, 60, 35)
    invest = a[4].slider("Launch €k", 50, 400, 180, 10) * 1000

    def gp(win):
        return base * (win / MKT) * stores * 52 * price * margin
    gp_rec, gp_gum = gp(pred), gp(predict(dict(fruity=1, pluribus=1, sugarpercent=.7, pricepercent=.45)))
    pay = invest / gp_rec * 12 if gp_rec else 0
    roi3 = (gp_rec * 3 - invest) / invest * 100 if invest else 0

    m = st.columns(4)
    m[0].metric("Annual gross profit", f"€{gp_rec:,.0f}")
    m[1].metric("Payback", f"{pay:.1f} mo")
    m[2].metric("3-year ROI", f"{roi3:.0f}%")
    m[3].metric("Value vs. gummy / yr", f"€{gp_rec - gp_gum:,.0f}")

    st.divider()
    st.markdown("#### 🤖 AI product concept")
    if not ai_enabled():
        st.info("Enter the access code in the sidebar to generate a concept.")
    else:
        if st.button("✨ Generate concept for this product", type="primary"):
            on = [NICE[k] for k in FLAGS if prof.get(k)]
            user = (f"Profile: {', '.join(on)}; sweetness {int(prof['sugarpercent']*100)}/100; "
                    f"price {int(prof['pricepercent']*100)}/100. Predicted preference {pred:.0f}% "
                    f"(beats {beats}% of market). Create ONE supermarket own-brand candy concept. "
                    'Return JSON with keys: name, tagline (<=8 words), description (2 sentences), '
                    "why_it_works (1 sentence).")
            with st.spinner("Generating…"):
                out, err = call_llm(
                    [{"role": "system", "content": "You are a concise confectionery brand strategist."},
                     {"role": "user", "content": user}], want_json=True)
            if err:
                st.error(err)
            else:
                try:
                    st.session_state["concept"] = json.loads(out)
                except Exception:
                    st.session_state["concept"] = {"name": "", "description": out}
        if "concept" in st.session_state:
            c = st.session_state["concept"]
            st.success(f"**{c.get('name','')}** — *{c.get('tagline','')}*")
            st.write(c.get("description", ""))
            if c.get("why_it_works"):
                st.caption("Why it works: " + c["why_it_works"])

# =================================================================
# TAB 5 — INSIGHTS (notebook visuals, in Plotly)
# =================================================================
with tab5:
    st.markdown("These are the analytical visuals from the notebook, rendered interactively. "
                "Heavy models (SHAP, the cross-validation bake-off) are **precomputed** and loaded as "
                "artifacts — in production you load cached model outputs, not retrain on every page view.")
    if ART is None:
        st.error("precomputed.json not found — commit it alongside app.py.")
    else:
        # A. distribution
        st.markdown("##### Distribution of customer preference")
        fig = go.Figure(go.Histogram(x=df["winpercent"], nbinsx=18, marker_color=BLUE))
        fig.add_vline(x=MKT, line_dash="dash", line_color=RED, annotation_text=f"mean {MKT:.0f}")
        fig.update_layout(height=300, template="plotly_white", xaxis_title="win-%", yaxis_title="count",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        ai_summary_button("hist", "Distribution of candy win-%",
                          f"n={len(df)}, min={df.winpercent.min():.0f}, max={df.winpercent.max():.0f}, "
                          f"mean={MKT:.1f}, median={df.winpercent.median():.0f}. Spread is wide (not bunched).")
        st.divider()

        # B. model selection
        st.markdown("##### Model selection — repeated cross-validation (do we need XGBoost?)")
        cv = ART["cv"]
        order = sorted(cv, key=lambda k: np.mean(cv[k]))
        fig = go.Figure()
        for name in order:
            fig.add_trace(go.Box(x=cv[name], name=name, orientation="h",
                                 marker_color=BLUE, boxmean=True, line_width=1.4))
        fig.add_vline(x=max(np.mean(v) for v in cv.values()), line_dash="dash", line_color=RED)
        fig.update_layout(height=330, template="plotly_white", showlegend=False,
                          xaxis_title="cross-validated R²  (50 folds; higher = better)",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        means = {k: float(np.mean(v)) for k, v in cv.items()}
        stds = {k: float(np.std(v)) for k, v in cv.items()}
        ai_summary_button("cv", "Model comparison via repeated cross-validation",
                          "Mean R2 ± std per model: " +
                          "; ".join(f"{k} {means[k]:.2f}±{stds[k]:.2f}" for k in order) +
                          ". Gaps between means (~0.08) are smaller than the spread (~0.25) — "
                          "models are statistically indistinguishable on 83 rows.")
        st.divider()

        # C. SHAP beeswarm
        st.markdown("##### Explainable AI — SHAP (which features move predictions, across all candies)")
        bee, ma = ART["shap_beeswarm"], ART["shap_mean_abs"]
        feats_order = sorted(ma, key=lambda k: ma[k])[-8:]
        xs, ys, cs = [], [], []
        for i, feat in enumerate(feats_order):
            s, fv = bee[feat]["shap"], bee[feat]["fval"]
            jit = (np.random.RandomState(i).rand(len(s)) - 0.5) * 0.4
            xs += s; ys += [i + j for j in jit]; cs += fv
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color=cs, colorscale="Bluered", size=6, opacity=0.7,
                        colorbar=dict(title="feature<br>value", tickvals=[0, 1], ticktext=["low", "high"]))))
        fig.add_vline(x=0, line_color="grey")
        fig.update_layout(height=380, template="plotly_white", margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="SHAP value (impact on predicted win-%)",
                          yaxis=dict(tickvals=list(range(len(feats_order))), ticktext=feats_order))
        st.plotly_chart(fig, use_container_width=True)
        ai_summary_button("bee", "SHAP beeswarm (global feature impact)",
                          "Mean |SHAP| per feature: " +
                          "; ".join(f"{k} {ma[k]}" for k in sorted(ma, key=lambda k: -ma[k])[:6]) +
                          ". Red = feature present/high, blue = absent/low; x = push on prediction.")
        st.divider()

        # D. SHAP waterfall
        st.markdown("##### Why the recommended bar scores high — SHAP waterfall")
        wf = ART["shap_waterfall"]
        cons = wf["contribs"][:7]
        measures = ["absolute"] + ["relative"] * len(cons) + ["total"]
        xlab = ["Average candy"] + [c["feature"] for c in cons] + ["Predicted"]
        yval = [wf["base"]] + [c["shap"] for c in cons] + [wf["final"]]
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=measures, x=xlab, y=yval,
            text=[f"{v:+.1f}" if m == "relative" else f"{v:.0f}" for v, m in zip(yval, measures)],
            textposition="outside",
            increasing=dict(marker_color=RED), decreasing=dict(marker_color=BLUE),
            totals=dict(marker_color=INK)))
        fig.update_layout(height=360, template="plotly_white", margin=dict(l=10, r=10, t=20, b=10),
                          yaxis_title="predicted win-%")
        st.plotly_chart(fig, use_container_width=True)
        ai_summary_button("wf", "SHAP waterfall for the recommended product",
                          f"Starts at average candy {wf['base']}, ends at {wf['final']}. "
                          "Contributions: " + "; ".join(f"{c['feature']} {c['shap']:+.1f}" for c in cons) + ".")
        st.divider()

        # E. concept comparison
        st.markdown("##### Predicted preference by concept")
        S, P = 0.70, 0.45
        concepts = {
            "Gummy": dict(fruity=1, pluribus=1, sugarpercent=S, pricepercent=P),
            "Plain chocolate bar": dict(chocolate=1, bar=1, sugarpercent=S, pricepercent=P),
            "Chocolate + caramel": dict(chocolate=1, caramel=1, bar=1, sugarpercent=S, pricepercent=P),
            "Choc + peanut + wafer": dict(chocolate=1, peanutyalmondy=1, crispedricewafer=1,
                                          bar=1, sugarpercent=S, pricepercent=P),
        }
        preds = {k: predict(v) for k, v in concepts.items()}
        fig = go.Figure(go.Bar(x=list(preds), y=list(preds.values()),
                               marker_color=[GREY, BLUE, BLUE, RED],
                               text=[f"{v:.0f}" for v in preds.values()], textposition="outside"))
        fig.add_hline(y=MKT, line_dash="dash", line_color=INK, annotation_text=f"market avg {MKT:.0f}")
        fig.update_layout(height=320, template="plotly_white", yaxis_title="predicted win-%",
                          yaxis_range=[0, 90], margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        ai_summary_button("concepts", "Predicted preference by concept",
                          "; ".join(f"{k} {v:.0f}" for k, v in preds.items()) + f"; market avg {MKT:.0f}.")
        st.divider()

        # F. ROI sensitivity
        st.markdown("##### ROI robustness — does the conclusion survive different assumptions?")
        rec_p = predict(concepts["Choc + peanut + wafer"])
        gum_p = predict(concepts["Gummy"])
        upw = list(range(20, 51, 2))
        gpf = lambda win, b: b * (win / MKT) * 230 * 52 * 1.29 * 0.32 / 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=upw, y=[gpf(rec_p, b) for b in upw], mode="lines+markers",
                                 name="Recommended", line=dict(color=RED, width=3)))
        fig.add_trace(go.Scatter(x=upw, y=[gpf(gum_p, b) for b in upw], mode="lines+markers",
                                 name="Gummy", line=dict(color=GREY, width=3)))
        fig.update_layout(height=320, template="plotly_white",
                          xaxis_title="assumed base units / store / week",
                          yaxis_title="annual gross profit (€000s)", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        ai_summary_button("roi", "ROI sensitivity to the sales assumption",
                          f"Recommended predicted win {rec_p:.0f} vs gummy {gum_p:.0f}. Recommended out-earns "
                          "the gummy across the whole 20–50 units/store/week range — conclusion is robust.")


# =================================================================
# TAB 4 — ASK THE DATA
# =================================================================
with tab4:
    st.markdown("#### 💬 Ask the data")
    st.caption("Natural-language questions answered from *this* analysis — grounded in the results, "
               "not the model's general knowledge.")
    if not ai_enabled():
        st.info("Enter the access code in the sidebar to enable Q&A.")
    else:
        examples = ["Why is chocolate better than fruity?",
                    "Should we worry about the gummy option?",
                    "What's the single most important feature, and how sure are we?",
                    "Explain the ROI to a non-technical manager."]
        ex = st.selectbox("Try an example, or type your own below:", [""] + examples)
        q = st.text_input("Your question", value=ex or "")
        if st.button("Ask", type="primary") and q.strip():
            sys = ("You are a data analyst presenting a candy-range study to Lidl's Controlling team. "
                   "Answer ONLY from the CONTEXT. Be concise, business-friendly, and honest about limits. "
                   "If the answer isn't in the context, say so.\n\nCONTEXT:\n" + data_context())
            with st.spinner("Thinking…"):
                out, err = call_llm(
                    [{"role": "system", "content": sys}, {"role": "user", "content": q}],
                    temperature=0.3, max_tokens=400)
            if err:
                st.error(err)
            else:
                st.markdown(out)
        with st.expander("What facts does the assistant see?"):
            st.code(data_context())

st.divider()
st.caption("FiveThirtyEight candy-power-ranking (CC BY 4.0) · OLS model, n=83 · "
           "ROI illustrative. Built with Streamlit + Plotly + OpenAI for the Lidl Data & AI case.")
