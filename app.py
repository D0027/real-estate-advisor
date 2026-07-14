import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLF_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_classification_pipeline.pkl")
REG_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_regression_pipeline.pkl")
META_PATH = os.path.join(BASE_DIR, "models", "model_metadata.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_real_estate_data.csv")

# ------------------------------------------------------------------
# CUSTOM CSS — glassmorphism + animations
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .hero-title { font-family: 'Space Grotesk', sans-serif; }

/* ---------- animated multi-layer background ---------- */
.stApp {
    background:
        radial-gradient(900px circle at 8% 8%, rgba(0,212,180,0.16), transparent 45%),
        radial-gradient(900px circle at 92% 15%, rgba(108,140,255,0.14), transparent 45%),
        radial-gradient(800px circle at 50% 100%, rgba(192,132,252,0.10), transparent 50%),
        linear-gradient(135deg, #0A0D13 0%, #0E121B 45%, #0A0D13 100%);
    background-size: 160% 160%, 160% 160%, 160% 160%, 200% 200%;
    animation: gradientShift 22s ease infinite;
}
@keyframes gradientShift {
    0%   { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 50%; }
    50%  { background-position: 20% 30%, 70% 40%, 40% 70%, 100% 50%; }
    100% { background-position: 0% 0%, 100% 0%, 50% 100%, 0% 50%; }
}

/* floating orb accents (decorative, layered behind content) */
.orb {
    position: fixed; border-radius: 50%; filter: blur(70px);
    pointer-events: none; z-index: 0; opacity: 0.5;
}
.orb-1 { width: 340px; height: 340px; top: -80px; left: -80px; background: #00D4B4; animation: floatOrb 14s ease-in-out infinite; }
.orb-2 { width: 300px; height: 300px; bottom: -100px; right: -60px; background: #6C8CFF; animation: floatOrb 18s ease-in-out infinite reverse; }
@keyframes floatOrb {
    0%, 100% { transform: translate(0,0) scale(1); }
    50% { transform: translate(30px,-25px) scale(1.12); }
}

@keyframes fadeRise {
    0% { opacity: 0; transform: translateY(18px); }
    100% { opacity: 1; transform: translateY(0); }
}
.block-container { animation: fadeRise 0.65s cubic-bezier(.22,1,.36,1); position: relative; z-index: 1; }

/* ---------- Hero ---------- */
.hero-wrap {
    padding: 2.6rem 2.6rem;
    border-radius: 26px;
    background: linear-gradient(120deg, rgba(0,212,180,0.16), rgba(108,140,255,0.10) 55%, rgba(192,132,252,0.10));
    border: 1px solid rgba(255,255,255,0.09);
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
}
.hero-wrap::before {
    content: "";
    position: absolute; inset: -2px;
    background: linear-gradient(90deg, transparent, rgba(0,212,180,0.5), transparent);
    height: 2px; top: 0; animation: shimmerLine 3.5s linear infinite;
}
@keyframes shimmerLine {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -0.01em;
    background: linear-gradient(90deg, #00E8C4, #6C8CFF 55%, #C084FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-size: 200% auto;
    animation: shineText 6s ease-in-out infinite;
}
@keyframes shineText {
    0%, 100% { background-position: 0% center; }
    50% { background-position: 100% center; }
}
.hero-sub { color: #A6B0BE; font-size: 1.05rem; margin-top: 0.55rem; position: relative; max-width: 640px; }

.chip-row { display: flex; gap: 0.6rem; margin-top: 1.1rem; flex-wrap: wrap; position: relative; }
.chip {
    padding: 0.4rem 0.9rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600;
    background: rgba(255,255,255,0.055); border: 1px solid rgba(255,255,255,0.1);
    color: #C7D0DA; backdrop-filter: blur(6px);
    transition: all 0.25s ease;
}
.chip:hover { border-color: rgba(0,212,180,0.55); color: #00E8C4; transform: translateY(-2px); }

/* ---------- glass cards ---------- */
.glass-card {
    background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    transition: transform 0.3s cubic-bezier(.22,1,.36,1), border-color 0.3s ease, box-shadow 0.3s ease;
    position: relative;
}
.glass-card::after {
    content: ""; position: absolute; inset: 0; border-radius: 20px; padding: 1px;
    background: linear-gradient(135deg, rgba(0,212,180,0.35), transparent 40%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor; mask-composite: exclude;
    opacity: 0; transition: opacity 0.3s ease;
}
.glass-card:hover { transform: translateY(-5px); box-shadow: 0 14px 36px rgba(0,212,180,0.14); }
.glass-card:hover::after { opacity: 1; }

/* ---------- metric tiles ---------- */
.metric-tile {
    text-align: center;
    padding: 1.25rem 0.7rem;
    border-radius: 18px;
    background: linear-gradient(160deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.08);
    animation: fadeRise 0.7s cubic-bezier(.22,1,.36,1);
    transition: transform 0.25s ease, border-color 0.25s ease;
}
.metric-tile:hover { transform: translateY(-3px) scale(1.015); border-color: rgba(0,212,180,0.4); }
.metric-tile .val {
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(90deg, #00E8C4, #6FE3FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-tile .lbl { font-size: 0.76rem; color: #8E98A6; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; font-weight: 600; }

/* ---------- verdict badges ---------- */
.badge-good {
    display:inline-block; padding: 0.65rem 1.5rem; border-radius: 999px;
    background: linear-gradient(90deg, #00D4B4, #17E3A0);
    color:#00201A; font-weight:800; font-size:1.1rem; letter-spacing: -0.01em;
    animation: popIn 0.55s cubic-bezier(.34,1.56,.64,1), glowPulse 2.4s ease-in-out infinite 0.6s;
}
.badge-bad {
    display:inline-block; padding: 0.65rem 1.5rem; border-radius: 999px;
    background: linear-gradient(90deg, #FF6B6B, #FF9472);
    color:#2A0000; font-weight:800; font-size:1.1rem; letter-spacing: -0.01em;
    animation: popIn 0.55s cubic-bezier(.34,1.56,.64,1);
    box-shadow: 0 0 26px rgba(255,107,107,0.42);
}
@keyframes popIn {
    0% { opacity:0; transform: scale(0.6) translateY(6px); }
    100% { opacity:1; transform: scale(1) translateY(0); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 22px rgba(0,212,180,0.35); }
    50% { box-shadow: 0 0 38px rgba(0,212,180,0.65); }
}

/* ---------- buttons ---------- */
.stButton>button {
    background: linear-gradient(90deg, #00D4B4, #17B8E3);
    background-size: 200% auto;
    color: #04140F; font-weight: 800; border: none; border-radius: 14px;
    padding: 0.7rem 1.5rem; transition: all 0.3s cubic-bezier(.22,1,.36,1);
    box-shadow: 0 6px 22px rgba(0,212,180,0.28);
    letter-spacing: 0.01em;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.015);
    background-position: right center;
    box-shadow: 0 12px 34px rgba(0,212,180,0.45);
}
.stButton>button:active { transform: translateY(-1px) scale(0.99); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C0F17, #0F131C);
    border-right: 1px solid rgba(255,255,255,0.07);
}
section[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.03); border-radius: 12px; padding: 0.6rem 0.8rem;
    border: 1px solid rgba(255,255,255,0.06); margin-bottom: 0.4rem;
}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(255,255,255,0.02); padding: 6px; border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    padding: 10px 20px;
    color: #8E98A6;
    font-weight: 600;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover { color: #C7D0DA; background: rgba(255,255,255,0.04); }
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, rgba(0,212,180,0.18), rgba(108,140,255,0.14)) !important;
    color: #00E8C4 !important;
    box-shadow: inset 0 0 0 1px rgba(0,212,180,0.3);
}

/* ---------- misc ---------- */
hr { border-color: rgba(255,255,255,0.08); }
::-webkit-scrollbar { width: 9px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #00D4B4, #6C8CFF); border-radius: 8px; }
[data-testid="stForm"] {
    border: 1px solid rgba(255,255,255,0.08); border-radius: 20px; padding: 1.4rem 1.6rem;
    background: rgba(255,255,255,0.02);
}
[data-testid="stMetricValue"] { color: #00E8C4; }
</style>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    clf = joblib.load(CLF_MODEL_PATH) if os.path.exists(CLF_MODEL_PATH) else None
    reg = joblib.load(REG_MODEL_PATH) if os.path.exists(REG_MODEL_PATH) else None
    meta = joblib.load(META_PATH) if os.path.exists(META_PATH) else None
    return clf, reg, meta


@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


clf_pipeline, reg_pipeline, meta = load_models()
df = load_data()

MODELS_READY = clf_pipeline is not None and reg_pipeline is not None and meta is not None

# ------------------------------------------------------------------
# HERO
# ------------------------------------------------------------------
hero_stats = ""
if df is not None:
    hero_stats = f"""
    <div class="chip-row">
        <div class="chip">🏘️ {len(df):,} Properties</div>
        <div class="chip">📍 {df['State'].nunique()} States</div>
        <div class="chip">🏙️ {df['City'].nunique()} Cities</div>
        <div class="chip">🤖 AI-Powered Predictions</div>
    </div>
    """

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">🏠 Real Estate Investment Advisor</div>
    <div class="hero-sub">AI-powered property intelligence — predict investment quality and 5-year value growth in seconds.</div>
    {hero_stats}
</div>
""", unsafe_allow_html=True)

if not MODELS_READY:
    st.warning(
        "⚠️ Model files not found. Place `best_classification_pipeline.pkl`, "
        "`best_regression_pipeline.pkl` and `model_metadata.pkl` inside the `models/` folder, "
        "and `processed_real_estate_data.csv` inside the `data/` folder. "
        "See the README for exact download steps from Kaggle."
    )

# ------------------------------------------------------------------
# SIDEBAR NAV / QUICK STATS
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2.6rem; line-height:1;">🏠</div>
        <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.1rem; margin-top:0.3rem;
             background: linear-gradient(90deg, #00E8C4, #6C8CFF); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
             Investment Advisor
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🧭 Navigation")
    st.caption("Use the tabs at the top to explore predictions, market insights and model performance.")
    st.markdown("---")
    if df is not None:
        st.markdown("### 📊 Dataset Snapshot")
        st.metric("Total Properties", f"{len(df):,}")
        if "City" in df.columns:
            st.metric("Cities Covered", df["City"].nunique())
        if "Good_Investment" in df.columns:
            st.metric("Good Investment Rate", f"{df['Good_Investment'].mean()*100:.1f}%")
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · Plotly")

# ------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------
tab_predict, tab_insights, tab_model, tab_about = st.tabs(
    ["🔮 Predict", "📈 Market Insights", "🧠 Model Performance", "ℹ️ About"]
)

# ====================================================================
# TAB 1 — PREDICT
# ====================================================================
with tab_predict:
    if not MODELS_READY:
        st.info("Load the trained models to enable predictions.")
    else:
        numeric_features = meta["numeric_features"]
        categorical_features = meta["categorical_features"]

        st.markdown("#### Enter Property Details")

        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                bhk = st.slider("BHK", 1, 6, 2)
                size_sqft = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1200, step=50)
                floor_no = st.number_input("Floor No.", min_value=0, max_value=80, value=3)
                total_floors = st.number_input("Total Floors", min_value=1, max_value=80, value=10)

            with c2:
                age = st.slider("Age of Property (Years)", 0, 60, 5)
                nearby_schools = st.slider("Nearby Schools", 0, 20, 5)
                nearby_hospitals = st.slider("Nearby Hospitals", 0, 20, 4)
                transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"], index=1)

            with c3:
                price_lakhs = st.number_input("Current Price (Lakhs ₹)", min_value=5.0, max_value=1000.0, value=75.0, step=1.0)
                amenities_count = st.slider("Amenities Count", 0, 15, 5)
                state = st.selectbox("State", sorted(df["State"].unique()) if df is not None else ["Maharashtra"])
                city_opts = sorted(df[df["State"] == state]["City"].unique()) if df is not None else ["Mumbai"]
                city = st.selectbox("City", city_opts)

            c4, c5, c6 = st.columns(3)
            with c4:
                property_type = st.selectbox("Property Type", ["Apartment", "Independent House", "Villa"])
            with c5:
                furnished = st.selectbox("Furnished Status", ["Furnished", "Semi-furnished", "Unfurnished"])
                parking = st.selectbox("Parking Space", ["Yes", "No"])
            with c6:
                security = st.selectbox("Security", ["Yes", "No"])
                facing = st.selectbox("Facing", ["North", "South", "East", "West"])
                owner_type = st.selectbox("Owner Type", ["Owner", "Builder", "Broker"])
                availability = st.selectbox("Availability Status", ["Ready_to_Move", "Under_Construction"])

            submitted = st.form_submit_button("🚀 Analyze Property")

        if submitted:
            price_per_sqft_calc = (price_lakhs * 100000) / size_sqft
            transport_map = {"Low": 0, "Medium": 1, "High": 2}
            school_min = df["Nearby_Schools"].min() if df is not None else 0
            school_max = df["Nearby_Schools"].max() if df is not None else 20
            hosp_min = df["Nearby_Hospitals"].min() if df is not None else 0
            hosp_max = df["Nearby_Hospitals"].max() if df is not None else 20

            school_density = (nearby_schools - school_min) / max(school_max - school_min, 1)
            hospital_density = (nearby_hospitals - hosp_min) / max(hosp_max - hosp_min, 1)

            input_row = pd.DataFrame([{
                "BHK": bhk,
                "Size_in_SqFt": size_sqft,
                "Price_per_SqFt_calc": price_per_sqft_calc,
                "Age_of_Property_calc": age,
                "Floor_No": floor_no,
                "Total_Floors": total_floors,
                "Nearby_Schools": nearby_schools,
                "Nearby_Hospitals": nearby_hospitals,
                "Amenities_Count": amenities_count,
                "School_Density_Score": school_density,
                "Hospital_Density_Score": hospital_density,
                "Transport_Score": transport_map[transport],
                "State": state,
                "City": city,
                "Property_Type": property_type,
                "Furnished_Status": furnished,
                "Parking_Space": parking,
                "Security": security,
                "Facing": facing,
                "Owner_Type": owner_type,
                "Availability_Status": availability,
            }])

            with st.spinner("Running AI models..."):
                clf_pred = clf_pipeline.predict(input_row)[0]
                clf_proba = clf_pipeline.predict_proba(input_row)[0][1] if hasattr(clf_pipeline, "predict_proba") else None
                future_price = reg_pipeline.predict(input_row)[0]

            st.markdown("---")
            st.markdown("### 📋 Analysis Result")
            st.caption("Here's what the AI models found for this property ↓")

            r1, r2, r3 = st.columns([1.2, 1, 1])
            with r1:
                if clf_pred == 1:
                    st.markdown('<span class="badge-good">✅ Good Investment</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-bad">⚠️ Not a Strong Investment</span>', unsafe_allow_html=True)
                if clf_proba is not None:
                    st.caption(f"Model confidence: **{clf_proba*100:.1f}%**")

            growth_pct = ((future_price - price_lakhs) / price_lakhs) * 100

            with r2:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="val">₹{future_price:,.1f} L</div>
                    <div class="lbl">Est. Price in 5 Years</div>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                st.markdown(f"""
                <div class="metric-tile">
                    <div class="val">+{growth_pct:.1f}%</div>
                    <div class="lbl">Projected Growth</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Today", "In 5 Years"],
                y=[price_lakhs, future_price],
                marker=dict(color=["#6C8CFF", "#00D4B4"]),
                text=[f"₹{price_lakhs:,.1f}L", f"₹{future_price:,.1f}L"],
                textposition="outside",
            ))
            fig.update_layout(
                title="Projected Value Growth",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=380,
                margin=dict(t=60, b=20),
            )
            st.plotly_chart(fig, width='stretch')

            if clf_proba is not None:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=clf_proba * 100,
                    title={"text": "Investment Confidence Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#00D4B4"},
                        "steps": [
                            {"range": [0, 40], "color": "rgba(255,107,107,0.35)"},
                            {"range": [40, 70], "color": "rgba(255,196,77,0.3)"},
                            {"range": [70, 100], "color": "rgba(0,212,180,0.35)"},
                        ],
                    },
                ))
                gauge.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                    height=320, margin=dict(t=50, b=10),
                )
                st.plotly_chart(gauge, width='stretch')

# ====================================================================
# TAB 2 — MARKET INSIGHTS
# ====================================================================
with tab_insights:
    if df is None:
        st.info("Processed dataset not found. Add `processed_real_estate_data.csv` to the `data/` folder.")
    else:
        st.markdown("#### Explore the Market")

        m1, m2, m3, m4 = st.columns(4)
        tiles = [
            (f"{len(df):,}", "Properties"),
            (f"₹{df['Price_in_Lakhs'].mean():,.1f}L", "Avg. Price"),
            (f"{df['Size_in_SqFt'].mean():,.0f}", "Avg. SqFt"),
            (f"{df['Good_Investment'].mean()*100:.1f}%", "Good Investment Rate"),
        ]
        for col, (val, lbl) in zip([m1, m2, m3, m4], tiles):
            col.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            city_price = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(15).reset_index()
            fig1 = px.bar(city_price, x="City", y="Price_in_Lakhs", color="Price_in_Lakhs",
                          color_continuous_scale="Tealgrn", title="Top 15 Cities by Avg. Price")
            fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, width='stretch')

        with colB:
            fig2 = px.histogram(df, x="Price_in_Lakhs", nbins=50, title="Price Distribution",
                                 color_discrete_sequence=["#00D4B4"])
            fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width='stretch')

        colC, colD = st.columns(2)
        with colC:
            fig3 = px.box(df, x="Property_Type", y="Price_per_SqFt_calc", color="Property_Type",
                          title="Price per SqFt by Property Type")
            fig3.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig3, width='stretch')

        with colD:
            transport_inv = df.groupby("Public_Transport_Accessibility")["Good_Investment"].mean().reindex(["Low", "Medium", "High"]).reset_index()
            fig4 = px.bar(transport_inv, x="Public_Transport_Accessibility", y="Good_Investment",
                          title="Good Investment Rate by Transport Access", color="Good_Investment",
                          color_continuous_scale="Tealgrn")
            fig4.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, width='stretch')

        st.markdown("#### Filter & Explore Raw Data")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            sel_states = st.multiselect("State", sorted(df["State"].unique()))
        with f2:
            sel_types = st.multiselect("Property Type", sorted(df["Property_Type"].unique()))
        with f3:
            bhk_range = st.slider("BHK Range", int(df["BHK"].min()), int(df["BHK"].max()),
                                   (int(df["BHK"].min()), int(df["BHK"].max())))
        with f4:
            price_range = st.slider("Price Range (Lakhs ₹)", float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max()),
                                     (float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())))

        filtered = df.copy()
        if sel_states:
            filtered = filtered[filtered["State"].isin(sel_states)]
        if sel_types:
            filtered = filtered[filtered["Property_Type"].isin(sel_types)]
        filtered = filtered[
            (filtered["BHK"].between(bhk_range[0], bhk_range[1])) &
            (filtered["Price_in_Lakhs"].between(price_range[0], price_range[1]))
        ]
        st.caption(f"Showing {len(filtered):,} of {len(df):,} properties matching filters")
        st.dataframe(filtered.head(200), width='stretch', height=350)

# ====================================================================
# TAB 3 — MODEL PERFORMANCE
# ====================================================================
with tab_model:
    if not MODELS_READY:
        st.info("Load model metadata to view performance details.")
    else:
        st.markdown("#### Model Summary")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <h4>🏷️ Classification — Good Investment</h4>
                <p style="color:#9AA5B1;">Best model selected: <b style="color:#00D4B4;">{meta.get('best_classification_model', 'N/A')}</b></p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card">
                <h4>📈 Regression — 5-Year Price</h4>
                <p style="color:#9AA5B1;">Best model selected: <b style="color:#00D4B4;">{meta.get('best_regression_model', 'N/A')}</b></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if df is not None:
            st.markdown("#### 📊 Evaluation Metrics")
            st.caption("Computed live by re-running the saved pipelines on a held-out sample of `processed_real_estate_data.csv` (80/20 split, same random_state as training).")

            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                                          mean_squared_error, mean_absolute_error, r2_score)

            X_eval = df[meta["all_features"]]

            with st.spinner("Scoring models..."):
                # Classification metrics
                y_clf = df["Good_Investment"]
                _, X_clf_test, _, y_clf_test = train_test_split(
                    X_eval, y_clf, test_size=0.2, random_state=42, stratify=y_clf
                )
                clf_preds = clf_pipeline.predict(X_clf_test)
                acc = accuracy_score(y_clf_test, clf_preds)
                f1 = f1_score(y_clf_test, clf_preds)
                cm = confusion_matrix(y_clf_test, clf_preds)

                # Regression metrics
                y_reg = df["Future_Price_5yr"]
                _, X_reg_test, _, y_reg_test = train_test_split(
                    X_eval, y_reg, test_size=0.2, random_state=42
                )
                reg_preds = reg_pipeline.predict(X_reg_test)
                rmse = np.sqrt(mean_squared_error(y_reg_test, reg_preds))
                mae = mean_absolute_error(y_reg_test, reg_preds)
                r2 = r2_score(y_reg_test, reg_preds)

            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            for col, (val, lbl) in zip(
                [mc1, mc2, mc3, mc4, mc5],
                [
                    (f"{acc*100:.2f}%", "Accuracy"),
                    (f"{f1:.3f}", "F1-Score"),
                    (f"{rmse:.2f}", "RMSE (Lakhs)"),
                    (f"{mae:.2f}", "MAE (Lakhs)"),
                    (f"{r2:.3f}", "R² Score"),
                ],
            ):
                col.markdown(f'<div class="metric-tile"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            cc1, cc2 = st.columns(2)
            with cc1:
                cm_fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Tealgrn",
                    x=["Predicted: Not Good", "Predicted: Good"],
                    y=["Actual: Not Good", "Actual: Good"],
                    title="Confusion Matrix — Classification",
                )
                cm_fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(cm_fig, width='stretch')

            with cc2:
                sample_idx = np.random.choice(len(y_reg_test), size=min(2000, len(y_reg_test)), replace=False)
                av_fig = go.Figure()
                av_fig.add_trace(go.Scatter(
                    x=np.array(y_reg_test)[sample_idx], y=np.array(reg_preds)[sample_idx],
                    mode="markers", marker=dict(color="#00D4B4", opacity=0.35, size=6),
                    name="Predictions",
                ))
                lims = [min(y_reg_test.min(), reg_preds.min()), max(y_reg_test.max(), reg_preds.max())]
                av_fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", line=dict(color="#FF6B6B", dash="dash"), name="Perfect Prediction"))
                av_fig.update_layout(
                    title="Actual vs Predicted — 5-Year Price",
                    xaxis_title="Actual (Lakhs)", yaxis_title="Predicted (Lakhs)",
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(av_fig, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Feature Set Used")
        f1c, f2c = st.columns(2)
        with f1c:
            st.markdown("**Numeric Features**")
            st.write(meta["numeric_features"])
        with f2c:
            st.markdown("**Categorical Features**")
            st.write(meta["categorical_features"])

        st.caption("These metrics were recomputed live in-app. Full MLflow experiment history (all candidate models, params, and artifacts) is in `mlruns/` from your Kaggle notebook — run `mlflow ui` locally to browse it.")

# ====================================================================
# TAB 4 — ABOUT
# ====================================================================
with tab_about:
    st.markdown("""
    <div class="glass-card">
    <h3>About this App</h3>
    <p style="color:#B7C0CC;">
    This Real Estate Investment Advisor uses machine learning trained on Indian housing data to:
    </p>
    <ul style="color:#B7C0CC;">
        <li>🏷️ Classify whether a property is a <b>Good Investment</b>, based on price-per-sqft, BHK, amenities, transport access, and more.</li>
        <li>📈 Predict the <b>estimated property value 5 years from now</b>, using location and property-type growth modeling.</li>
    </ul>
    <p style="color:#B7C0CC;">Pipelines were trained with scikit-learn (Logistic Regression / Random Forest / XGBoost) and tracked with MLflow.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    <h4>⚙️ Tech Stack</h4>
    <p style="color:#B7C0CC;">Python · Pandas · Scikit-learn · XGBoost · MLflow · Streamlit · Plotly</p>
    </div>
    """, unsafe_allow_html=True)
