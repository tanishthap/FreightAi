"""
FreightAI — Logistics Decision-Support Dashboard
=================================================
Built with Streamlit, GeoPandas, Plotly, and scikit-learn.

Structure
---------
1.  Data loading & preprocessing  (cached)
2.  ML model training             (cached)
3.  Sidebar: shipment input form
4.  Main area:
    - KPI cards
    - 3-column layout: Prediction | Map | Recommendations
    - AI Explanation & Risk Check panel
"""

import warnings, os
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
import math

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FreightAI — Logistics Decision Support",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #f7f8fb; }
[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8eaf0; }
[data-testid="stSidebar"] .stMarkdown p { color: #555; font-size: 13px; }

/* Cards */
.kpi-card {
    background: #ffffff; border-radius: 14px; padding: 18px 22px;
    border: 1px solid #e8eaf0; box-shadow: 0 2px 8px rgba(0,0,0,.04);
    text-align: center;
}
.kpi-label { font-size: 12px; color: #8a8fa8; font-weight: 600;
             letter-spacing: .06em; text-transform: uppercase; margin-bottom: 6px; }
.kpi-value { font-size: 30px; font-weight: 700; color: #1a2340; line-height: 1; }
.kpi-sub   { font-size: 12px; color: #a0a5bc; margin-top: 4px; }

.panel {
    background: #ffffff; border-radius: 16px; padding: 22px 24px;
    border: 1px solid #e8eaf0; box-shadow: 0 2px 10px rgba(0,0,0,.04);
    height: 100%; margin-bottom: 16px;
}
.panel-title {
    font-size: 13px; font-weight: 700; color: #8a8fa8;
    letter-spacing: .07em; text-transform: uppercase; margin-bottom: 16px;
}

/* Risk badge */
.badge {
    display: inline-block; padding: 5px 16px; border-radius: 20px;
    font-size: 13px; font-weight: 700; letter-spacing: .04em;
}
.badge-low    { background: #e6f7ee; color: #1a7a45; }
.badge-medium { background: #fff7e0; color: #9a6700; }
.badge-high   { background: #fde8e8; color: #b82b2b; }

/* Metric row */
.metric-row { display: flex; justify-content: space-between;
              align-items: center; padding: 10px 0;
              border-bottom: 1px solid #f1f3f8; }
.metric-row:last-child { border-bottom: none; }
.metric-key { font-size: 13px; color: #8a8fa8; }
.metric-val { font-size: 14px; font-weight: 600; color: #1a2340; }

/* Risk factor tags */
.tag { display: inline-block; padding: 4px 12px; border-radius: 20px;
       font-size: 12px; font-weight: 600; margin: 3px 4px 3px 0; }
.tag-red    { background: #fde8e8; color: #b82b2b; }
.tag-amber  { background: #fff7e0; color: #9a6700; }
.tag-blue   { background: #e6f0ff; color: #1a4fa8; }
.tag-green  { background: #e6f7ee; color: #1a7a45; }

.explain-box {
    background: #f4f6ff; border-left: 3px solid #3d6beb;
    border-radius: 0 10px 10px 0; padding: 13px 16px;
    font-size: 14px; color: #3a4060; line-height: 1.65; margin-bottom: 14px;
}
.rec-item { display: flex; align-items: flex-start; gap: 12px;
            padding: 10px 0; border-bottom: 1px solid #f1f3f8; }
.rec-item:last-child { border-bottom: none; }
.rec-icon { width: 34px; height: 34px; border-radius: 10px; display: flex;
            align-items: center; justify-content: center; font-size: 16px;
            flex-shrink: 0; }
.rec-text-label { font-size: 11px; color: #a0a5bc; font-weight: 600; text-transform: uppercase; }
.rec-text-val   { font-size: 14px; color: #1a2340; font-weight: 600; margin-top: 1px; }
.disclaimer { font-size: 11px; color: #a0a5bc; margin-top: 12px;
              padding-top: 10px; border-top: 1px solid #f1f3f8; }

/* Header */
.header-wrap { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.header-logo { width: 42px; height: 42px; background: #1a3d7c; border-radius: 10px;
               display: flex; align-items: center; justify-content: center;
               font-size: 22px; }
.header-title { font-size: 26px; font-weight: 800; color: #1a2340; }
.header-sub   { font-size: 14px; color: #8a8fa8; margin-left: 56px; }

div[data-testid="stHorizontalBlock"] > div { gap: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

FIPS_TO_STATE = {
    1:"Alabama", 2:"Alaska", 4:"Arizona", 5:"Arkansas", 6:"California",
    8:"Colorado", 9:"Connecticut", 10:"Delaware", 11:"DC", 12:"Florida",
    13:"Georgia", 15:"Hawaii", 16:"Idaho", 17:"Illinois", 18:"Indiana",
    19:"Iowa", 20:"Kansas", 21:"Kentucky", 22:"Louisiana", 23:"Maine",
    24:"Maryland", 25:"Massachusetts", 26:"Michigan", 27:"Minnesota",
    28:"Mississippi", 29:"Missouri", 30:"Montana", 31:"Nebraska", 32:"Nevada",
    33:"New Hampshire", 34:"New Jersey", 35:"New Mexico", 36:"New York",
    37:"North Carolina", 38:"North Dakota", 39:"Ohio", 40:"Oklahoma",
    41:"Oregon", 42:"Pennsylvania", 44:"Rhode Island", 45:"South Carolina",
    46:"South Dakota", 47:"Tennessee", 48:"Texas", 49:"Utah", 50:"Vermont",
    51:"Virginia", 53:"Washington", 54:"West Virginia", 55:"Wisconsin",
    56:"Wyoming",
}
STATE_TO_FIPS = {v: k for k, v in FIPS_TO_STATE.items()}

# FAF mode codes
MODE_MAP = {1:"Truck", 2:"Rail", 3:"Water", 4:"Air", 5:"Multiple/Mail",
            6:"Pipeline", 7:"Other/Unknown"}
MODE_REV = {v: k for k, v in MODE_MAP.items()}

# Approximate state centroids (lat, lon) for map line drawing
STATE_CENTROIDS = {
    "Alabama":(32.8,-86.8),"Alaska":(64.2,-153.4),"Arizona":(34.3,-111.1),
    "Arkansas":(34.9,-92.4),"California":(36.8,-119.4),"Colorado":(39.0,-105.5),
    "Connecticut":(41.6,-72.7),"Delaware":(39.0,-75.5),"DC":(38.9,-77.0),
    "Florida":(28.7,-82.5),"Georgia":(32.7,-83.4),"Hawaii":(20.9,-157.0),
    "Idaho":(44.4,-114.6),"Illinois":(40.0,-89.2),"Indiana":(39.8,-86.1),
    "Iowa":(42.1,-93.2),"Kansas":(38.5,-98.4),"Kentucky":(37.5,-85.3),
    "Louisiana":(31.2,-91.8),"Maine":(45.3,-69.0),"Maryland":(39.0,-76.7),
    "Massachusetts":(42.3,-71.8),"Michigan":(44.3,-85.4),"Minnesota":(46.4,-93.1),
    "Mississippi":(32.7,-89.7),"Missouri":(38.5,-92.5),"Montana":(46.9,-110.4),
    "Nebraska":(41.5,-99.9),"Nevada":(39.3,-116.6),"New Hampshire":(43.7,-71.6),
    "New Jersey":(40.1,-74.5),"New Mexico":(34.5,-106.1),"New York":(42.9,-75.6),
    "North Carolina":(35.6,-79.8),"North Dakota":(47.5,-100.5),"Ohio":(40.4,-82.8),
    "Oklahoma":(35.6,-97.5),"Oregon":(43.9,-120.6),"Pennsylvania":(40.9,-77.8),
    "Rhode Island":(41.7,-71.5),"South Carolina":(33.9,-80.9),
    "South Dakota":(44.4,-100.2),"Tennessee":(35.9,-86.4),"Texas":(31.1,-97.6),
    "Utah":(39.3,-111.1),"Vermont":(44.0,-72.7),"Virginia":(37.5,-79.5),
    "Washington":(47.4,-120.5),"West Virginia":(38.6,-80.6),"Wisconsin":(44.3,-90.1),
    "Wyoming":(43.0,-107.6),
}

# Carriers per mode
MODE_CARRIERS = {
    "Truck":["XPO Logistics","J.B. Hunt","Werner Enterprises","Schneider National"],
    "Rail":["Union Pacific","BNSF Railway","CSX Transportation","Norfolk Southern"],
    "Water":["Crowley Maritime","SEACOR Holdings","American Commercial Barge"],
    "Air":["FedEx Freight","UPS Supply Chain","DHL Express"],
    "Multiple/Mail":["FedEx","UPS","DHL","USPS"],
    "Pipeline":["Kinder Morgan","Enterprise Products","Plains All American"],
    "Other/Unknown":["Contact carrier directly"],
}


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_freight_data():
    """Load FAF4 state-level CSV, keep domestic flows, build ML features."""
    df = pd.read_csv(os.path.join(DATA_DIR, "FAF4_4_State.csv"))

    # Keep domestic trade (trade_type == 1), cross-state, valid mode
    df = df[
        (df["trade_type"] == 1) &
        (df["dms_orig"].notna()) &
        (df["dms_dest"].notna()) &
        (df["dms_mode"].notna()) &
        (df["tons_2012"] > 0) &
        (df["tmiles_2012"] > 0)
    ].copy()

    df["dms_orig"] = df["dms_orig"].astype(int)
    df["dms_dest"] = df["dms_dest"].astype(int)
    df["dms_mode"] = df["dms_mode"].astype(int)

    # --- Derived features -------------------------------------------------------
    # Approximate distance in miles (ton-miles / tons)
    df["approx_dist_mi"] = df["tmiles_2012"] / df["tons_2012"] * 1000

    # Transit time proxy: distance / avg speed by mode (hours)
    # Mode speeds (mph): Truck=50, Rail=25, Water=10, Air=450, Multi=40, Pipeline=5, Other=40
    speed = {1:50, 2:25, 3:10, 4:450, 5:40, 6:5, 7:40, 8:40}
    df["avg_speed"] = df["dms_mode"].map(speed).fillna(40)
    df["transit_hours"] = df["approx_dist_mi"] / df["avg_speed"]
    df["transit_hours"] = df["transit_hours"].clip(1, 240)

    # Value density ($/ton) — high density = higher risk if delayed
    df["value_per_ton"] = (df["value_2012"] / df["tons_2012"]).clip(0, 5000)

    # Risk target: High = long distance or low-speed mode or high value density
    # Percentile-based classification
    dist_q66 = df["approx_dist_mi"].quantile(0.66)
    dist_q33 = df["approx_dist_mi"].quantile(0.33)
    val_q66  = df["value_per_ton"].quantile(0.66)

    def risk_label(row):
        score = 0
        if row["approx_dist_mi"] > dist_q66: score += 2
        elif row["approx_dist_mi"] > dist_q33: score += 1
        if row["dms_mode"] in [3, 6]: score += 2          # slow modes
        if row["dms_mode"] in [5, 7]: score += 1          # uncertain modes
        if row["value_per_ton"] > val_q66: score += 1
        if row["dms_orig"] == row["dms_dest"]: score = 0  # intrastate always low
        if score >= 4: return 2   # High
        if score >= 2: return 1   # Medium
        return 0                  # Low

    df["risk"] = df.apply(risk_label, axis=1)

    return df


@st.cache_data(show_spinner=False)
def load_geodata():
    """Load and dissolve shapefile to state level."""
    shp = os.path.join(DATA_DIR, "CFS_AREA_shapefile_010215.shp")
    gdf = gpd.read_file(shp)
    # Dissolve county-level shapes to state using ANSI_ST (FIPS str, zero-padded)
    gdf_state = gdf.dissolve(by="ANSI_ST", as_index=False)
    gdf_state["fips_int"] = gdf_state["ANSI_ST"].astype(int)
    gdf_state["state_name"] = gdf_state["fips_int"].map(FIPS_TO_STATE)
    # Project to WGS84 for Plotly
    gdf_state = gdf_state.to_crs(epsg=4326)
    return gdf_state


@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    """Train regression (transit time) and classification (risk) models."""
    features = ["dms_orig", "dms_dest", "dms_mode", "approx_dist_mi", "value_per_ton"]
    target_reg   = "transit_hours"
    target_clf   = "risk"

    sample = df[features + [target_reg, target_clf]].dropna()
    # Sample for speed
    if len(sample) > 80_000:
        sample = sample.sample(80_000, random_state=42)

    X = sample[features]
    y_reg = sample[target_reg]
    y_clf = sample[target_clf]

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    reg.fit(X_tr, yr_tr)

    clf = RandomForestClassifier(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, yc_tr)

    reg_score = reg.score(X_te, yr_te)
    clf_score = clf.score(X_te, yc_te)

    return reg, clf, reg_score, clf_score


# ── Map builder ────────────────────────────────────────────────────────────────
def build_map(gdf_state, origin_state, dest_state):
    """
    Build a Plotly choropleth map highlighting origin and destination states,
    with a great-circle arc between them.
    """
    # Assign color codes: 0=base, 1=origin, 2=dest, 3=both
    gdf_state = gdf_state.copy()
    gdf_state["highlight"] = 0
    gdf_state.loc[gdf_state["state_name"] == origin_state, "highlight"] = 1
    gdf_state.loc[gdf_state["state_name"] == dest_state,   "highlight"] = 2
    if origin_state == dest_state:
        gdf_state.loc[gdf_state["state_name"] == origin_state, "highlight"] = 3

    color_map = {0:"#dde4f0", 1:"#3d6beb", 2:"#e94c4c", 3:"#8b5cf6"}
    label_map = {0:"Other states", 1:f"Origin: {origin_state}",
                 2:f"Destination: {dest_state}", 3:f"Origin & Destination: {origin_state}"}

    gdf_state["color_label"] = gdf_state["highlight"].map(label_map)

    # Convert to GeoJSON for plotly
    import json
    geojson = json.loads(gdf_state.to_json())

    fig = px.choropleth(
        gdf_state,
        geojson=geojson,
        locations=gdf_state.index,
        color="highlight",
        color_discrete_map=color_map,
        hover_name="state_name",
        hover_data={"highlight": False, "color_label": False},
    )

    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><extra></extra>",
        marker_line_color="#b0bad0",
        marker_line_width=0.5,
    )

    # Add arc between origin and destination
    if origin_state in STATE_CENTROIDS and dest_state in STATE_CENTROIDS:
        o_lat, o_lon = STATE_CENTROIDS[origin_state]
        d_lat, d_lon = STATE_CENTROIDS[dest_state]

        # Generate great-circle intermediate points for arc effect
        n_pts = 30
        lats = np.linspace(o_lat, d_lat, n_pts)
        lons = np.linspace(o_lon, d_lon, n_pts)
        # Add slight parabolic curve
        t = np.linspace(0, 1, n_pts)
        arc_height = min(8, abs(d_lat - o_lat) * 0.5 + abs(d_lon - o_lon) * 0.2)
        lats = lats + arc_height * 4 * t * (1 - t)

        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines",
            line=dict(width=2.5, color="#f59e0b"),
            name="Route",
            hoverinfo="skip",
        ))

        # Origin marker
        fig.add_trace(go.Scattergeo(
            lat=[o_lat], lon=[o_lon], mode="markers+text",
            marker=dict(size=12, color="#3d6beb", symbol="circle",
                        line=dict(color="white", width=2)),
            text=[origin_state[:2].upper()], textposition="top center",
            textfont=dict(size=10, color="#1a2340"),
            name=f"Origin: {origin_state}", hoverinfo="skip",
        ))

        # Destination marker
        fig.add_trace(go.Scattergeo(
            lat=[d_lat], lon=[d_lon], mode="markers+text",
            marker=dict(size=12, color="#e94c4c", symbol="circle",
                        line=dict(color="white", width=2)),
            text=[dest_state[:2].upper()], textposition="top center",
            textfont=dict(size=10, color="#1a2340"),
            name=f"Destination: {dest_state}", hoverinfo="skip",
        ))

    fig.update_layout(
        geo=dict(
            scope="usa",
            showland=True, landcolor="#f0f2f8",
            showlakes=True, lakecolor="#cfe2ff",
            showcoastlines=True, coastlinecolor="#b0bad0",
            showframe=False,
            projection_type="albers usa",
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=300,
        showlegend=False,
        coloraxis_showscale=False,
    )
    return fig


# ── Prediction logic ───────────────────────────────────────────────────────────
def haversine_mi(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance in miles."""
    R = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def predict_shipment(reg, clf, df, origin_st, dest_st, mode_str):
    orig_fips = STATE_TO_FIPS[origin_st]
    dest_fips = STATE_TO_FIPS[dest_st]
    mode_code = MODE_REV.get(mode_str, 1)

    # Estimate distance from centroids
    if origin_st in STATE_CENTROIDS and dest_st in STATE_CENTROIDS:
        o = STATE_CENTROIDS[origin_st]
        d = STATE_CENTROIDS[dest_st]
        dist_mi = max(50, haversine_mi(o[0], o[1], d[0], d[1]))
    else:
        dist_mi = 1000

    # Look up average value per ton for this route from data
    mask = (
        (df["dms_orig"] == orig_fips) &
        (df["dms_dest"] == dest_fips) &
        (df["dms_mode"] == mode_code)
    )
    sub = df[mask]
    if len(sub) > 0:
        val_per_ton = float(sub["value_per_ton"].median())
        dist_mi_data = float(sub["approx_dist_mi"].median())
        dist_mi = (dist_mi + dist_mi_data) / 2  # blend geo + data
    else:
        val_per_ton = 50.0

    X_new = pd.DataFrame([{
        "dms_orig": orig_fips,
        "dms_dest": dest_fips,
        "dms_mode": mode_code,
        "approx_dist_mi": dist_mi,
        "value_per_ton": val_per_ton,
    }])

    transit_h = float(reg.predict(X_new)[0])
    risk_code  = int(clf.predict(X_new)[0])
    risk_proba = clf.predict_proba(X_new)[0]

    return {
        "transit_hours": round(transit_h, 1),
        "transit_days":  round(transit_h / 24, 1),
        "risk_code":     risk_code,
        "risk_label":    ["Low", "Medium", "High"][risk_code],
        "risk_proba":    risk_proba,
        "dist_mi":       round(dist_mi),
        "val_per_ton":   round(val_per_ton, 1),
    }


def get_risk_factors(pred, origin_st, dest_st, mode_str, ship_date):
    """Generate rule-based risk factor tags and explanation."""
    tags = []
    reasons = []

    if pred["dist_mi"] > 1500:
        tags.append(("Long-haul corridor", "red"))
        reasons.append("long interstate distance")
    elif pred["dist_mi"] > 700:
        tags.append(("Mid-range distance", "amber"))
        reasons.append("moderate route distance")

    if mode_str == "Truck":
        tags.append(("Driver HOS limits", "amber"))
        reasons.append("truck driver hours-of-service regulations apply")
    if mode_str == "Water":
        tags.append(("Port dwell risk", "red"))
        reasons.append("port terminal congestion can cause dwell delays")
    if mode_str == "Rail":
        tags.append(("Rail network congestion", "amber"))
        reasons.append("rail network capacity constraints may apply")
    if mode_str == "Air":
        tags.append(("Weather sensitivity", "amber"))
    if mode_str in ["Multiple/Mail", "Other/Unknown"]:
        tags.append(("Mode uncertainty", "amber"))
        reasons.append("multi-mode transfers add coordination complexity")

    month = ship_date.month if ship_date else date.today().month
    if month in [11, 12, 1, 2]:
        tags.append(("Winter weather risk", "red"))
        reasons.append("winter conditions may affect transit reliability")
    elif month in [6, 7, 8]:
        tags.append(("Peak freight season", "amber"))
        reasons.append("summer demand peaks often reduce capacity availability")

    if pred["val_per_ton"] > 200:
        tags.append(("High-value cargo", "blue"))
        reasons.append("high-value commodity requires enhanced handling")

    if pred["risk_code"] == 2:
        tags.append(("Route complexity", "red"))
    elif pred["risk_code"] == 1:
        tags.append(("Moderate risk route", "amber"))
    else:
        tags.append(("Efficient corridor", "green"))

    explanation = (
        f"The model estimates a transit time of <b>{pred['transit_hours']}h "
        f"({pred['transit_days']} days)</b> for the "
        f"<b>{origin_st} → {dest_st}</b> corridor via <b>{mode_str}</b> "
        f"over approximately <b>{pred['dist_mi']:,} miles</b>. "
    )
    if reasons:
        explanation += "Key contributing factors include: " + ", ".join(reasons[:3]) + "."

    return tags, explanation


def get_carrier(mode_str, risk_code):
    carriers = MODE_CARRIERS.get(mode_str, ["Contact carrier"])
    # For high risk pick first (most established), else vary
    return carriers[risk_code % len(carriers)]


def get_savings(dist_mi, mode_str, risk_code):
    base_rate = {"Truck":3.5,"Rail":1.8,"Water":0.9,"Air":14.0,
                 "Multiple/Mail":4.0,"Pipeline":0.5,"Other/Unknown":3.0}
    rate = base_rate.get(mode_str, 3.0)
    # Savings if switched to next-best mode for medium/high risk
    alt_savings = dist_mi * rate * 0.12 * (1 + risk_code * 0.08)
    return round(alt_savings, 0)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    with st.spinner("Loading freight data and training models…"):
        df       = load_freight_data()
        gdf_state = load_geodata()
        reg, clf, reg_r2, clf_acc = train_models(df)

    states = sorted(FIPS_TO_STATE.values())
    modes  = ["Truck", "Rail", "Water", "Air", "Multiple/Mail", "Pipeline", "Other/Unknown"]

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="header-wrap">
      <div class="header-logo">🚛</div>
      <div class="header-title">FreightAI</div>
    </div>
    <div class="header-sub">
      Smart Logistics Decision Support — predict transit time & shipment risk before execution
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📦 Shipment Details")
        st.markdown("---")

        origin_st = st.selectbox("🔵  Origin state", states,
                                 index=states.index("California"))
        dest_st   = st.selectbox("🔴  Destination state", states,
                                 index=states.index("New York"))
        mode_str  = st.selectbox("🚚  Mode of transport", modes)
        ship_date = st.date_input("📅  Shipment date",
                                  value=date.today() + timedelta(days=1))
        weight_lb = st.number_input("⚖️  Freight weight (lbs)", min_value=100,
                                    max_value=200_000, value=12_000, step=500)

        st.markdown("---")
        run = st.button("⚡  Generate Insights", use_container_width=True,
                        type="primary")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:11px;color:#a0a5bc">
        <b>Model performance</b><br>
        Regression R² &nbsp;: {reg_r2:.3f}<br>
        Classifier Acc: {clf_acc:.3f}<br><br>
        Data: FAF4 (2012 base year)<br>
        Source: Bureau of Transportation Statistics
        </div>
        """, unsafe_allow_html=True)

    # ── Platform KPIs (always visible) ─────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("""<div class="kpi-card">
            <div class="kpi-label">Shipments Analyzed</div>
            <div class="kpi-value">550K+</div>
            <div class="kpi-sub">FAF4 records</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Regression R²</div>
            <div class="kpi-value">{reg_r2:.2f}</div>
            <div class="kpi-sub">Transit time model</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-label">Classifier Accuracy</div>
            <div class="kpi-value">{clf_acc:.0%}</div>
            <div class="kpi-sub">Risk level model</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown("""<div class="kpi-card">
            <div class="kpi-label">Risk Categories</div>
            <div class="kpi-value">3</div>
            <div class="kpi-sub">Low · Medium · High</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

    # ── Default or prediction ──────────────────────────────────────────────────
    pred = None
    if run or "last_pred" in st.session_state:
        if run:
            if origin_st == dest_st:
                st.warning("⚠️  Origin and destination are the same state. "
                           "Results shown for intrastate routing.")
            with st.spinner("Running models…"):
                pred = predict_shipment(reg, clf, df, origin_st, dest_st, mode_str)
                pred["origin"]  = origin_st
                pred["dest"]    = dest_st
                pred["mode"]    = mode_str
                pred["date"]    = ship_date
                pred["weight"]  = weight_lb
                st.session_state["last_pred"] = pred
        else:
            pred = st.session_state["last_pred"]
            origin_st = pred["origin"]
            dest_st   = pred["dest"]
            mode_str  = pred["mode"]
            ship_date = pred["date"]
            weight_lb = pred["weight"]

    # ── 3-column main body ────────────────────────────────────────────────────
    col_pred, col_map, col_rec = st.columns([1, 1.5, 1])

    # ── LEFT: Prediction Results ───────────────────────────────────────────────
    with col_pred:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction Results</div>', unsafe_allow_html=True)

        if pred:
            badge_cls = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high"}[pred["risk_label"]]
            st.markdown(f"""
            <div style="margin-bottom:16px">
              <div class="kpi-label">Predicted transit time</div>
              <div style="font-size:36px;font-weight:800;color:#1a2340;line-height:1.1">
                {pred['transit_days']} days
              </div>
              <div style="font-size:14px;color:#8a8fa8;margin-top:2px">
                ≈ {pred['transit_hours']} hours
              </div>
            </div>
            <div style="margin-bottom:20px">
              <div class="kpi-label">Risk level</div>
              <div style="margin-top:4px">
                <span class="badge {badge_cls}">{pred['risk_label']} Risk</span>
              </div>
              <div style="height:6px;background:#f1f3f8;border-radius:3px;margin-top:10px;overflow:hidden">
                <div style="height:100%;border-radius:3px;width:{int(pred['risk_proba'][pred['risk_code']]*100)}%;
                     background:{'#1a7a45' if pred['risk_code']==0 else '#d97706' if pred['risk_code']==1 else '#b82b2b'}">
                </div>
              </div>
              <div style="font-size:11px;color:#a0a5bc;margin-top:4px">
                Model confidence: {pred['risk_proba'][pred['risk_code']]:.0%}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Arrival estimate
            from datetime import datetime, timedelta as td
            arrival_dt = datetime.combine(ship_date, datetime.min.time()) + td(hours=pred["transit_hours"])
            st.markdown(f"""
            <div class="metric-row">
              <span class="metric-key">Estimated arrival</span>
              <span class="metric-val">{arrival_dt.strftime('%b %d, %Y')}</span>
            </div>
            <div class="metric-row">
              <span class="metric-key">Route distance</span>
              <span class="metric-val">{pred['dist_mi']:,} mi</span>
            </div>
            <div class="metric-row">
              <span class="metric-key">Freight weight</span>
              <span class="metric-val">{weight_lb:,} lbs</span>
            </div>
            <div class="metric-row">
              <span class="metric-key">Mode</span>
              <span class="metric-val">{mode_str}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#a0a5bc;font-size:14px;padding:20px 0">
              Enter shipment details in the sidebar and click<br>
              <b>Generate Insights</b> to see predictions.
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── MIDDLE: Map ────────────────────────────────────────────────────────────
    with col_map:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Route Map</div>', unsafe_allow_html=True)

        if pred:
            fig_map = build_map(gdf_state, origin_st, dest_st)
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            st.markdown(f"""
            <div style="display:flex;gap:16px;justify-content:center;margin-top:6px">
              <span style="font-size:12px;color:#3d6beb">● Origin: {origin_st}</span>
              <span style="font-size:12px;color:#e94c4c">● Destination: {dest_st}</span>
              <span style="font-size:12px;color:#f59e0b">— Route arc</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            fig_map = build_map(gdf_state, "California", "New York")
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})
            st.markdown("""
            <div style="text-align:center;font-size:12px;color:#a0a5bc;margin-top:4px">
              Sample route shown — generate insights to update
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── RIGHT: Recommendations ────────────────────────────────────────────────
    with col_rec:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AI Recommendations</div>', unsafe_allow_html=True)

        if pred:
            carrier   = get_carrier(mode_str, pred["risk_code"])
            savings   = get_savings(pred["dist_mi"], mode_str, pred["risk_code"])
            alt_modes = [m for m in modes if m != mode_str]
            alt_mode  = alt_modes[pred["risk_code"] % len(alt_modes)]
            buffer_h  = {0: "2–4 hrs", 1: "6–10 hrs", 2: "12–20 hrs"}[pred["risk_code"]]

            st.markdown(f"""
            <div class="rec-item">
              <div class="rec-icon" style="background:#e6f0ff">🗺️</div>
              <div>
                <div class="rec-text-label">Optimal route</div>
                <div class="rec-text-val">{origin_st[:3].upper()} → {dest_st[:3].upper()} via I-corridor</div>
              </div>
            </div>
            <div class="rec-item">
              <div class="rec-icon" style="background:#e6f7ee">🏢</div>
              <div>
                <div class="rec-text-label">Suggested carrier</div>
                <div class="rec-text-val">{carrier}</div>
              </div>
            </div>
            <div class="rec-item">
              <div class="rec-icon" style="background:#fff7e0">💡</div>
              <div>
                <div class="rec-text-label">Alt. mode to consider</div>
                <div class="rec-text-val">{alt_mode}</div>
              </div>
            </div>
            <div class="rec-item">
              <div class="rec-icon" style="background:#fde8e8">⏱️</div>
              <div>
                <div class="rec-text-label">Suggested time buffer</div>
                <div class="rec-text-val">{buffer_h}</div>
              </div>
            </div>
            <div class="rec-item">
              <div class="rec-icon" style="background:#e6f7ee">💰</div>
              <div>
                <div class="rec-text-label">Est. cost savings</div>
                <div class="rec-text-val">${savings:,.0f} vs. current plan</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="color:#a0a5bc;font-size:14px;padding:10px 0">
              AI recommendations will appear after you generate insights.
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── AI Explanation & Risk Check ──────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🔍 AI Explanation & Risk Check</div>', unsafe_allow_html=True)

    if pred:
        tags, explanation = get_risk_factors(pred, origin_st, dest_st, mode_str, ship_date)

        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

        tag_html = ""
        for label, color in tags:
            tag_html += f'<span class="tag tag-{color}">{label}</span>'
        st.markdown(f"""
        <div style="margin-bottom:10px">
          <div style="font-size:12px;color:#a0a5bc;font-weight:600;
                      text-transform:uppercase;margin-bottom:8px">
            Identified Risk Factors
          </div>
          {tag_html}
        </div>
        """, unsafe_allow_html=True)

        # Risk probability breakdown
        rp = pred["risk_proba"]
        st.markdown("""
        <div style="font-size:12px;color:#a0a5bc;font-weight:600;
                    text-transform:uppercase;margin:14px 0 8px">
          Risk probability breakdown
        </div>""", unsafe_allow_html=True)

        rb1, rb2, rb3 = st.columns(3)
        for col, label, pct, color in [
            (rb1, "Low", rp[0], "#1a7a45"),
            (rb2, "Medium", rp[1], "#d97706"),
            (rb3, "High", rp[2], "#b82b2b"),
        ]:
            with col:
                st.markdown(f"""
                <div style="background:#f7f8fb;border-radius:10px;padding:12px;text-align:center">
                  <div style="font-size:11px;color:{color};font-weight:700;
                              text-transform:uppercase">{label}</div>
                  <div style="font-size:22px;font-weight:700;color:#1a2340">{pct:.0%}</div>
                  <div style="height:4px;background:#e8eaf0;border-radius:2px;
                              margin-top:6px;overflow:hidden">
                    <div style="height:100%;width:{pct*100:.1f}%;background:{color};border-radius:2px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
          ⚠️ FreightAI predictions are based on historical FAF4 data (2012 base year).
          Outputs are estimates to support — not replace — professional logistics decisions.
          Real-time conditions (weather, traffic, disruptions) are not captured in this model.
          Always apply domain expertise before execution.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color:#a0a5bc;font-size:14px;padding:6px 0">
          Risk explanations and contributing factors will appear here after analysis.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
