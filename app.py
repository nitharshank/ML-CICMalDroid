import streamlit as st
import numpy as np
import json
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Android Malware Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer     {visibility: hidden;}
header     {visibility: hidden;}

.main-header {
    background: linear-gradient(135deg, #4361ee 0%, #7b2ff7 60%, #f72585 100%);
    padding: 2.2rem 1rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 1.8rem;
    color: white;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    color: white;
    letter-spacing: 0.02em;
}
.main-header p {
    margin: 0.4rem 0 0;
    color: rgba(255,255,255,0.78);
    font-size: 0.95rem;
}
.section-label {
    font-size: 0.7rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
div.stButton > button {
    background: #4361ee;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: background 0.2s;
}
div.stButton > button:hover  { background: #2f4bd6; color: white; }
div.stButton > button:active { background: #2438b0; color: white; }
.prob-box {
    background: #f4f5f7;
    border-radius: 10px;
    padding: 1rem 0.5rem;
    text-align: center;
    margin-top: 1rem;
}
.prob-label { font-size: 0.78rem; color: #666; margin-bottom: 0.2rem; }
.prob-val-benign  { font-size: 1.9rem; font-weight: 700; color: #00a86b; }
.prob-val-malware { font-size: 1.9rem; font-weight: 700; color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner="Loading models…")
def load_artifacts():
    feats  = json.load(open(os.path.join(BASE, "processed/drebin_features.json")))
    scaler = joblib.load(os.path.join(BASE, "processed/drebin_scaler.joblib"))
    vt     = joblib.load(os.path.join(BASE, "processed/drebin_variance_selector.joblib"))
    rf     = joblib.load(os.path.join(BASE, "models/rf_drebin.joblib"))
    xgb    = joblib.load(os.path.join(BASE, "models/xgb_drebin.joblib"))
    mlp    = joblib.load(os.path.join(BASE, "models/mlp_drebin.joblib"))
    return feats, scaler, vt, rf, xgb, mlp

features, scaler, vt, rf_model, xgb_model, mlp_model = load_artifacts()
feature_set = set(features)
feat_idx    = {f: i for i, f in enumerate(features)}   # fast lookup

# ── Session state initialisation ──────────────────────────────
# _uploaded_vals: plain dict {feat: 0/1} — NEVER cleared by Streamlit.
# Checkbox widget keys cb_* are only used for features VISIBLE on screen.
# At inference time we merge both sources.
if "_uploaded_vals" not in st.session_state:
    st.session_state["_uploaded_vals"] = {}   # empty until JSON uploaded

# ── Inference ─────────────────────────────────────────────────
def run_ensemble(vec: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x     = vec.astype(np.float32).reshape(1, -1)
        x_sc  = scaler.transform(x)
        x_vt  = vt.transform(x_sc)
        p_rf  = float(rf_model.predict_proba(x_vt)[0, 1])
        p_xgb = float(xgb_model.predict_proba(x_vt)[0, 1])
        p_mlp = float(mlp_model.predict_proba(x_vt)[0, 1])
        p_ens = (p_rf + p_xgb + p_mlp) / 3
    return {"ensemble": p_ens, "rf": p_rf, "xgb": p_xgb, "mlp": p_mlp}

def build_feature_vector(displayed_feats: list) -> np.ndarray:
    """
    Merge uploaded JSON values (stored in _uploaded_vals, always available)
    with checkbox state (only valid for currently displayed features).
    Checkbox value takes priority so manual toggles always win.
    """
    uploaded = st.session_state["_uploaded_vals"]
    vec = np.zeros(len(features), dtype=np.float32)

    # 1. Apply uploaded JSON values for ALL 215 features
    for f, v in uploaded.items():
        if f in feat_idx:
            vec[feat_idx[f]] = float(v)

    # 2. Override with live checkbox values for displayed features only
    for f in displayed_feats:
        cb_key = f"cb_{f}"
        if cb_key in st.session_state:
            vec[feat_idx[f]] = 1.0 if st.session_state[cb_key] else 0.0

    return vec

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 🛡️ Android Malware Detection System</h1>
    <p>AI-Powered Static & Dynamic Android Malware Detection</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────
st.markdown('<div class="section-label">Upload Feature JSON</div>',
            unsafe_allow_html=True)

uploaded = st.file_uploader(
    label="",
    type="json",
    label_visibility="collapsed",
    help='JSON mapping feature names to 1 (present) or 0 (absent). '
         'Example: {"SEND_SMS": 1, "READ_CONTACTS": 0}',
)

if uploaded is not None:
    file_id = getattr(uploaded, "file_id", uploaded.name)
    if st.session_state.get("_last_file_id") != file_id:
        try:
            data    = json.load(uploaded)
            matched = ignored = 0
            new_vals = {}
            for key, val in data.items():
                if key in feature_set:
                    new_vals[key] = int(bool(int(val)))
                    # Also prime the checkbox key so it shows correctly
                    st.session_state[f"cb_{key}"] = bool(int(val))
                    matched += 1
                else:
                    ignored += 1

            # Store in plain session state — survives across reruns
            st.session_state["_uploaded_vals"] = new_vals
            st.session_state["_last_file_id"]  = file_id
            st.session_state["_json_status"]   = (matched, ignored)
            st.session_state["_result"]        = None   # clear stale result
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

elif st.session_state.get("_last_file_id"):
    # File was removed — reset uploaded values and checkboxes
    st.session_state["_uploaded_vals"] = {}
    st.session_state["_last_file_id"]  = None
    st.session_state["_json_status"]   = None

if st.session_state.get("_json_status"):
    m, ig = st.session_state["_json_status"]
    n_active = sum(st.session_state["_uploaded_vals"].values())
    st.success(
        f"Uploaded JSON applied. "
        f"Matched keys: {m} | Ignored keys: {ig} | "
        f"Active features: {n_active}"
    )

# ── Feature checkboxes ────────────────────────────────────────
st.markdown('<div class="section-label">Select Feature Manually</div>',
            unsafe_allow_html=True)

col_search, col_slider = st.columns([4, 1])
with col_search:
    search = st.text_input("Search Feature", placeholder="Type to filter…",
                           label_visibility="visible")
with col_slider:
    show_n = st.slider("Show", min_value=10, max_value=len(features),
                       value=20, step=5)

filtered  = [f for f in features if search.lower() in f.lower()] if search \
            else features
displayed = filtered[:show_n]

uploaded_vals = st.session_state["_uploaded_vals"]

cols = st.columns(3)
for idx, feat in enumerate(displayed):
    with cols[idx % 3]:
        # Default: uploaded value if present, else False
        default = bool(uploaded_vals.get(feat, False))
        st.checkbox(feat, value=default, key=f"cb_{feat}")

# ── Analyse button ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
analyse_col, _ = st.columns([1, 2])
with analyse_col:
    analyse = st.button("🔍 Analyse For Malware", use_container_width=True)

if analyse:
    vec      = build_feature_vector(displayed)
    n_active = int(vec.sum())
    if n_active == 0:
        st.warning("No features selected. Upload a JSON file or tick at least one feature.")
    else:
        with st.spinner("Running ensemble…"):
            result = run_ensemble(vec)
            result["n_active"] = n_active
            st.session_state["_result"] = result

# ── Result ────────────────────────────────────────────────────
if st.session_state.get("_result"):
    r      = st.session_state["_result"]
    p_mal  = r["ensemble"]
    p_ben  = 1.0 - p_mal
    is_mal = p_mal >= 0.5

    confidence = p_mal if is_mal else p_ben
    color      = "#dc3545" if is_mal else "#00a86b"
    icon       = "⚠️" if is_mal else "✅"
    label      = "MALWARE" if is_mal else "BENIGN"
    v_class    = "verdict-malware" if is_mal else "verdict-benign"

    st.markdown(f"""
    <div style="background:white; border-radius:14px; padding:2rem 1.5rem 1.5rem;
                text-align:center; box-shadow:0 6px 28px rgba(0,0,0,0.13);
                color:#111; margin-top:1rem;">
        <div style="font-size:3.5rem; line-height:1.1;">{icon}</div>
        <div style="color:{color}; font-size:2.4rem; font-weight:800;
                    letter-spacing:0.15em; margin:0.3rem 0;">{label}</div>
        <div style="color:#555; font-size:0.85rem; margin-bottom:0.8rem;">
            {r['n_active']} feature(s) active
        </div>
        <div style="background:#e9ecef; border-radius:20px; height:26px;
                    overflow:hidden; margin:0.4rem 0;">
            <div style="width:{confidence*100:.1f}%; background:{color}; height:100%;
                        border-radius:20px; display:flex; align-items:center;
                        justify-content:center;">
                <span style="color:white; font-size:0.82rem; font-weight:600;
                             white-space:nowrap;">
                    {confidence*100:.2f}% Confidence
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="prob-box">
            <div class="prob-label">Benign Probability</div>
            <div class="prob-val-benign">{p_ben*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="prob-box">
            <div class="prob-label">Malware Probability</div>
            <div class="prob-val-malware">{p_mal*100:.2f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔬 Individual model breakdown"):
        m1, m2, m3 = st.columns(3)
        for col, name, prob in [
            (m1, "Random Forest", r["rf"]),
            (m2, "XGBoost",       r["xgb"]),
            (m3, "MLP",           r["mlp"]),
        ]:
            with col:
                v = "🔴 Malware" if prob >= 0.5 else "🟢 Benign"
                st.metric(label=name, value=f"{prob*100:.2f}%", delta=v,
                          delta_color="inverse" if prob >= 0.5 else "normal")
