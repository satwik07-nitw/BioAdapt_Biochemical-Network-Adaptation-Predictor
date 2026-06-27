"""
app.py — BioAdapt: Biochemical Network Adaptation Predictor
Streamlit frontend. Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from predictor import predict, LINK_COLS, MOTIF_DESCRIPTIONS

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioAdapt — Network Adaptation Predictor",
    page_icon="🧬",
    layout="wide",
)

# ── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    h1 { color: #4fc3f7; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        border: 1px solid #2d3145;
    }
    .result-adaptive {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        border: 2px solid #43a047;
    }
    .result-nonadaptive {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-radius: 12px; padding: 1.5rem; text-align: center;
        border: 2px solid #e53935;
    }
    .motif-box {
        background: #1e2130;
        border-left: 4px solid #4fc3f7;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin-top: 0.5rem;
    }
    .edge-label { font-size: 13px; color: #90caf9; }
    div[data-testid="stSelectbox"] label { color: #b0bec5 !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("# 🧬 BioAdapt")
st.markdown(
    "**Biochemical Network Adaptation Predictor** — "
    "Predict whether a 3-node signaling network topology achieves perfect adaptation. "
    "Based on [Ma et al. (2009)](https://doi.org/10.1016/j.cell.2009.09.047)."
)
st.markdown("---")

# ── SIDEBAR: ABOUT ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
**BioAdapt** uses an XGBoost classifier trained on ODE simulation data 
from 1,400 three-node biochemical topologies (358 adaptive).

**Model performance (structural features only):**
- AUC = **0.882**
- F1 Score = **0.708**

**Nodes:**
- **A** = Input/Stimulus node  
- **B** = Intermediate node  
- **C** = Output node  

**Edge types:**
- **+1** Activation →
- **-1** Inhibition ⊣
- **0** No edge

**Key adaptive motifs:**
- **NFBLB** — Negative Feedback Loop with Buffer
- **IFFLP** — Incoherent Feedforward Loop

*Based on Ma et al. (2009), Cell 138:760-773*
""")
    st.markdown("---")
    st.markdown("Built by **Satwik Kumar Yadav** | NIT Warangal")
    st.markdown("[GitHub](https://github.com/satwik07-nitw) | [LinkedIn](https://linkedin.com)")

# ── MAIN: TOPOLOGY INPUT ─────────────────────────────────────────────────────
st.markdown("### Step 1 — Define your 3-node network topology")
st.markdown(
    "Set the edge type between each pair of nodes. "
    "Rows = **source**, Columns = **target**. "
    "e.g. `link_AB` = edge from A → B."
)

EDGE_OPTIONS = {"No edge (0)": 0, "Activation (+1)": 1, "Inhibition (−1)": -1}
NODES = ["A", "B", "C"]

# 3×3 grid of selectboxes
link_values = {}
cols_header = st.columns([1, 1, 1, 1])
cols_header[1].markdown("<center><b>→ A</b></center>", unsafe_allow_html=True)
cols_header[2].markdown("<center><b>→ B</b></center>", unsafe_allow_html=True)
cols_header[3].markdown("<center><b>→ C</b></center>", unsafe_allow_html=True)

defaults = {
    # Default: real NFBLB adaptive topology from Ma et al. (2009) — AUC 97.5%
    'link_AA': -1, 'link_AB':  1, 'link_AC': -1,
    'link_BA': -1, 'link_BB': -1, 'link_BC':  1,
    'link_CA': -1, 'link_CB': -1, 'link_CC': -1,
}

for src in NODES:
    row_cols = st.columns([1, 1, 1, 1])
    row_cols[0].markdown(f"<b>{src} →</b>", unsafe_allow_html=True)
    for i, tgt in enumerate(NODES):
        key = f"link_{src}{tgt}"
        default_val = defaults.get(key, 0)
        default_label = {0: "No edge (0)", 1: "Activation (+1)", -1: "Inhibition (−1)"}[default_val]
        choice = row_cols[i + 1].selectbox(
            label=f"{src}→{tgt}",
            options=list(EDGE_OPTIONS.keys()),
            index=list(EDGE_OPTIONS.keys()).index(default_label),
            key=key,
            label_visibility="collapsed"
        )
        link_values[key] = EDGE_OPTIONS[choice]

st.markdown("---")

# ── PREDICT BUTTON ───────────────────────────────────────────────────────────
predict_btn = st.button("🔬 Predict Adaptation", type="primary", use_container_width=True)

if predict_btn:
    result = predict(link_values)
    st.markdown("### Step 2 — Prediction Results")

    col_pred, col_motif = st.columns([1, 1])

    # ── LEFT: Prediction verdict ──
    with col_pred:
        if result['is_adaptive']:
            st.markdown(f"""
<div class="result-adaptive">
  <h2 style="color:white; margin:0">✅ ADAPTIVE</h2>
  <p style="color:#c8e6c9; font-size:18px; margin:8px 0 0">
    Confidence: <b>{result['probability']}%</b>
  </p>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-nonadaptive">
  <h2 style="color:white; margin:0">❌ NON-ADAPTIVE</h2>
  <p style="color:#ffcdd2; font-size:18px; margin:8px 0 0">
    Confidence: <b>{100 - result['probability']}%</b>
  </p>
</div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total edges", result['n_edges'])
        m2.metric("Activations", result['n_positive'])
        m3.metric("Inhibitions", result['n_negative'])

    # ── RIGHT: Motif detection ──
    with col_motif:
        motif = result['motif']
        motif_color = {
            "NFBLB": "#4fc3f7", "IFFLP": "#81c784",
            "Both": "#ce93d8", "None": "#ef9a9a"
        }[motif]
        st.markdown(f"""
<div class="metric-card">
  <p style="color:#90a4ae; margin:0; font-size:13px">DETECTED MOTIF</p>
  <h2 style="color:{motif_color}; margin:4px 0">{motif}</h2>
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="motif-box" style="margin-top:12px">
  <p style="color:#b0bec5; font-size:14px; margin:0">{result['motif_description']}</p>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── NETWORK VISUALIZATION ──
    col_net, col_feat = st.columns([1, 1])

    with col_net:
        st.markdown("#### Network diagram")
        fig, ax = plt.subplots(figsize=(4, 3.5))
        fig.patch.set_facecolor('#1e2130')
        ax.set_facecolor('#1e2130')

        # Node positions
        pos = {'A': (0.5, 0.85), 'B': (0.15, 0.2), 'C': (0.85, 0.2)}
        node_colors = {'A': '#4fc3f7', 'B': '#81c784', 'C': '#ef9a9a'}

        for node, (x, y) in pos.items():
            circle = plt.Circle((x, y), 0.1, color=node_colors[node], zorder=5)
            ax.add_patch(circle)
            ax.text(x, y, node, ha='center', va='center', fontsize=14,
                    fontweight='bold', color='black', zorder=6)

        arrowprops_act = dict(arrowstyle='->', color='#81c784', lw=2)
        arrowprops_inh = dict(arrowstyle='-[', color='#ef9a9a', lw=2)

        for src in NODES:
            for tgt in NODES:
                val = link_values[f"link_{src}{tgt}"]
                if val == 0:
                    continue
                sx, sy = pos[src]
                tx, ty = pos[tgt]
                if src == tgt:
                    # Self loop
                    color = '#81c784' if val == 1 else '#ef9a9a'
                    loop = mpatches.Arc((sx + 0.12, sy + 0.05), 0.15, 0.15,
                                        angle=0, theta1=0, theta2=300, color=color, lw=2)
                    ax.add_patch(loop)
                else:
                    color = '#81c784' if val == 1 else '#ef9a9a'
                    dx, dy = tx - sx, ty - sy
                    dist = (dx**2 + dy**2)**0.5
                    offset = 0.12
                    sx2 = sx + dx/dist * offset
                    sy2 = sy + dy/dist * offset
                    tx2 = tx - dx/dist * offset
                    ty2 = ty - dy/dist * offset
                    arrowstyle = '->' if val == 1 else '-['
                    ax.annotate("", xy=(tx2, ty2), xytext=(sx2, sy2),
                                arrowprops=dict(arrowstyle=arrowstyle, color=color,
                                                lw=2, mutation_scale=15))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(0.5, 0.0,
                "Green → Activation   |   Red ⊣ Inhibition",
                ha='center', va='bottom', fontsize=8, color='#90a4ae',
                transform=ax.transAxes)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── FEATURE IMPORTANCE ──
    with col_feat:
        st.markdown("#### Top structural features driving prediction")
        feature_names = [f[0] for f in result['top_features']]
        feature_vals  = [f[1] for f in result['top_features']]

        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        fig2.patch.set_facecolor('#1e2130')
        ax2.set_facecolor('#1e2130')

        colors = ['#4fc3f7' if v > np.mean(feature_vals) else '#546e7a' for v in feature_vals]
        bars = ax2.barh(range(len(feature_names)), feature_vals, color=colors, height=0.6)

        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels([n.replace('_', ' ') for n in feature_names],
                             color='#b0bec5', fontsize=10)
        ax2.set_xlabel("Feature importance (XGBoost)", color='#90a4ae', fontsize=9)
        ax2.tick_params(colors='#90a4ae', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#2d3145')
        ax2.xaxis.label.set_color('#90a4ae')

        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── PAPER CONTEXT ──
    st.markdown("---")
    st.info(
        "**Scientific context:** Ma et al. (2009) showed that across 16,038 possible 3-node topologies, "
        "only ~395 (~2.5%) achieve perfect adaptation under Michaelis-Menten kinetics. "
        "All adaptive topologies contain either NFBLB or IFFLP as a core motif. "
        "This model (AUC = 0.882) learns to predict adaptation from network structure alone, "
        "without running ODE simulations."
    )

# ── EXPLORE DATA TAB ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📊 Explore training data"):
    df = pd.read_csv("data/training_data.csv")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total topologies", len(df))
    col2.metric("Adaptive", int(df['is_adaptive'].sum()))
    col3.metric("Non-adaptive", int((df['is_adaptive']==0).sum()))
    col4.metric("Adaptive rate", f"{df['is_adaptive'].mean()*100:.1f}%")

    st.markdown("**Motif distribution in training set:**")
    motif_counts = df['topo_motif'].value_counts().reset_index()
    motif_counts.columns = ['Motif', 'Count']
    st.bar_chart(motif_counts.set_index('Motif'))

    st.markdown("**Sample rows:**")
    st.dataframe(df.head(20), use_container_width=True)
