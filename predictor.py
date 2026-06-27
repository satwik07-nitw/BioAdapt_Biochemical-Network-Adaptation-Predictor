"""
predictor.py — BioAdapt core logic
Feature engineering + XGBoost inference for 3-node biochemical network topologies.
Based on: Ma et al. (2009), "Defining Network Topologies that Can Achieve Biochemical Adaptation"
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"

# Edge encoding: -1 = inhibition, 0 = no edge, 1 = activation
LINK_COLS = ['link_AA','link_AB','link_AC','link_BA','link_BB',
             'link_BC','link_CA','link_CB','link_CC']

MOTIF_LABELS = {
    (1, 0): "NFBLB",   # has nfblb only
    (0, 1): "IFFLP",   # has ifflp only
    (1, 1): "Both",    # has both
    (0, 0): "None",    # neither
}

MOTIF_DESCRIPTIONS = {
    "NFBLB": "Negative Feedback Loop with Buffer node — the primary mechanism for perfect adaptation in E. coli chemotaxis.",
    "IFFLP": "Incoherent Feedforward Loop — achieves adaptation via opposing fast and slow pathways.",
    "Both":  "Contains both NFBLB and IFFLP motifs — maximum adaptation potential.",
    "None":  "No canonical adaptation motif detected — adaptation unlikely via known mechanisms.",
}

def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

MODEL = _load_model()


def engineer_features(link_values: dict) -> pd.DataFrame:
    """
    Takes a dict of {link_XX: int} where values are -1, 0, or 1.
    Returns a single-row DataFrame with all engineered features.
    """
    row = {col: link_values.get(col, 0) for col in LINK_COLS}
    d = {}

    for col in LINK_COLS:
        d[col] = row[col]

    links = list(row.values())
    d['n_links_total']    = sum(1 for v in links if v != 0)
    d['n_links_positive'] = sum(1 for v in links if v == 1)
    d['n_links_negative'] = sum(1 for v in links if v == -1)
    d['selfloop_A'] = int(row['link_AA'] != 0)
    d['selfloop_B'] = int(row['link_BB'] != 0)
    d['selfloop_C'] = int(row['link_CC'] != 0)
    d['n_selfloops'] = d['selfloop_A'] + d['selfloop_B'] + d['selfloop_C']
    d['C_to_B_neg']    = int(row['link_CB'] == -1)
    d['C_to_B_pos']    = int(row['link_CB'] ==  1)
    d['B_to_C_exists'] = int(row['link_BC'] != 0)
    d['A_to_C_pos']    = int(row['link_AC'] ==  1)
    d['A_to_C_neg']    = int(row['link_AC'] == -1)
    d['A_to_B_exists'] = int(row['link_AB'] != 0)
    d['B_to_A_exists'] = int(row['link_BA'] != 0)
    d['nfblb_motif']   = int(row['link_AB'] != 0 and row['link_BC'] != 0 and row['link_CB'] == -1)
    d['ifflp_motif']   = int(row['link_AB'] == 1 and row['link_AC'] != 0 and row['link_BC'] == -1)
    d['has_neg_feedback'] = int(any(v == -1 for v in links))
    d['pos_feedback']  = sum(1 for v in links if v == 1)
    d['frac_negative'] = d['n_links_negative'] / (d['n_links_total'] + 1e-6)

    # Topo motif derived from NFBLB/IFFLP detection
    has_nfblb = d['nfblb_motif']
    has_ifflp = d['ifflp_motif']
    d['is_nfblb'] = int(has_nfblb and not has_ifflp or (has_nfblb and has_ifflp))
    d['is_ifflp'] = int(has_ifflp)

    return pd.DataFrame([d])


def detect_motif(link_values: dict) -> str:
    nfblb = int(
        link_values.get('link_AB', 0) != 0 and
        link_values.get('link_BC', 0) != 0 and
        link_values.get('link_CB', 0) == -1
    )
    ifflp = int(
        link_values.get('link_AB', 0) == 1 and
        link_values.get('link_AC', 0) != 0 and
        link_values.get('link_BC', 0) == -1
    )
    return MOTIF_LABELS[(nfblb, ifflp)]


def predict(link_values: dict) -> dict:
    """
    Main prediction function.
    Input:  dict of link values, e.g. {'link_AB': 1, 'link_CB': -1, ...}
    Output: dict with prediction, probability, motif, feature importances
    """
    X = engineer_features(link_values)
    prob = float(MODEL.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)
    motif = detect_motif(link_values)

    # Top feature importances
    importances = dict(zip(X.columns, MODEL.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:6]

    return {
        "is_adaptive": pred,
        "probability": round(prob * 100, 1),
        "motif": motif,
        "motif_description": MOTIF_DESCRIPTIONS[motif],
        "top_features": top_features,
        "n_edges": X['n_links_total'].iloc[0],
        "n_positive": X['n_links_positive'].iloc[0],
        "n_negative": X['n_links_negative'].iloc[0],
    }
