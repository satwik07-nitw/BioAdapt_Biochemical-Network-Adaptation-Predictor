# BioAdapt 🧬

**Biochemical Network Adaptation Predictor**

A machine learning web app that predicts whether a 3-node biochemical signaling network topology achieves **perfect adaptation** — the ability of a biological system to respond to a stimulus and then return to its pre-stimulus output level.

🔗 **Live Demo:** [bioadapt.streamlit.app](https://bioadapt.streamlit.app) *(deploy to update)*

---

## Background

This project replicates and extends **Ma et al. (2009)** — *"Defining Network Topologies that Can Achieve Biochemical Adaptation"* (Cell, 138:760-773).

The paper exhaustively simulated all **16,038** possible 3-node signaling topologies using Michaelis-Menten ODEs and found that only ~395 (~2.5%) achieve perfect adaptation. All adaptive topologies contain one of two core motifs:

- **NFBLB** — Negative Feedback Loop with Buffer node *(used in E. coli chemotaxis)*
- **IFFLP** — Incoherent Feedforward Loop with a Proportioner node

This app trains an XGBoost classifier on ODE-verified labels and asks: **can network structure alone predict adaptation, without running expensive ODE simulations?**

---

## What this project demonstrates

| Stage | What was done |
|---|---|
| Data generation | Enumerated all 3^9 = 19,683 topologies (filtered to 16,038 biologically valid), encoded as 9-element edge vectors |
| ODE simulation | Ran Michaelis-Menten ODEs (LSODA/Radau solvers via SciPy) on pre-filtered topologies (NFBLB/IFFLP candidates), identifying **358 adaptive** out of 1,400 simulated |
| Feature engineering | Derived 30 structural graph features: edge counts, motif indicators, feedback ratios, self-loop flags |
| ML training | XGBoost with class-imbalance weighting; **structural-only model AUC = 0.882** |
| Ablation test | ODE-feature model (AUC = 1.000) vs structural-only (AUC = 0.882); ΔF1 = 0.337 quantifies domain knowledge value |
| Deployment | Streamlit app with real-time topology input, motif detection, network visualization, feature importance |

---

## Model performance

```
Structural-only XGBoost (no ODE features):
  AUC  = 0.882
  F1   = 0.708
  Prec = 0.64   Recall = 0.79

ODE-feature XGBoost (upper bound / leaky):
  AUC  = 1.000
  F1   = 1.000
```

The gap (ΔF1 = 0.337) quantifies how much biological domain knowledge (ODE simulation outputs) adds over blind structural learning — the core scientific finding of the ablation experiment.

---

## Project structure

```
bioadapt/
├── app.py              # Streamlit frontend
├── predictor.py        # Feature engineering + XGBoost inference
├── requirements.txt    # Python dependencies
├── model/
│   └── model.pkl       # Trained XGBoost model
└── data/
    └── training_data.csv   # ODE simulation results (1,400 topologies)
```

---

## Run locally

```bash
git clone https://github.com/satwik07-nitw/bioadapt
cd bioadapt
pip install -r requirements.txt
streamlit run app.py
```

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set `app.py` as the entry point
4. Done — live URL in ~2 minutes

---

## Tech stack

- **ML:** XGBoost, Scikit-learn
- **Data:** Pandas, NumPy, SciPy (ODE simulation)
- **Viz:** Matplotlib
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

## Scientific reference

Ma, W., Trusina, A., El-Samad, H., Lim, W. A., & Tang, C. (2009).  
*Defining network topologies that can achieve biochemical adaptation.*  
Cell, 138(4), 760–773. https://doi.org/10.1016/j.cell.2009.09.047

---

*Built as part of a research internship under Dr. Anbumathi P, Dept. of Biotechnology, NIT Warangal.*  
*Author: Satwik Kumar Yadav | [GitHub](https://github.com/satwik07-nitw) | [LinkedIn](https://linkedin.com)*
