# Pavement Performance Inference (XGBoost-Based)

This project performs predictive inference of pavement performance indicators — such as **IRI**, **PSI**, **D0**, **IDS**, and **ATRMED** — using pre-trained **XGBoost** models.  
It is part of a broader research on long-term pavement performance modeling and deterioration analysis.

---

## 📘 Overview

The script reads a dataset containing historical measurements of pavement indicators, structural and traffic data, and maintenance summaries.  
For each year and each target variable, it:
1. Loads the corresponding trained XGBoost model and scaler.
2. Predicts the next year's indicator values.
3. Applies constraints (e.g., blocks unjustified reductions or increases without maintenance).
4. Updates the dataset and generates evolution plots for each road group.

This process simulates the temporal deterioration of pavement sections year-by-year.

---

## 🧩 Features

- Automatic loading of pre-trained XGBoost models and scalers  
- Support for multiple indicators (PSI, IRI, D0, IDS, ATRMED)  
- Conditional rules to prevent unrealistic improvements or deteriorations  
- Sequential inference across all pavement ages (2–30 years)  
- Export of results to Excel  
- Generation of evolution charts (`.jpg`) by group and variable  

---

## 🧠 Requirements

The script uses only a few essential Python libraries:

```txt
numpy
pandas
xgboost
joblib
matplotlib
```

You can install them manually or via:
```txt
pip install -r requirements.txt
```

---

## 🧮 How It Works (Methodology)

1. Filter by Year → Selects rows matching the current Idade.
2. Load Scaler & Model → Loads .pkl (scaler) and .json (XGBoost model).
3. Scale Inputs → Normalizes feature columns.
4. Predict → Uses xgboost.Booster.predict() to generate next-year values.
5. Apply Constraints → Blocks unrealistic variations if no reinforcement/cut occurred.
6. Write Back → Inserts inferred values into next-year rows.
7. Iterate → Repeats for years 2 to 30 for all targets.

---

## 📂 Project Structure Example
```
📁 Pavement_Inference/
├── main_inference.py
├── requirements.txt
├── _Dados sinteticos do BD.xlsx
├── _Dados sinteticos - Processado.xlsx
├── PSI.jpg
├── D0.jpg
├── IRI-convertido.jpg
├── ATRMED.jpg
├── IDS.jpg
└── /Output dos treinamento/
     ├── XGB-*.json
     └── XGB-*-scaler_x.pkl
```

---

## 🧑‍🔬 Author

Vinícius Camillo
Pavement Performance Modeling Research
Master’s Dissertation – Pavement Engineering
📍 Brazil, 2025

---

## How to cite
Camillo, V. C., & Brito, L. T. (2025). Development of flexible pavement performance prediction models for the states of RS and PR using artificial intelligence (1.0). Zenodo. https://doi.org/10.5281/zenodo.17281701

---

## 🧱 License

This project is distributed for academic and research purposes.
For commercial or redistribution rights, please contact the author.

