# 🏗️ MatRisk AI Lab
**AI-Driven Infrastructure & Commodity Simulation Engine**

A Streamlit dashboard that uses a Random Forest model to predict Remaining Useful Life (RUL) of bridge infrastructure, simulate stress scenarios, and quantify financial risk across material portfolios.

---

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📸 Features

- **Portfolio Risk Dashboard** — KPI cards for NPV, Average RUL, Cost of Quality, and Assets at Risk
- **Stress Scenario Engine** — Simulate climate-accelerated corrosion, steel quality degradation, and supply crises
- **AI-Powered RUL Prediction** — Random Forest model trained on historical failure data
- **Interactive Visualizations** — Degradation curves, treemaps by topology & material
- **Asset-Level Risk Table** — Sortable, color-coded table for the top 20 at-risk assets

---

## 📁 Project Structure

```
MatRisk-AI-Lab/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
│
├── DS1_material_properties_5500.csv
├── DS2_commodity_prices_10yr.csv
├── DS3_infrastructure_bridges_5000.csv
├── DS4_crossdomain_features_daily.csv
├── DS5_element_prices_monthly.csv
├── DS6_historical_failures_2000.csv
├── MatRisk_AI_Sample_Datasets.xlsx
│
└── README.md
```

---

## ⚙️ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/matrisk-ai-lab.git
cd matrisk-ai-lab

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

---

## 📊 Datasets

| File | Description |
|------|-------------|
| `DS1_material_properties_5500.csv` | Material strength & corrosion properties |
| `DS2_commodity_prices_10yr.csv` | 10-year commodity price history |
| `DS3_infrastructure_bridges_5000.csv` | Bridge portfolio with age, material, cost |
| `DS4_crossdomain_features_daily.csv` | Cross-domain daily feature set |
| `DS5_element_prices_monthly.csv` | Monthly element/raw material prices |
| `DS6_historical_failures_2000.csv` | Historical failure events used to train the model |

> **Note:** If dataset files are missing, the app automatically falls back to synthetic data so you can still explore the dashboard.

---

## 🧠 Model

- **Algorithm:** Random Forest Regressor (`scikit-learn`)
- **Target:** Predicted failure age (years)
- **Features:** Material type (encoded), corrosion rate
- **Training data:** `DS6_historical_failures_2000.csv`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Streamlit | Web dashboard |
| Pandas / NumPy | Data processing |
| Scikit-learn | ML model |
| Plotly | Interactive charts |

---

## 📄 License

MIT License — feel free to use and adapt.
