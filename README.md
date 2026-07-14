<div align="center">

# 🏠 Real Estate Investment Advisor

### AI-powered property intelligence — predict investment quality & 5-year value growth in seconds

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit_Cloud-00D4B4?style=for-the-badge&logo=streamlit&logoColor=white)](https://real-estate-advisor-lvxveof4ymqie5ckxauudi.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-6C8CFF?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-C084FC?style=for-the-badge)](LICENSE)

<img src="https://img.shields.io/badge/scikit--learn-Pipelines-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/XGBoost-Regression-00D4B4?style=flat-square" />
<img src="https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
<img src="https://img.shields.io/badge/MLflow-Experiment_Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />

**[🔗 Try the live app →](https://real-estate-advisor-lvxveof4ymqie5ckxauudi.streamlit.app/)**

</div>

---

## ✨ Overview

**Real Estate Investment Advisor** is a full-stack ML web app that helps buyers and investors answer two questions in one click:

1. 🏷️ **Is this property a good investment?** — a classification model scores it Good / Not a Strong Investment with a confidence gauge.
2. 📈 **What will it be worth in 5 years?** — a regression model projects future value and expected growth %.

Built on top of Indian real-estate data, wrapped in a custom glassmorphism UI with animated gradients, floating orbs, and buttery-smooth micro-interactions — because a good model deserves a great front end.

---

## 🖥️ Preview

<div align="center">

| 🔮 Predict | 📈 Market Insights |
|:---:|:---:|
| Instant verdict + 5-yr price projection | City-wise pricing, distributions & filters |

| 🧠 Model Performance | ℹ️ About |
|:---:|:---:|
| Live accuracy, F1, RMSE, confusion matrix | Tech stack & methodology |

*(Add screenshots/GIFs here — drop them in a `/assets` folder and reference them, e.g. `![Predict Tab](assets/predict.png)`)*

</div>

---

## 🚀 Features

- 🎯 **Dual ML pipeline** — classification (Good/Bad investment) + regression (5-year price forecast), both served from a single form submission
- 📊 **Live-scored metrics** — Accuracy, F1, RMSE, MAE, R² recomputed on the fly from a held-out test split, not hardcoded
- 🗺️ **Market Insights dashboard** — top cities by price, price distributions, price/sqft by property type, transport-access vs investment rate
- 🔎 **Interactive data explorer** — filter by state, property type, BHK range, and price range across the full dataset
- 🎨 **Custom animated UI** — glassmorphism cards, gradient hero banner, floating orbs, shimmer effects, and a glowing confidence gauge, all hand-crafted in CSS
- 📱 **Responsive layout** — wide mode with a persistent sidebar snapshot of dataset stats
- ⚡ **Cached model & data loading** — `st.cache_resource` / `st.cache_data` for fast reruns

---

## 🧠 How It Works

```
Raw Housing Data (Kaggle)
        │
        ▼
Cleaning · Feature Engineering
(Price/SqFt, Age, School/Hospital Density, Transport Score)
        │
        ▼
┌────────────────────┬─────────────────────┐
│  Classification     │   Regression         │
│  (Good Investment)  │   (5-Year Price)      │
│  LogReg / RF / XGB   │   LogReg / RF / XGB   │
└────────────────────┴─────────────────────┘
        │
        ▼
Best pipelines saved via joblib + MLflow tracked
        │
        ▼
   Streamlit App (this repo)
```

The best-performing pipeline for each task is selected automatically and serialized with `joblib`; the Streamlit app simply loads and serves them.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **Modeling** | scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express & Graph Objects |
| **Frontend / App** | Streamlit + custom CSS (glassmorphism, animations) |
| **Deployment** | Streamlit Community Cloud |

---

## 📂 Project Structure

```
real-estate-app/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Python version pin for deployment
├── models/
│   ├── best_classification_pipeline.pkl
│   ├── best_regression_pipeline.pkl
│   └── model_metadata.pkl          # Feature lists + best-model names
├── data/
│   └── processed_real_estate_data.csv
└── .streamlit/                     # Streamlit config (theme, etc.)
```

---

## ⚙️ Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/D0027/real-estate-app.git
cd real-estate-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model & data files

Place the following inside their respective folders (see the notebook / Kaggle source for how they're generated):

```
models/best_classification_pipeline.pkl
models/best_regression_pipeline.pkl
models/model_metadata.pkl
data/processed_real_estate_data.csv
```

### 4. Run locally

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🌐 Live Demo

> ### 👉 **[real-estate-advisor-lvxveof4ymqie5ckxauudi.streamlit.app](https://real-estate-advisor-lvxveof4ymqie5ckxauudi.streamlit.app/)**

No installation needed — just open the link, fill in a property's details, and hit **🚀 Analyze Property**.

---

## 📊 Model Performance

Metrics are computed **live inside the app** (Model Performance tab) from an 80/20 held-out split — not hardcoded — so numbers always reflect the currently deployed pipeline. Expect to see:

- ✅ Classification: Accuracy, F1-Score, Confusion Matrix
- 📈 Regression: RMSE, MAE, R², Actual vs. Predicted scatter plot

---

## 🗺️ Roadmap

- [ ] Add SHAP-based explainability for individual predictions
- [ ] Map view of properties by lat/long
- [ ] User accounts to save & compare past predictions
- [ ] Export analysis report as PDF

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

### Built with ❤️ using Streamlit · Scikit-learn · Plotly

**[⭐ Star this repo](https://github.com/D0027/real-estate-app)** if you found it useful!

</div>
