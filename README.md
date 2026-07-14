# 🏠 Real Estate Investment Advisor — Streamlit App

AI-powered Streamlit app that predicts whether a property is a **good investment**
and estimates its **price 5 years from now**, using models trained in your Kaggle notebook.

## 1. What to download from Kaggle

Open your Kaggle notebook's **Output** panel (the one in your screenshot: `/kaggle/working`) and download these 4 files:

| From Kaggle `/kaggle/working/...`        | Put it in this project folder |
|-------------------------------------------|--------------------------------|
| `models/best_classification_pipeline.pkl` | `models/`                      |
| `models/best_regression_pipeline.pkl`     | `models/`                      |
| `models/model_metadata.pkl`               | `models/`                      |
| `processed_real_estate_data.csv`          | `data/`                        |

How to download from Kaggle:
1. In the notebook Output panel, click the `models` folder → download each `.pkl` file individually
   (or click the copy/download icon next to the `models` folder to grab it as a zip, then extract).
2. Click the download icon next to `processed_real_estate_data.csv`.
3. Move the 3 `.pkl` files into this project's `models/` folder.
4. Move `processed_real_estate_data.csv` into this project's `data/` folder.

You do **not** need `mlflow.db` or `mlruns/` for the app to run — those are only for viewing
experiment logs locally with `mlflow ui` if you want to.

## 2. Project structure

```
real-estate-app/
├── app.py                     # Main Streamlit app
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # Dark theme config
├── models/
│   ├── best_classification_pipeline.pkl   ← you add this
│   ├── best_regression_pipeline.pkl       ← you add this
│   └── model_metadata.pkl                 ← you add this
└── data/
    └── processed_real_estate_data.csv     ← you add this
```

## 3. Run it locally

```bash
# 1. Create & activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## 4. Features

- **Predict tab** — full property input form → Good Investment verdict + confidence gauge +
  5-year price projection chart, all with smooth animated UI.
- **Market Insights tab** — city/price/property-type charts, transport vs. investment analysis,
  filterable raw data table.
- **Model Performance tab** — shows which algorithm won for classification & regression.
- **About tab** — tech stack & methodology summary.
- Dark glassmorphism theme, animated hero header, hover-lift cards, animated gradient background.

## 5. Notes

- If `models/` or `data/` files are missing, the app still loads and shows a clear warning
  telling you exactly what to add — it won't crash.
- To view full MLflow experiment logs locally, copy the `mlruns/` folder from Kaggle output
  into this project and run `mlflow ui` from this directory.
