# Travel Cost Prediction


## Problem Statement

Predict total travel cost based on trip characteristics such as destination, duration, transport type, accommodation, season, and traveler profile.

This is a **supervised regression** problem.

---

## Tech Stack

* **Python**
* **scikit‑learn** (RandomForest / XGBoost style models)
* **pandas / numpy**


## Project Structure

```
Travel_Cost_Prediction/
│
├── app.py                # Flask API inference service
├── train.py              # Model training pipeline entrypoint
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── data_loader.py    # Data ingestion & validation
│   ├── features.py       # Feature engineering
│   ├── model.py          # Model training & evaluation
│   └── utils.py          # Common helpers
│
└── tests/                # Unit tests (optional but recommended)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Karishma-Srivastava/Travel_Cost_Prediction.git
cd Travel_Cost_Prediction
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train.py
```

What happens:

* Loads data
* Runs feature engineering
* Trains model

---

## Run Inference API

```bash
uvicorn app:app --reload
```

Open: `http://127.0.0.1:8000/docs`

Example request:

```json
{
  "destination": "Paris",
  "duration_days": 7,
  "transport": "flight",
  "hotel_rating": 4,
  "season": "summer",
  "traveler_type": "solo"
}
```


---

## Future Improvements

* Hyperparameter tuning with Optuna
* Feature store integration
* CI pipeline for training & testing
* Cloud deployment (AWS/GCP/Azure)

---

## Author

Karishma Srivastava


---

## License

MIT

