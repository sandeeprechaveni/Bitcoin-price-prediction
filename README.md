# 📈 Bitcoin Price Prediction

A Python-based project that uses **Machine Learning** to predict the future price of **Bitcoin (BTC)** based on historical price data and technical indicators.

---

## 🧠 Overview

This project explores various ML models to forecast the price of Bitcoin using time-series data. It includes preprocessing, feature engineering, model training, evaluation, and prediction visualization.

---

## 🚀 Features

- ⏳ Time-series forecasting using historical BTC data
- 🧮 Feature engineering with technical indicators (Moving Averages, RSI, etc.)
- 🧠 ML models: Linear Regression, Random Forest, XGBoost, LSTM (optional)
- 📊 Visualization of predictions vs actual prices
- 💾 Supports CSV historical data inputs

---

## 🛠 Tech Stack

- **Language:** Python 3
- **Libraries:**
  - `pandas`, `numpy`
  - `scikit-learn`
  - `matplotlib`, `seaborn`
  - `xgboost` (for boosting models)
  - `tensorflow` or `keras` (optional, for LSTM)

---

## 📁 Project Structure
bitcoin-price-prediction/
│
├── data/ # Historical BTC price data
├── models/ # Saved models
├── notebooks/ # Jupyter notebooks for EDA & modeling
├── src/
│ ├── preprocess.py # Data cleaning and feature engineering
│ ├── train_model.py # Model training scripts
│ ├── predict.py # Make predictions on new data
│ └── utils.py # Utility functions
├── requirements.txt
└── README.md

---

## 🧪 How to Use

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/bitcoin-price-prediction.git
cd bitcoin-price-prediction
```
2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
3. Train the model
```bash
python src/train_model.py --input data/BTC-USD.csv
```
4. Make predictions
```bash
python src/predict.py --model models/btc_model.pkl --input data/latest.csv


