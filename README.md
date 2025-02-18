# Stock Price Prediction Using K-Nearest Neighbors (KNN)

## Overview
This project implements **Stock Price Prediction** using the **K-Nearest Neighbors (KNN) algorithm**. The model uses historical stock price data to predict future closing prices based on past trends.

## Features
- Fetches historical stock data using `yfinance`.
- Preprocesses data using MinMax scaling.
- Uses a rolling window approach for feature extraction.
- Trains a **KNN regression model** to predict future stock prices.
- Evaluates performance using MAE and RMSE.
- Visualizes actual vs. predicted stock prices.

## Installation
### Prerequisites
Ensure you have Python installed (>= 3.7) and install the required libraries using:
```bash
pip install numpy pandas scikit-learn yfinance matplotlib
```

## Usage
### 1. Clone the repository
```bash
git clone https://github.com/your-repo/knn-stock-prediction.git
cd knn-stock-prediction
```

### 2. Run the script
Execute the script to train and test the model:
```bash
python knn_stock_prediction.py
```

### 3. Modify stock symbol and parameters
You can change the stock symbol and window size in the script:
```python
stock_data = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
window_size = 5  # Modify as needed
k = 5  # Number of neighbors
```

## Implementation Details
1. **Load Data**: Fetches stock price data from Yahoo Finance.
2. **Preprocess Data**: Normalizes stock prices using MinMaxScaler.
3. **Feature Engineering**: Uses past `n` days' prices as input features.
4. **Train KNN Model**: Fits the KNN regressor using training data.
5. **Predict & Evaluate**: Predicts future prices and computes MAE & RMSE.
6. **Visualization**: Plots actual vs. predicted stock prices.

## Example Output
```
Mean Absolute Error: 3.25
Root Mean Squared Error: 4.78
```
The visualization shows actual vs. predicted stock prices over time.

## Limitations & Future Improvements
- **KNN is sensitive to feature scaling**; try different normalization techniques.
- **Limited long-term accuracy**; consider LSTM or other deep learning models.
- **Optimize hyperparameters** (`k` value, window size) for better performance.



