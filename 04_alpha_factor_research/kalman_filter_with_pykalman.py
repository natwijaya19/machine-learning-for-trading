import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import talib
import yfinance as yf
from numpy import ndarray
from pykalman import KalmanFilter

sns.set_style('darkgrid')

## Download historical data for required stocks
tickers = ["MSFT", "AAPL", ]

ohlcv_data: dict[str, pd.DataFrame] = {}
start_date: str = '2015-01-01'
end_date: str = '2020-12-31'
for ticker in tickers:
    ohlcv_data[ticker] = yf.download(ticker, start_date, end_date)

## Create kalman filter
symbol: str = 'MSFT'
data: pd.DataFrame = ohlcv_data[symbol]['Adj Close'].to_frame()
kf: KalmanFilter = KalmanFilter(transition_matrices=[1],
                                observation_matrices=[1],
                                initial_state_mean=0,
                                initial_state_covariance=1,
                                observation_covariance=1,
                                transition_covariance=0.01)

# state_means: object
data_values: ndarray = data.values
state_means, _ = kf.filter(data_values[:, None])

data['Kalman Filter'] = state_means

# Create moving averages for period 1,2,3,4
periods: list[int] = [1, 2, 3, 4]
for period in periods:
    data["SMA_" + str(period)] = talib.SMA(data['Adj Close'].values,
                                           timeperiod=period)

# Plot adjusted close price, SMA, and Kalman Filter
data[['Adj Close', 'SMA_1', 'SMA_2', 'SMA_3', 'SMA_4', 'Kalman Filter']].plot(
    figsize=(10, 6))
plt.title('Kalman Filter vs Moving Average')
plt.legend()
plt.tight_layout()
plt.show()
