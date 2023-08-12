import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from numpy import ndarray

sns.set_style('darkgrid')

## Download MSFT stock price data from Yahoo Finance
tickers = ["MSFT"]
data_df: pd.DataFrame = yf.download(tickers, start='2020-01-01',
                                    end='2023-01-01')
observations: ndarray = np.array(data_df['Close'])

# Create the Kalman filter model
kf: KalmanFilter = KalmanFilter(dim_x=2, dim_z=1)

# Define the state transition matrix (constant velocity model)
kf.F = np.array([[1., 1.],
                 [0., 1.]])

# Define the measurement function (observe position)
kf.H = np.array([[1., 0.]])

# Define the process noise covariance matrix
kf.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.001)

# Define the measurement noise covariance matrix
kf.R = np.array([[0.1]])

# Set the initial state
kf.x = np.array([[observations[0]], [0]])

# Set the initial state covariance matrix
kf.P = np.array([[1., 0.],
                 [0., 1.]])

# Initialize lists to store the smoothed state and covariance
smoothed_states: list[None] = []
smoothed_covariances: list[None] = []

# Apply the Kalman filter to smooth the observations
for measurement in observations:
    kf.predict()
    kf.update(measurement)
    smoothed_states.append(kf.x[0, 0])
    smoothed_covariances.append(kf.P[0, 0])

# Plot the original and smoothed prices
plt.figure(figsize=(10, 5))
plt.plot(data_df.index, observations, label='Original Prices', color='blue')
plt.plot(data_df.index, smoothed_states, label='Smoothed Prices', color='red')
plt.fill_between(data_df.index,
                 np.array(smoothed_states) - np.sqrt(smoothed_covariances),
                 np.array(smoothed_states) + np.sqrt(smoothed_covariances),
                 color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"Kalman Filter vs Original Prices for {tickers[0]}")
plt.legend()
plt.xticks(rotation=45)
plt.show()

## compare to moving average of 1,2,3,4
periods = [5.10, 20, 40, 60]
for period in periods:
    data_df[f"SMA_{period}"] = (data_df['Close'].rolling(
        window=int(period)).mean())

# Put all including Kalman Filter into one dataframe for plotting
data: pd.DataFrame = data_df['Close'].to_frame()
for period in periods:
    data[f"SMA_{period}"] = data_df[f"SMA_{period}"]

data['Kalman Filter'] = smoothed_states

# plot for period 2020-01-01 to 2020-06-30
start: str = '2020-01-01'
end: str = '2020-06-30'

data[start:end].plot(figsize=(10, 6))
plt.title(f'Kalman Filter vs Moving Average of {tickers[0]}')
plt.legend()
plt.tight_layout()
plt.show()

# Plot for period 2020-01-01 to 2020-06-30 for kalman filter and close price
# only
data[start:end][['Close', 'Kalman Filter']].plot(figsize=(10, 6))
plt.title(f"Kalman Filter vs Close Price for {start} to {end} of {tickers[0]}")
plt.legend()
plt.tight_layout()
plt.show()
