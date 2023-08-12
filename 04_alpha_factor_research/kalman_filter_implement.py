from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from numpy import ndarray
from pandas import DataFrame
from scipy.linalg import inv

# Download MSFT stock prices from Yahoo Finance
msft_data = yf.download("MSFT", start="2020-01-01", end="2023-08-01")

# Select the 'Adj Close' prices for filtering
prices = msft_data['Adj Close'].values


# Kalman filter implementation
def kalman_filter(x: np.ndarray, P: np.ndarray, F: np.ndarray, Q: np.ndarray,
                  H: np.ndarray, R: np.ndarray, z: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    This function implements the Kalman Filter for a single measurement update
    :param x: State vector
    :param P: Covariance matrix
    :param F: State transition matrix
    :param Q: Process noise covariance
    :param H: Measurement matrix
    :param R: Measurement noise covariance
    :param z: Measurement
    :return: Updated state vector and covariance matrix

    example:
    x = np.array([0.])
    P = np.eye(1)
    F = np.array([[1]])
    Q = np.eye(1) * 1e-5
    H = np.array([[1]])
    R = np.eye(1) * 1e-3
    z = 1.1

    x, P = kalman_filter(x, P, F, Q, H, R, z)

    """
    # Predict
    x = np.dot(F, x)
    P = np.dot(F, np.dot(P, F.T)) + Q

    # Update
    y = z - np.dot(H, x)
    S = np.dot(H, np.dot(P, H.T)) + R
    K = np.dot(P, np.dot(H.T, inv(S)))
    x = x + np.dot(K, y)
    P = P - np.dot(K, np.dot(H, P))

    return x, P


# Initial state and covariance matrix
x: ndarray = np.array([prices[0]])  # Initial state
P: np.ndarray = np.eye(1)  # Initial covariance matrix

# Process and measurement noise
Q: np.ndarray = np.eye(1) * 1e-5  # Process noise covariance
R: np.ndarray = np.eye(1) * 1e-3  # Measurement noise covariance

# State transition matrix and measurement matrix
F: ndarray = np.array([[1]])  # State transition matrix (constant model)
H: ndarray = np.array(
    [[1]])  # Measurement matrix (identity, direct measurement)

# Apply Kalman filter to smooth out the prices
filtered_state_means = np.zeros_like(prices)
t: int
for t in range(len(prices)):
    x, P = kalman_filter(x, P, F, Q, H, R, z=prices[t])
    filtered_state_means[t] = x[0]

# Create a DataFrame with the original and filtered prices
filtered_data: DataFrame = pd.DataFrame(
    data={'Date': msft_data.index, 'Original': prices,
          'Filtered': filtered_state_means},
    index=msft_data.index)

# Plot the original and filtered prices
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['Date'], filtered_data['Original'],
         label='Original Prices', color='blue')
plt.plot(filtered_data['Date'], filtered_data['Filtered'],
         label='Filtered Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('MSFT Stock Prices - Original vs Filtered (Kalman Filter)')
plt.legend()
plt.grid(True)
plt.show()
