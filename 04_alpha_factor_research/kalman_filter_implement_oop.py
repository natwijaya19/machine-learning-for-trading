from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf
from numpy import ndarray
from pandas import DataFrame
from scipy.linalg import inv

# set style for seaborn plots
sns.set_style('darkgrid')


class KalmanFilter:
    """
    Kalman filter implementation.
    """

    def __init__(self, initial_state_mean: float, initial_covariance: float,
                 process_noise: float, measurement_noise: float,
                 state_transition: np.ndarray, measurement_matrix: np.ndarray):
        """
        Initialize the Kalman filter.

        Parameters:
            initial_state_mean (float): Initial estimate of the state mean.
            initial_covariance (float): Initial estimate of the state
            covariance.
            process_noise (float): Covariance matrix of process noise.
            measurement_noise (float): Covariance matrix of measurement noise.
            state_transition (np.ndarray): State transition matrix.
            measurement_matrix (np.ndarray): Measurement matrix.
        """
        self.x = np.array([initial_state_mean])
        self.P = np.eye(1) * initial_covariance
        self.Q = np.eye(1) * process_noise
        self.R = np.eye(1) * measurement_noise
        self.F = state_transition
        self.H = measurement_matrix

    def kalman_filter_step(self, z: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform a single step of the Kalman filter.

        Parameters:
            z (float): Measurement value.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated state mean and covariance.
        """
        # Predict
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q

        # Update
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, inv(S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

        return self.x, self.P


def main():
    # Define the stock ticker
    ticker: str = "MSFT"
    start: str = "2020-01-01"
    end: str = "2023-08-01"

    # Download stock prices from Yahoo Finance
    stock_data: pd.DataFrame = yf.download(ticker, start=start, end=end)

    # Select the 'Adj Close' prices for filtering
    prices: ndarray = stock_data['Adj Close'].values

    # Initialize Kalman filter
    initial_state_mean: float = prices[0]
    initial_covariance: float = 1.0
    process_noise: float = 1e-5
    measurement_noise: float = 1e-3
    state_transition: ndarray = np.array(
        [[1]])  # State transition matrix (constant model)
    measurement_matrix: ndarray = np.array(
        [[1]])  # Measurement matrix (identity, direct measurement)

    # Create Kalman filter object
    kf: KalmanFilter = KalmanFilter(initial_state_mean, initial_covariance,
                                    process_noise,
                                    measurement_noise, state_transition,
                                    measurement_matrix)

    # Apply Kalman filter to smooth out the prices
    filtered_state_means: List[float] = []
    t: int | None = None
    filtered_state_mean: ndarray | None = None
    for t in range(len(prices)):
        filtered_state_mean, _ = kf.kalman_filter_step(z=prices[t])
        filtered_state_means.append(float(filtered_state_mean))

    # Create a DataFrame with the original and filtered prices
    filtered_data: DataFrame = pd.DataFrame(
        data={'Date': stock_data.index, 'Original': prices,
              'Filtered': filtered_state_means},
        index=stock_data.index)

    # Plot the original and filtered prices
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['Date'], filtered_data['Original'],
             label='Original Prices', color='blue')
    plt.plot(filtered_data['Date'], filtered_data['Filtered'],
             label='Filtered Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'{ticker} Stock Prices - Original vs Filtered (Kalman Filter)')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
