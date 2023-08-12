# Import zipline functions
import matplotlib.pyplot as plt
# Import pandas and matplotlib to plot the results
import pandas as pd
from zipline.api import order_target, record, symbol
from zipline.finance import commission
import zipline

# Define the algorithm
def initialize(context):
    # Set the stock we want to trade
    context.stock = symbol('AAPL')

    # Set the moving average window
    context.window = 20

    # Set the commission fee
    context.set_commission(commission.PerShare(cost=0.01, min_trade_cost=0))


def handle_data(context, data):
    # Get the price history for the stock
    price_history = data.history(context.stock, 'price', context.window, '1d')

    # Calculate the simple moving average
    sma = price_history.mean()

    # Get the current price of the stock
    current_price = data.current(context.stock, 'price')

    # Get the current position of the stock
    current_position = context.portfolio.positions[context.stock].amount

    # Check if the current price is above or below the moving average
    if current_price > sma:
        # If we are not long, buy 100 shares of the stock
        if current_position == 0:
            order_target(context.stock, 100)
    elif current_price < sma:
        # If we are not short, sell 100 shares of the stock
        if current_position != 0:
            order_target(context.stock, 0)

    # Record the current price and moving average
    record(price=current_price, sma=sma)


# Load the stock data from yfinance
data = pd.read_csv('AAPL.csv', index_col='Date', parse_dates=True)

# Run the algorithm and get the results
results = zipline.run_algorithm(start=data.index[0],
                                end=data.index[-1],
                                initialize=initialize,
                                handle_data=handle_data,
                                capital_base=10000,
                                data=data)

# Plot the portfolio value, price and moving average
results[['portfolio_value', 'price', 'sma']].plot()
plt.show()
