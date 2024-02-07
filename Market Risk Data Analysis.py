#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data
market_data = pd.read_csv('market_data.csv', index_col='Date', parse_dates=True)

# Ensure complete index (handle missing dates if needed)
market_data = market_data.dropna()

# Set num_periods based on actual data length
num_periods = len(market_data.index)  # Assuming you want to visualize all data

# Calculate moving average
window = 20  # Adjust window size as needed
market_data['MA'] = market_data['Close'].rolling(window=window).mean()

# Calculate volatility
volatility = np.std(market_data['Close'])
print(f"Volatility: {volatility:.4f}")

# Calculate returns
market_data['Returns'] = market_data['Close'].pct_change()

# Analyze specific stocks (optional)
aapl_data = market_data[market_data['Ticker'] == 'AAPL']

# Visualizations
plt.plot(market_data.index, market_data['Close'], label='Close Price')
plt.plot(market_data.index, market_data['MA'], label='Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price')
plt.legend()
plt.show()


# Monte Carlo Simulation
num_simulations = 1000
# Ensure num_periods reflects actual data length
num_periods = len(market_data.index)  # Assuming all data is used

initial_price = market_data['Close'].iloc[-1]

simulated_prices = np.zeros((num_periods, num_simulations))
simulated_returns = np.zeros((num_periods, num_simulations))

for i in range(num_simulations):
    returns = np.random.normal(0, volatility, num_periods)  # Assuming normally distributed returns
    prices = initial_price * np.exp(np.cumsum(returns))
    simulated_prices[:, i] = prices
    simulated_returns[:, i] = returns

# Visualize simulated price paths (optional)
percentiles = [0.05, 0.5, 0.95]  # Choose desired percentiles
for percentile in percentiles:
    price_percentile = np.percentile(simulated_prices, percentile, axis=1)
    plt.plot(market_data.index[-num_periods:], price_percentile, label=f"{percentile:.2f}th Percentile")

plt.plot(market_data.index[-num_periods:], market_data['Close'].iloc[-num_periods:], label='Historical Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price (Simulated Percentiles)')
plt.legend()
plt.show()


# 1. Distribution of Returns
plt.hist(market_data['Returns'])
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.title('Distribution of AAPL Stock Returns')
plt.show()

# 2. Volatility Over Time
plt.figure(figsize=(12, 6))
plt.plot(market_data.index, market_data['Close'], label='Close Price')
plt.plot(market_data.index, market_data['MA'], label='Moving Average')
plt.fill_between(market_data.index, market_data['Close'] + volatility, market_data['Close'] - volatility, alpha=0.2, label='Volatility Band')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('AAPL Stock Price with Volatility Band')
plt.legend()
plt.show()


print(f"Volatility: {volatility:.4f}")

print("Market data analysis completed. Please refer to the results and visu")


# Value at Risk (VaR) Calculation
confidence_level = 0.95
var_historical = np.percentile(simulated_returns[-1, :], 1 - confidence_level) * -initial_price
print(f"Value at Risk ({confidence_level:.2f} confidence): {var_historical:.2f}")

# Further analysis (optional)
# ... Explore additional calculations, visualizations, or statistical tests based on your research questions ...

print("Market data analysis completed. Please refer to the results and visualizations above.")
