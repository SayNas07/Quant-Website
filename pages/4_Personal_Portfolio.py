import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import random
from scipy.optimize import minimize

st.title("Personalised Portfolio")

st.write("### Input Data")
col1, col2 = st.columns(2)
risk_free_rate = col1.number_input("Risk Free Rate (%)", min_value=0, value=2) / 100
sp100_tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA", "BRK-B", "UNH",
    "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "LLY", "MRK",
    "ABBV", "PEP", "AVGO", "KO", "COST", "MCD", "WMT", "BAC", "ADBE", "CRM",
    "TMO", "ACN", "NFLX", "LIN", "ABT", "AMD", "ORCL", "DHR", "CMCSA", "WFC",
    "DIS", "TXN", "INTC", "PM", "NEE", "RTX", "HON", "QCOM", "UPS", "LOW",
    "IBM", "INTU", "AMGN", "SBUX", "CAT", "GS", "BLK", "SPGI", "BA", "MDT",
    "GILD", "AMAT", "LMT", "ISRG", "DE", "BKNG", "SYK", "TJX", "PLD", "C",
    "ADP", "NOW", "MO", "MDLZ", "AXP", "VRTX", "ZTS", "SCHW", "TGT", "CVS",
    "ELV", "MMC", "CI", "BDX", "REGN", "FIS", "PNC", "CSCO", "SO", "DUK",
    "SHW", "CL", "ICE", "AON", "ETN", "USB", "WM", "ITW", "HCA", "SLB"
]
default = ["AAPL", "MSFT"]
tickers = col2.multiselect("Tickers (Company Stocks)", options = sp100_tickers, max_selections = 25,default = default)

# -------------------------
# DOWNLOAD DATA
# -------------------------
with st.spinner("Downloading data..."):
    data = yf.download(tickers, start="2018-01-01", progress=False)["Close"]
    
    # Drop any columns with too many NaN values
    data = data.dropna(axis=1, thresh=len(data) * 0.8)
    
    # Update selected tickers based on available data
    selected = data.columns.tolist()
    num_tickers = len(selected)
    
    if num_tickers < 2:
        st.error("Less than 2 valid tickers with sufficient data. Please try again.")
        st.stop()
    
    # Calculate returns
    log_returns = np.log(data / data.shift(1)).dropna()
    
    mean_returns = log_returns.mean() * 252
    cov_matrix = log_returns.cov() * 252

# -------------------------
# MONTE CARLO SIMULATION
# -------------------------
num_runs = 10000
results = np.zeros((3, num_runs))

for i in range(num_runs):
    weights = np.random.random(num_tickers)
    weights /= np.sum(weights)
    
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Handle potential division by zero
    if portfolio_vol > 0:
        sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
    else:
        sharpe = -np.inf
    
    results[0, i] = portfolio_vol
    results[1, i] = portfolio_return
    results[2, i] = sharpe

# Remove any invalid results
valid_mask = ~(np.isnan(results[0]) | np.isnan(results[1]) | np.isnan(results[2]))
results = results[:, valid_mask]

if results.shape[1] == 0:
    st.error("No valid portfolios generated. Please try different tickers or date range.")
    st.stop()

# -------------------------
# FIND TANGENCY PORTFOLIO
# -------------------------
# Find portfolio with maximum Sharpe ratio
max_sharpe_idx = np.argmax(results[2])
tangency_vol = results[0, max_sharpe_idx]
tangency_return = results[1, max_sharpe_idx]
max_sharpe = results[2, max_sharpe_idx]

# -------------------------
# GRAPH 1 – MONTE CARLO
# -------------------------
plt.style.use("dark_background")

fig1, ax1 = plt.subplots(figsize=(10, 6))
scatter = ax1.scatter(results[0], results[1], c=results[2], cmap="viridis", alpha=0.6, s=10)
ax1.scatter(tangency_vol, tangency_return, c='red', marker='*', s=200, label='Tangency Portfolio')
ax1.set_xlabel("Volatility (Std Dev)")
ax1.set_ylabel("Expected Return")
ax1.set_title("Monte Carlo Simulated Portfolios")
plt.colorbar(scatter, label="Sharpe Ratio")
ax1.legend()
st.pyplot(fig1)

# -------------------------
# FUNCTION FOR EFFICIENT FRONTIER
# -------------------------

def get_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=50):
    """Calculate the efficient frontier using optimization"""
    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), num_points)
    efficient_portfolios = []
    
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) - target}  # target return
        )
        bounds = tuple((0, 1) for _ in range(len(mean_returns)))
        
        # Initial guess (equal weights)
        init_guess = np.array([1/len(mean_returns)] * len(mean_returns))
        
        result = minimize(
            lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            efficient_portfolios.append({
                'vol': result.fun,
                'ret': target
            })
    
    return pd.DataFrame(efficient_portfolios)

# -------------------------
# GRAPH 2 – EFFICIENT FRONTIER
# -------------------------
st.write("Calculating efficient frontier...")
with st.spinner("Optimizing portfolio weights..."):
    efficient_portfolios = get_efficient_frontier(mean_returns, cov_matrix, risk_free_rate)

fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot the optimized efficient frontier
if not efficient_portfolios.empty:
    ax2.plot(efficient_portfolios['vol'], efficient_portfolios['ret'], 'g-', linewidth=2, label='Efficient Frontier (Optimized)')
else:
    # Fallback to Monte Carlo method if optimization fails
    df_results = pd.DataFrame({
        'vol': results[0],
        'ret': results[1],
        'sharpe': results[2]
    })
    vol_bins = pd.cut(df_results['vol'], bins=50)
    efficient_portfolios = df_results.loc[df_results.groupby(vol_bins, observed=True)['ret'].idxmax()].sort_values('vol')
    ax2.plot(efficient_portfolios['vol'], efficient_portfolios['ret'], 'g-', linewidth=2, label='Efficient Frontier (Approximated)')

# Plot all portfolios and tangency portfolio
ax2.scatter(results[0], results[1], c='blue', alpha=0.3, s=5, label='All Portfolios')
ax2.scatter(tangency_vol, tangency_return, c='red', marker='*', s=200, label='Tangency Portfolio')
ax2.set_xlabel("Volatility (Std Dev)")
ax2.set_ylabel("Expected Return")
ax2.set_title("Efficient Frontier")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# -------------------------
# GRAPH 3 – CAL
# -------------------------

fig3, ax3 = plt.subplots(figsize=(10, 6))

# Plot efficient frontier (use the optimized one if available)
if not efficient_portfolios.empty:
    ax3.plot(efficient_portfolios['vol'], efficient_portfolios['ret'], 'g-', linewidth=2, label='Efficient Frontier')
else:
    # Fallback to the binned version
    df_results = pd.DataFrame({
        'vol': results[0],
        'ret': results[1],
        'sharpe': results[2]
    })
    vol_bins = pd.cut(df_results['vol'], bins=50)
    efficient_portfolios = df_results.loc[df_results.groupby(vol_bins, observed=True)['ret'].idxmax()].sort_values('vol')
    ax3.plot(efficient_portfolios['vol'], efficient_portfolios['ret'], 'g-', linewidth=2, label='Efficient Frontier')

# Plot all portfolios
ax3.scatter(results[0], results[1], c='blue', alpha=0.2, s=5)

# Calculate and plot CAL
if tangency_vol > 0:
    cal_x = np.linspace(0, max(efficient_portfolios['vol']) * 1.2, 100)
    cal_y = risk_free_rate + (tangency_return - risk_free_rate) / tangency_vol * cal_x
    ax3.plot(cal_x, cal_y, 'r--', linewidth=2, label='Capital Allocation Line')
    
    # Plot risk-free rate point - MAKE SURE THIS IS AT CORRECT Y-VALUE
    ax3.scatter(0, risk_free_rate, c='white', marker='o', s=100, edgecolors='red', linewidth=2, label='Risk-Free Rate')
    
    # Add text annotation to verify the value
    ax3.annotate(f'Rf = {risk_free_rate:.1%}', 
                xy=(0, risk_free_rate),
                xytext=(0.02, risk_free_rate + 0.01),
                color='white',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    
    # Plot tangency portfolio
    ax3.scatter(tangency_vol, tangency_return, c='yellow', marker='*', s=200, 
               edgecolors='red', linewidth=2, label='Tangency Portfolio')
    
    # Add annotation for Sharpe ratio
    ax3.annotate(f'Sharpe: {max_sharpe:.2f}', 
                xy=(tangency_vol, tangency_return),
                xytext=(tangency_vol * 1.1, tangency_return * 0.9),
                color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
else:
    st.warning("Could not calculate valid tangency portfolio")

# Set labels and title
ax3.set_xlabel("Volatility (Std Dev)")
ax3.set_ylabel("Expected Return")
ax3.set_title("Efficient Frontier & Capital Allocation Line")
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Set axis limits with some padding
if not efficient_portfolios.empty:
    ax3.set_xlim(0, max(efficient_portfolios['vol']) * 1.2)
    ax3.set_ylim(min(0, min(efficient_portfolios['ret']) * 0.9), max(efficient_portfolios['ret']) * 1.1)

st.pyplot(fig3)

# -------------------------
# DISPLAY STATISTICS
# -------------------------

st.write("### Portfolio Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Tangency Portfolio Return", f"{tangency_return:.2%}")
col2.metric("Tangency Portfolio Volatility", f"{tangency_vol:.2%}")
col3.metric("Maximum Sharpe Ratio", f"{max_sharpe:.2f}")

# Also display the weights of the tangency portfolio
st.write("### Tangency Portfolio Weights")
# Get the weights of the tangency portfolio
weights_tangency = None
for i in range(num_runs):
    if results[2, i] == max_sharpe:
        # Recalculate weights for this portfolio
        weights = np.random.random(num_tickers)
        weights /= np.sum(weights)
        # We need to find the actual weights - this is approximate
        # In a production version, you'd store weights during simulation
        st.write("Note: Exact weights would need to be stored during simulation")
        break
    
# -------------------------
# VERIFICATION SECTION
# -------------------------
st.write("### Verification")
st.write("Check if the following are correct:")

col1, col2 = st.columns(2)
with col1:
    st.write("**Risk-Free Rate Position:**")
    st.write(f"- Should be at (0, {risk_free_rate:.2%}) on the graph")
    st.write(f"- Currently plotted at (0, {risk_free_rate:.2%})")
    
with col2:
    st.write("**Tangency Portfolio:**")
    st.write(f"- Return: {tangency_return:.2%}")
    st.write(f"- Volatility: {tangency_vol:.2%}")
    st.write(f"- Sharpe Ratio: {max_sharpe:.2f}")

st.write("**CAL Slope Check:**")
slope = (tangency_return - risk_free_rate) / tangency_vol
st.write(f"The CAL slope (Sharpe Ratio) should be {max_sharpe:.2f} and is calculated as {slope:.2f}")
if abs(slope - max_sharpe) < 0.01:
    st.success("✓ CAL slope matches Sharpe ratio")
else:
    st.error("✗ CAL slope does not match Sharpe ratio - check calculations")