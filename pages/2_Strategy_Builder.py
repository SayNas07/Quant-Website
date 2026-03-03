import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(layout="wide")
st.title("Quant Options Strategy Builder - Fixed Greeks")

# ---------------------------
# Black-Scholes Greeks Functions
# ---------------------------

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks using Black-Scholes
    S: Current stock price
    K: Strike price
    T: Time to expiry (in years)
    r: Risk-free rate
    sigma: Volatility
    """
    if T <= 0 or sigma <= 0:
        return 0, 0
    
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    else:  # put
        delta = norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return delta, gamma

def long_call_payoff(S, K, premium):
    return np.maximum(S - K, 0) - premium

def long_put_payoff(S, K, premium):
    return np.maximum(K - S, 0) - premium

# ---------------------------
# Sidebar Inputs
# ---------------------------

st.sidebar.header("Strategy Parameters")
strategy = st.sidebar.selectbox(
    "Select Strategy",
    ["Long Call", "Long Put", "Straddle", "Strangle", "Bull Call Spread"]
)

# Market parameters
st.sidebar.header("Market Parameters")
current_price = st.sidebar.number_input("Current Stock Price", value=100.0, step=1.0)
volatility = st.sidebar.slider("Implied Volatility (%)", 10, 100, 30) / 100
days_to_expiry = st.sidebar.number_input("Days to Expiry", 1, 365, 30)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0) / 100

T = days_to_expiry / 365  # Time in years

# Price range for analysis
price_range_pct = st.sidebar.slider("Price Range (% of current)", 50, 200, (50, 150))
S_range = np.linspace(current_price * price_range_pct[0] / 100,
                      current_price * price_range_pct[1] / 100,
                      500)

# ---------------------------
# Strategy Inputs
# ---------------------------

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Strategy Configuration")
    
    if strategy == "Long Call":
        K = st.number_input("Strike Price", value=current_price, step=1.0)
        premium = st.number_input("Premium Paid", value=5.0, min_value=0.01)
        
        # Calculate payoff
        payoff = long_call_payoff(S_range, K, premium)
        
        # Calculate Greeks for each price point
        deltas = []
        gammas = []
        for S in S_range:
            delta, gamma = calculate_greeks(S, K, T, risk_free_rate, volatility, 'call')
            deltas.append(delta)
            gammas.append(gamma)
            
    elif strategy == "Long Put":
        K = st.number_input("Strike Price", value=current_price, step=1.0)
        premium = st.number_input("Premium Paid", value=5.0, min_value=0.01)
        
        payoff = long_put_payoff(S_range, K, premium)
        
        deltas = []
        gammas = []
        for S in S_range:
            delta, gamma = calculate_greeks(S, K, T, risk_free_rate, volatility, 'put')
            deltas.append(delta)
            gammas.append(gamma)
    
    elif strategy == "Straddle":
        K = st.number_input("Strike Price", value=current_price, step=1.0)
        col_a, col_b = st.columns(2)
        with col_a:
            call_p = st.number_input("Call Premium", value=5.0)
        with col_b:
            put_p = st.number_input("Put Premium", value=5.0)
        
        payoff = long_call_payoff(S_range, K, call_p) + long_put_payoff(S_range, K, put_p)
        
        # Greeks for straddle (sum of call and put Greeks)
        deltas = []
        gammas = []
        for S in S_range:
            delta_call, gamma_call = calculate_greeks(S, K, T, risk_free_rate, volatility, 'call')
            delta_put, gamma_put = calculate_greeks(S, K, T, risk_free_rate, volatility, 'put')
            deltas.append(delta_call + delta_put)
            gammas.append(gamma_call + gamma_put)

# ---------------------------
# Display Table of Greeks
# ---------------------------

with col2:
    st.header("Greeks Analysis")
    
    # Create sample data at key price points
    price_points = [current_price * 0.8, current_price * 0.9, current_price,
                   current_price * 1.1, current_price * 1.2]
    
    greeks_data = []
    for price in price_points:
        idx = np.argmin(np.abs(S_range - price))
        greeks_data.append({
            'Stock Price': f'${price:.2f}',
            'Delta': f'{deltas[idx]:.3f}',
            'Gamma': f'{gammas[idx]:.3f}',
            'Position': 'ATM' if abs(price - K) < 1 else 'ITM' if price > K else 'OTM'
        })
    
    st.table(pd.DataFrame(greeks_data))
    
    # Risk metrics
    st.header("Risk Metrics")
    max_profit = np.max(payoff)
    max_loss = np.min(payoff)
    
    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
    col_metrics1.metric("Max Profit", f"${max_profit:.2f}")
    col_metrics2.metric("Max Loss", f"${max_loss:.2f}")
    
    # Current Greeks
    current_idx = np.argmin(np.abs(S_range - current_price))
    col_metrics3.metric("Current Delta", f"{deltas[current_idx]:.3f}")

# ---------------------------
# Plotting
# ---------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'{strategy} Strategy Analysis', fontsize=16, fontweight='bold')

# Plot 1: Payoff Diagram
ax1 = axes[0, 0]
ax1.plot(S_range, payoff, 'b-', linewidth=2, label='Payoff')
ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax1.axvline(x=current_price, color='orange', linestyle='--', label=f'Current: ${current_price}')
if 'K' in locals():
    ax1.axvline(x=K, color='green', linestyle=':', label=f'Strike: ${K}')
ax1.fill_between(S_range, 0, payoff, where=(payoff > 0), color='green', alpha=0.3)
ax1.fill_between(S_range, 0, payoff, where=(payoff < 0), color='red', alpha=0.3)
ax1.set_xlabel('Stock Price ($)')
ax1.set_ylabel('Profit/Loss ($)')
ax1.set_title('Payoff Diagram')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Delta
ax2 = axes[0, 1]
ax2.plot(S_range, deltas, 'r-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax2.axvline(x=K, color='green', linestyle=':', alpha=0.5)
ax2.fill_between(S_range, 0, deltas, where=(np.array(S_range) < K), color='blue', alpha=0.2, label='OTM')
ax2.fill_between(S_range, 0, deltas, where=(np.array(S_range) > K), color='red', alpha=0.2, label='ITM')
ax2.set_xlabel('Stock Price ($)')
ax2.set_ylabel('Delta')
ax2.set_title('Delta (Price Sensitivity)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gamma
ax3 = axes[1, 0]
ax3.plot(S_range, gammas, 'g-', linewidth=2)
ax3.axvline(x=K, color='green', linestyle=':', alpha=0.5)
ax3.fill_between(S_range, 0, gammas, alpha=0.3, color='purple')
ax3.set_xlabel('Stock Price ($)')
ax3.set_ylabel('Gamma')
ax3.set_title('Gamma (Delta Sensitivity)')
ax3.grid(True, alpha=0.3)

# Plot 4: Greeks Heat Map
ax4 = axes[1, 1]
ax4.axis('off')
info_text = f"""
Greeks Explanation:

Δ Delta: Rate of change of option price with respect to stock price
• Call: 0 → 1 (ITM)
• Put: -1 → 0 (ITM)

Γ Gamma: Rate of change of delta
• Highest ATM
• Positive for long options

Current Values at ${current_price:.2f}:
Δ Delta: {deltas[current_idx]:.3f}
Γ Gamma: {gammas[current_idx]:.3f}

Time to Expiry: {days_to_expiry} days
IV: {volatility*100:.1f}%
"""
ax4.text(0.1, 0.5, info_text, fontsize=12, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
st.pyplot(fig)

# ---------------------------
# Educational Notes
# ---------------------------

with st.expander("📚 Understanding Greeks"):
    st.markdown("""
    ### Delta (Δ)
    - Measures how much the option price changes for a $1 change in stock price
    - Call options: 0 to 1 (positive)
    - Put options: -1 to 0 (negative)
    - ATM options typically have |delta| ≈ 0.5
    
    ### Gamma (Γ)
    - Measures how much delta changes for a $1 change in stock price
    - Highest for ATM options
    - Positive for long options (both calls and puts)
    - Declines as options go deep ITM or OTM
    
    ### Relationship
    - Gamma is the derivative of Delta
    - When Gamma is high, Delta is more sensitive to stock price movements
    - Options near expiry have higher Gamma ATM
    """)