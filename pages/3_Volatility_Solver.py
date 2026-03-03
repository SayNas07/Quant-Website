import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm
import time

# Page configuration
st.set_page_config(
    page_title="Implied Volatility Calculator",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📈 Implied Volatility Solver</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Professional-grade option implied volatility calculator using Newton-Raphson method</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("### ⚙️ Input Parameters")
    
    option_type = st.selectbox(
        "Option Type",
        ["call", "put"],
        help="Select call or put option"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        S = st.number_input(
            "Spot Price (S)",
            value=100.0,
            min_value=0.01,
            step=1.0,
            help="Current underlying asset price"
        )
        K = st.number_input(
            "Strike Price (K)",
            value=100.0,
            min_value=0.01,
            step=1.0,
            help="Option strike price"
        )
    
    with col2:
        T = st.number_input(
            "Time to Expiry (Years)",
            value=1.0,
            min_value=0.01,
            max_value=10.0,
            step=0.1,
            help="Time until option expiration in years"
        )
        r = st.number_input(
            "Risk-Free Rate (%)",
            value=5.0,
            min_value=0.0,
            max_value=20.0,
            step=0.25,
            help="Annual risk-free interest rate"
        ) / 100
    
    market_price = st.number_input(
        "Market Option Price",
        value=10.0,
        min_value=0.01,
        step=0.5,
        help="Current market price of the option"
    )
    
    st.markdown("---")
    st.markdown("### 🔧 Solver Settings")
    
    method = st.selectbox(
        "Numerical Method",
        ["Newton-Raphson", "Bisection (Backup)"],
        help="Choose the numerical method for IV calculation"
    )
    
    show_details = st.checkbox("Show detailed convergence", value=True)
    show_plot = st.checkbox("Show price-volatility curve", value=True)

# Black-Scholes functions
@st.cache_data
def black_scholes(S, K, T, r, sigma, option_type):
    """Calculate Black-Scholes option price"""
    if sigma <= 0 or T <= 0:
        return max(0, (S - K) if option_type == "call" else (K - S))
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    return max(price, 0)

def vega(S, K, T, r, sigma):
    """Calculate option vega"""
    if sigma <= 0 or T <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def newton_raphson_iv(S, K, T, r, market_price, option_type, max_iter=100, tol=1e-6):
    """Newton-Raphson method for implied volatility"""
    sigma = 0.3  # initial guess (30%)
    convergence = []
    
    for i in range(max_iter):
        price = black_scholes(S, K, T, r, sigma, option_type)
        v = vega(S, K, T, r, sigma)
        
        price_diff = price - market_price
        
        # Store convergence data
        convergence.append({
            'iteration': i,
            'sigma': sigma,
            'price': price,
            'error': abs(price_diff)
        })
        
        if abs(price_diff) < tol:
            return sigma, convergence
        
        # Avoid division by zero
        if abs(v) < 1e-8:
            sigma = 0.3  # reset guess
            continue
            
        sigma = sigma - price_diff / v
        
        # Bounds checking
        if sigma <= 1e-4:
            sigma = 1e-4
        elif sigma >= 5.0:
            sigma = 5.0
    
    return sigma, convergence

def bisection_iv(S, K, T, r, market_price, option_type, max_iter=100, tol=1e-6):
    """Bisection method for implied volatility (backup)"""
    low_sigma, high_sigma = 0.001, 5.0
    convergence = []
    
    for i in range(max_iter):
        mid_sigma = (low_sigma + high_sigma) / 2
        price = black_scholes(S, K, T, r, mid_sigma, option_type)
        
        convergence.append({
            'iteration': i,
            'sigma': mid_sigma,
            'price': price,
            'error': abs(price - market_price)
        })
        
        if abs(price - market_price) < tol:
            return mid_sigma, convergence
        
        if price < market_price:
            low_sigma = mid_sigma
        else:
            high_sigma = mid_sigma
    
    return mid_sigma, convergence

def calculate_price_range(S, K, T, r, option_type):
    """Calculate price bounds for volatility scan"""
    # Price at zero volatility
    price_min = black_scholes(S, K, T, r, 0.001, option_type)
    # Price at high volatility
    price_max = black_scholes(S, K, T, r, 2.0, option_type)
    return price_min, price_max

# Main content
col1, col2, col3 = st.columns(3)

# Check if market price is within theoretical bounds
price_min, price_max = calculate_price_range(S, K, T, r, option_type)

if market_price < price_min or market_price > price_max:
    st.warning(f"⚠️ Market price outside theoretical bounds. Valid range: [{price_min:.2f}, {price_max:.2f}]")

# Calculate implied volatility
with st.spinner('Calculating implied volatility...'):
    start_time = time.time()
    
    if method == "Newton-Raphson":
        iv, convergence = newton_raphson_iv(S, K, T, r, market_price, option_type)
    else:
        iv, convergence = bisection_iv(S, K, T, r, market_price, option_type)
    
    calc_time = time.time() - start_time

# Calculate model price
model_price = black_scholes(S, K, T, r, iv, option_type)

# Display results
st.markdown("### 📊 Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Implied Volatility",
        f"{iv*100:.2f}%",
        delta=None
    )

with col2:
    st.metric(
        "Model Price",
        f"${model_price:.4f}",
        delta=f"${model_price - market_price:.4f}"
    )

with col3:
    st.metric(
        "Price Error",
        f"{abs(model_price - market_price):.2e}",
        delta=None
    )

with col4:
    st.metric(
        "Calculation Time",
        f"{calc_time*1000:.1f} ms",
        delta=None
    )

# Verification
st.markdown("### ✅ Verification")
st.write(f"**Input Market Price:** ${market_price:.4f}")
st.write(f"**Black-Scholes Price (with IV):** ${model_price:.4f}")
st.write(f"**Difference:** ${model_price - market_price:.4f}")

if abs(model_price - market_price) < 1e-4:
    st.success("✓ Solution verified: BS price matches market price")
else:
    st.error("✗ Warning: Solution may not be accurate")

# Detailed convergence
if show_details and convergence:
    st.markdown("### 📈 Convergence Details")
    
    # Create convergence dataframe
    df_conv = pd.DataFrame(convergence)
    
    # Display iterations
    st.dataframe(
        df_conv.style.format({
            'sigma': '{:.4f}',
            'price': '${:.4f}',
            'error': '{:.2e}'
        }),
        use_container_width=True
    )
    
    # Plot convergence
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_conv['iteration'],
        y=df_conv['sigma'] * 100,
        mode='lines+markers',
        name='Implied Volatility',
        line=dict(color='#1E88E5', width=2)
    ))
    
    fig.update_layout(
        title='Implied Volatility Convergence',
        xaxis_title='Iteration',
        yaxis_title='Implied Volatility (%)',
        hovermode='x',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Price vs Volatility curve
if show_plot:
    st.markdown("### 📉 Price-Volatility Relationship")
    
    # Generate volatility range
    vol_range = np.linspace(0.01, 2.0, 100)
    prices = [black_scholes(S, K, T, r, vol, option_type) for vol in vol_range]
    
    fig = go.Figure()
    
    # Price curve
    fig.add_trace(go.Scatter(
        x=vol_range * 100,
        y=prices,
        mode='lines',
        name='Option Price',
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Market price line
    fig.add_hline(
        y=market_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Market Price: ${market_price:.2f}"
    )
    
    # IV point
    fig.add_trace(go.Scatter(
        x=[iv * 100],
        y=[model_price],
        mode='markers',
        name='Implied Vol',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title='Option Price vs Volatility',
        xaxis_title='Volatility (%)',
        yaxis_title='Option Price ($)',
        hovermode='x',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Greeks calculation
st.markdown("### 📐 Option Greeks")
greek_cols = st.columns(4)

# Calculate Greeks at IV
d1 = (np.log(S/K) + (r + 0.5*iv**2)*T) / (iv*np.sqrt(T)) if iv > 0 and T > 0 else 0
d2 = d1 - iv*np.sqrt(T) if iv > 0 and T > 0 else 0

with greek_cols[0]:
    # Delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    st.metric("Delta", f"{delta:.4f}")

with greek_cols[1]:
    # Gamma
    gamma = norm.pdf(d1) / (S * iv * np.sqrt(T)) if iv > 0 and T > 0 else 0
    st.metric("Gamma", f"{gamma:.4f}")

with greek_cols[2]:
    # Theta
    theta_term1 = - (S * norm.pdf(d1) * iv) / (2 * np.sqrt(T))
    if option_type == "call":
        theta_term2 = - r * K * np.exp(-r*T) * norm.cdf(d2)
        theta = theta_term1 + theta_term2
    else:
        theta_term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)
        theta = theta_term1 + theta_term2
    st.metric("Theta", f"{theta:.4f}")

with greek_cols[3]:
    # Vega (per 1% change)
    vega_value = vega(S, K, T, r, iv) / 100
    st.metric("Vega (1%)", f"${vega_value:.4f}")

# Educational section
with st.expander("ℹ️ What is Implied Volatility?"):
    st.markdown("""
    **Implied volatility (IV)** is the market's forecast of a likely movement in a security's price.
    
    Key points:
    - It is derived from an option's price and shows what the market "implies" about the stock's future volatility
    - Unlike historical volatility, IV looks forward
    - Traders often quote option prices in terms of implied volatility
    - Higher IV means higher option premium (more uncertainty)
    
    **In this calculator:**
    1. You input the market price of an option
    2. The solver finds the volatility that makes the Black-Scholes price match the market price
    3. The result is the implied volatility that traders are using
    """)

# Risk warning
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for educational purposes. Always verify calculations and consult with a qualified financial professional before making investment decisions.")