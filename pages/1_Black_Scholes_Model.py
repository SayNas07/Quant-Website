import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Black-Scholes Option Pricer",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3D59;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #17A2B8;
        text-align: center;
        margin-bottom: 2rem;
    }
    .call-box {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .call-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .put-box {
        background: linear-gradient(135deg, #dc3545 0%, #e74c3c 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .put-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .price-label {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    .price-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .greek-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #dee2e6;
        margin: 5px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .greek-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .greek-value {
        font-size: 1.2rem;
        font-weight: 500;
        color: #17A2B8;
    }
    .info-text {
        color: #6c757d;
        font-size: 0.9rem;
        text-align: center;
    }
    .greek-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📊 Black-Scholes Option Pricing Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">European Options Pricer with Heatmap Analysis & Greeks</p>', unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("### Created by: Say Nas")
    st.markdown("---")
    
    st.markdown("### ⚙️ Input Parameters")
    st.markdown("---")
    
    # Asset Information
    st.markdown("#### 📈 Asset Information")
    current_asset_price = st.number_input(
        "Current Asset Price ($)",
        min_value=0.01,
        max_value=10000.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )
    
    strike_price = st.number_input(
        "Strike Price ($)",
        min_value=0.01,
        max_value=10000.0,
        value=100.0,
        step=1.0,
        format="%.2f"
    )
    
    # Market Parameters
    st.markdown("#### 📊 Market Parameters")
    col1, col2 = st.columns(2)
    with col1:
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=-5.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            format="%.1f"
        ) / 100
    
    with col2:
        dividend_yield = st.number_input(
            "Dividend Yield (%)",
            min_value=0.0,
            max_value=20.0,
            value=0.0,
            step=0.1,
            format="%.1f"
        ) / 100
    
    # Option Parameters
    st.markdown("#### ⏰ Option Parameters")
    col1, col2 = st.columns(2)
    with col1:
        vol = st.number_input(
            "Volatility (σ)",
            min_value=0.01,
            max_value=2.0,
            value=0.2,
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        ttm = st.number_input(
            "Time to Maturity (Years)",
            min_value=0.01,
            max_value=50.0,
            value=1.0,
            step=0.1,
            format="%.1f"
        )
    
    st.markdown("---")
    
    # Heatmap Parameters
    st.markdown("#### 🔥 Heatmap Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        min_spot = st.number_input(
            "Min Spot Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=max(1.0, current_asset_price * 0.5),
            step=5.0,
            format="%.2f"
        )
    
    with col2:
        max_spot = st.number_input(
            "Max Spot Price ($)",
            min_value=0.01,
            max_value=10000.0,
            value=current_asset_price * 1.5,
            step=5.0,
            format="%.2f"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        min_vol = st.slider(
            "Min Volatility",
            min_value=0.01,
            max_value=1.0,
            value=max(0.05, vol * 0.5),
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        max_vol = st.slider(
            "Max Volatility",
            min_value=0.01,
            max_value=1.0,
            value=min(1.0, vol * 1.5),
            step=0.01,
            format="%.2f"
        )
    
    # Heatmap resolution
    resolution = st.slider(
        "Heatmap Resolution",
        min_value=5,
        max_value=15,
        value=8,
        step=1,
        help="Higher values give more detailed heatmaps but smaller text"
    )
    
    st.markdown("---")
    st.markdown("#### ℹ️ About")
    st.markdown("""
    This model prices European options using the Black-Scholes formula and includes:
    - **Option Prices** (Call & Put)
    - **Heatmaps** (Price sensitivity to Spot & Volatility)
    - **The Greeks** (Risk measures)
    """)

# Black-Scholes function with dividend yield (Merton model)
def black_scholes(S, K, T, r, q, sigma, option_type="call"):
    """
    Calculate Black-Scholes option price with continuous dividend yield
    """
    if T <= 0 or sigma <= 0:
        return 0
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def calculate_greeks(S, K, T, r, q, sigma, option_type="call"):
    """
    Calculate option Greeks
    """
    if T <= 0 or sigma <= 0:
        return {
            'delta': 0,
            'gamma': 0,
            'theta': 0,
            'vega': 0,
            'rho': 0
        }
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == "call":
        delta = np.exp(-q*T) * norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q*T) / (2*np.sqrt(T)) 
                - r * K * np.exp(-r*T) * norm.cdf(d2) 
                + q * S * np.exp(-q*T) * norm.cdf(d1))
    else:
        delta = -np.exp(-q*T) * norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma * np.exp(-q*T) / (2*np.sqrt(T)) 
                + r * K * np.exp(-r*T) * norm.cdf(-d2) 
                - q * S * np.exp(-q*T) * norm.cdf(-d1))
    
    gamma = norm.pdf(d1) * np.exp(-q*T) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-q*T) / 100  # Scaled for 1% change
    
    if option_type == "call":
        rho = K * T * np.exp(-r*T) * norm.cdf(d2) / 100  # Scaled for 1% change
    else:
        rho = -K * T * np.exp(-r*T) * norm.cdf(-d2) / 100  # Scaled for 1% change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Daily theta
        'vega': vega,
        'rho': rho
    }

# Calculate current prices
call_price = black_scholes(current_asset_price, strike_price, ttm, 
                          risk_free_rate, dividend_yield, vol, "call")
put_price = black_scholes(current_asset_price, strike_price, ttm, 
                         risk_free_rate, dividend_yield, vol, "put")

# Calculate Greeks
call_greeks = calculate_greeks(current_asset_price, strike_price, ttm, 
                              risk_free_rate, dividend_yield, vol, "call")
put_greeks = calculate_greeks(current_asset_price, strike_price, ttm, 
                             risk_free_rate, dividend_yield, vol, "put")

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
        <div class="call-box">
            <div class="price-label">CALL OPTION</div>
            <div class="price-value">${call_price:.2f}</div>
            <div style="font-size: 1rem;">Intrinsic Value: ${max(current_asset_price - strike_price, 0):.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="put-box">
            <div class="price-label">PUT OPTION</div>
            <div class="price-value">${put_price:.2f}</div>
            <div style="font-size: 1rem;">Intrinsic Value: ${max(strike_price - current_asset_price, 0):.2f}</div>
        </div>
    """, unsafe_allow_html=True)

# Heatmaps
st.markdown("### 🔥 Option Price Heatmaps")

# Create grid for heatmap
spot_range = np.linspace(min_spot, max_spot, resolution)
vol_range = np.linspace(min_vol, max_vol, resolution)

# Calculate prices for heatmap
call_heatmap_data = np.zeros((len(vol_range), len(spot_range)))
put_heatmap_data = np.zeros((len(vol_range), len(spot_range)))

for i, v in enumerate(vol_range):
    for j, s in enumerate(spot_range):
        call_heatmap_data[i, j] = black_scholes(s, strike_price, ttm, 
                                               risk_free_rate, dividend_yield, v, "call")
        put_heatmap_data[i, j] = black_scholes(s, strike_price, ttm, 
                                              risk_free_rate, dividend_yield, v, "put")

col1, col2 = st.columns(2)

with col1:
    # Call Option Heatmap with Matplotlib
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im_call = ax_call.imshow(call_heatmap_data, cmap='YlGn', aspect='auto', 
                            origin='lower', vmin=call_heatmap_data.min(), vmax=call_heatmap_data.max())
    
    # Add colorbar
    plt.colorbar(im_call, ax=ax_call, label='Call Price ($)')
    
    # Set ticks
    ax_call.set_xticks(np.arange(len(spot_range)))
    ax_call.set_yticks(np.arange(len(vol_range)))
    ax_call.set_xticklabels([f'${s:.0f}' for s in spot_range])
    ax_call.set_yticklabels([f'{v:.2f}' for v in vol_range])
    
    # Rotate x-axis labels
    plt.setp(ax_call.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add price labels to each cell
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            text = ax_call.text(j, i, f'${call_heatmap_data[i, j]:.1f}',
                              ha='center', va='center', color='black', fontsize=8)

    ax_call.set_xlabel('Spot Price ($)')
    ax_call.set_ylabel('Volatility')
    ax_call.set_title('CALL Option Prices', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_call)

with col2:
    # Put Option Heatmap with Matplotlib
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    im_put = ax_put.imshow(put_heatmap_data, cmap='YlOrRd', aspect='auto', 
                          origin='lower', vmin=put_heatmap_data.min(), vmax=put_heatmap_data.max())
    
    # Add colorbar
    plt.colorbar(im_put, ax=ax_put, label='Put Price ($)')
    
    # Set ticks
    ax_put.set_xticks(np.arange(len(spot_range)))
    ax_put.set_yticks(np.arange(len(vol_range)))
    ax_put.set_xticklabels([f'${s:.0f}' for s in spot_range])
    ax_put.set_yticklabels([f'{v:.2f}' for v in vol_range])
    
    # Rotate x-axis labels
    plt.setp(ax_put.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add price labels to each cell
    for i in range(len(vol_range)):
        for j in range(len(spot_range)):
            text = ax_put.text(j, i, f'${put_heatmap_data[i, j]:.1f}',
                             ha='center', va='center', color='black', fontsize=8)
    
    ax_put.set_xlabel('Spot Price ($)')
    ax_put.set_ylabel('Volatility')
    ax_put.set_title('PUT Option Prices', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig_put)

# Greeks Section at the bottom
st.markdown("---")
st.markdown("### 📐 The Greeks - Risk Measures")

# Create tabs for Call and Put Greeks
tab1, tab2 = st.tabs(["📈 Call Option Greeks", "📉 Put Option Greeks"])

with tab1:
    st.markdown("#### Call Option Greeks")
    cols = st.columns(5)
    
    greeks_data = [
        ("Delta (Δ)", call_greeks['delta'], "Rate of change of option price with respect to underlying asset price", "#17A2B8"),
        ("Gamma (Γ)", call_greeks['gamma'], "Rate of change of Delta with respect to underlying asset price", "#28A745"),
        ("Theta (Θ)", call_greeks['theta'], "Time decay (daily)", "#DC3545"),
        ("Vega (ν)", call_greeks['vega'], "Sensitivity to volatility (per 1% change)", "#FFC107"),
        ("Rho (ρ)", call_greeks['rho'], "Sensitivity to interest rate (per 1% change)", "#6F42C1")
    ]
    
    for i, (name, value, desc, color) in enumerate(greeks_data):
        with cols[i]:
            st.markdown(f"""
                <div class="greek-box">
                    <div class="greek-title" style="color: {color};">{name}</div>
                    <div class="greek-value">{value:.4f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 5px;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("#### Put Option Greeks")
    cols = st.columns(5)
    
    greeks_data = [
        ("Delta (Δ)", put_greeks['delta'], "Rate of change of option price with respect to underlying asset price", "#17A2B8"),
        ("Gamma (Γ)", put_greeks['gamma'], "Rate of change of Delta with respect to underlying asset price", "#28A745"),
        ("Theta (Θ)", put_greeks['theta'], "Time decay (daily)", "#DC3545"),
        ("Vega (ν)", put_greeks['vega'], "Sensitivity to volatility (per 1% change)", "#FFC107"),
        ("Rho (ρ)", put_greeks['rho'], "Sensitivity to interest rate (per 1% change)", "#6F42C1")
    ]
    
    for i, (name, value, desc, color) in enumerate(greeks_data):
        with cols[i]:
            st.markdown(f"""
                <div class="greek-box">
                    <div class="greek-title" style="color: {color};">{name}</div>
                    <div class="greek-value">{value:.4f}</div>
                    <div style="font-size: 0.8rem; color: #6c757d; margin-top: 5px;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

# Greeks Explanation
with st.expander("📚 Understanding the Greeks"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Delta (Δ)** 
        - Call: 0 to 1 (positive)
        - Put: -1 to 0 (negative)
        - Measures directional risk
        """)
        
        st.markdown("""
        **Gamma (Γ)** 
        - Same for calls and puts
        - Highest for at-the-money options
        - Measures convexity/acceleration
        """)
    
    with col2:
        st.markdown("""
        **Theta (Θ)** 
        - Usually negative for long positions
        - Measures time decay
        - Accelerates as expiration approaches
        """)
        
        st.markdown("""
        **Vega (ν)** 
        - Same for calls and puts
        - Higher for longer-dated options
        - Measures volatility risk
        """)
    
    with col3:
        st.markdown("""
        **Rho (ρ)** 
        - Call: positive
        - Put: negative
        - Higher for longer-dated options
        - Measures interest rate risk
        """)

# Additional Information
with st.expander("📊 How to use the Heatmaps"):
    st.markdown("""
    - **X-axis**: Spot price range (configured in sidebar)
    - **Y-axis**: Volatility range (configured in sidebar)
    - **Colors**: Darker colors indicate higher option prices
    - **Numbers**: Each cell shows the exact option price
    - **Red/Blue X**: Your current position based on input parameters
    
    The heatmaps help visualize how option prices change across different spot prices and volatility levels.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<p class='info-text'>⚠️ This model is for educational purposes only. "
    "Real-world options trading involves significant risk.</p>",
    unsafe_allow_html=True
)