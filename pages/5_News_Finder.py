import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

# Popular stock tickers with company names for better UX
POPULAR_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "NVDA": "NVIDIA",
    "META": "Meta (Facebook)",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "PG": "Procter & Gamble",
    "UNH": "UnitedHealth",
    "HD": "Home Depot",
    "DIS": "Disney",
    "PYPL": "PayPal",
    "NFLX": "Netflix",
    "INTC": "Intel",
    "AMD": "AMD",
    "CRM": "Salesforce"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_stock_news(ticker, company_name=None, days_back=7):
    """
    Fetch top 10 news articles for a given stock ticker
    
    Args:
        ticker: Stock symbol
        company_name: Optional company name for better search
        days_back: How many days back to search
    
    Returns:
        List of articles or None if error
    """
    
    API_KEY = st.secrets.get("NEWS_API_KEY")
    
    # Use company name if available for better results
    query = company_name if company_name else ticker
    
    # Date range (last 7 days for relevant news)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # NewsAPI endpoint with parameters
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"({query} OR {ticker}) AND (stock OR market OR earnings OR shares)",
        "from": from_date,
        "sortBy": "relevancy",  # or "popularity" or "publishedAt"
        "pageSize": 10,
        "language": "en",
        "apiKey": API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            return data["articles"]
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return None

def format_published_date(date_str):
    """Format ISO date to readable format"""
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        now = datetime.now(date.tzinfo)
        
        diff = now - date
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    except:
        return date_str

def get_source_logo(source_name):
    """Return emoji for common news sources (can be replaced with actual logos)"""
    logos = {
        "reuters": "📰",
        "bloomberg": "📊",
        "cnbc": "📺",
        "yahoo": "💹",
        "wsj": "📈",
        "ft": "🏛️",
        "forbes": "💰",
        "business insider": "💼",
        "seeking alpha": "α",
        "motley fool": "🃏"
    }
    
    source_lower = source_name.lower()
    for key, logo in logos.items():
        if key in source_lower:
            return logo
    return "📰"

# ============================================================================
# MAIN UI
# ============================================================================

def render_news_page():
    """Main function to render the news page"""
    
    # Page header with styling
    st.markdown("""
    <style>
    .news-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }
    .news-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-color: #b0b0b0;
    }
    .news-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .news-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .news-description {
        color: #333;
        margin-bottom: 0.5rem;
    }
    .source-badge {
        display: inline-block;
        background-color: #f0f0f0;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📰 Stock Market News")
    st.markdown("Top 10 most relevant news articles for your selected stock")
    
    # Stock selector with company names
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_ticker = st.selectbox(
            "Select a stock",
            options=list(POPULAR_STOCKS.keys()),
            format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}"
        )
    
    with col2:
        days_back = st.selectbox(
            "News from",
            options=[1, 3, 7, 14, 30],
            format_func=lambda x: f"Last {x} days",
            index=2  # Default to 7 days
        )
    
    # Fetch button
    if st.button("🔍 Get Latest News", type="primary", use_container_width=True):
        with st.spinner(f"Fetching latest news for {selected_ticker}..."):
            articles = fetch_stock_news(
                selected_ticker, 
                POPULAR_STOCKS[selected_ticker],
                days_back
            )
            
            if articles:
                st.success(f"Found {len(articles)} articles")
                
                # Display articles
                for i, article in enumerate(articles, 1):
                    source_name = article["source"]["name"]
                    source_logo = get_source_logo(source_name)
                    
                    # Create a card-like container
                    with st.container():
                        st.markdown(f'<div class="news-card">', unsafe_allow_html=True)
                        
                        # Layout with columns for image and content
                        col_img, col_content = st.columns([1, 4])
                        
                        with col_img:
                            if article.get("urlToImage"):
                                st.image(
                                    article["urlToImage"], 
                                    use_column_width=True,
                                    caption=f"Source: {source_name}"
                                )
                            else:
                                # Placeholder for articles without images
                                st.markdown(
                                    f'<div style="background-color:#f0f0f0; height:100px; display:flex; align-items:center; justify-content:center; border-radius:5px;">{source_logo}</div>',
                                    unsafe_allow_html=True
                                )
                        
                        with col_content:
                            # Title with link
                            st.markdown(
                                f'<div class="news-title"><a href="{article["url"]}" target="_blank">{article["title"]}</a></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Meta information
                            published = format_published_date(article["publishedAt"])
                            st.markdown(
                                f'<div class="news-meta">'
                                f'<span class="source-badge">{source_logo} {source_name}</span> '
                                f'⏱️ {published}'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Description/preview
                            if article.get("description"):
                                st.markdown(
                                    f'<div class="news-description">{article["description"][:200]}...</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Read more button
                            st.markdown(
                                f'<a href="{article["url"]}" target="_blank">'
                                f'<button style="background-color:#4CAF50; color:white; border:none; padding:5px 15px; border-radius:3px; cursor:pointer;">📖 Read Full Article</button>'
                                f'</a>',
                                unsafe_allow_html=True
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add separator between articles (except last)
                        if i < len(articles):
                            st.markdown("---")
                            
            else:
                st.warning(f"No recent news found for {selected_ticker}. Try a different stock or expand the date range.")

# ============================================================================
# FALLBACK: Demo data if no API key
# ============================================================================

def render_demo_news():
    """Fallback function with sample data for demonstration"""
    
    st.info("⚠️ Using demo data. Add NEWS_API_KEY to secrets for live news.")
    
    demo_articles = [
        {
            "title": f"Apple Reports Strong Q4 Earnings, Beats Estimates",
            "description": "Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales and services revenue.",
            "source": {"name": "Bloomberg"},
            "publishedAt": datetime.now().isoformat(),
            "url": "https://www.bloomberg.com",
            "urlToImage": "https://via.placeholder.com/300x200"
        },
        # Add more demo articles...
    ]
    
    # Display demo articles (similar styling as above)
    # ...

# ============================================================================
# PAGE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Check if API key is configured
    if "NEWS_API_KEY" in st.secrets:
        render_news_page()
    else:
        st.warning("⚠️ News API key not found. Using demo data.")
        st.info("To enable live news, add NEWS_API_KEY to your .streamlit/secrets.toml file")
        render_demo_news()