import streamlit as st
import requests
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
from collections import Counter

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
    
    API_KEY = st.secrets.get("NEWS_API_KEY", "YOUR_API_KEY_HERE")
    
    # Use company name if available for better results
    query = company_name if company_name else ticker
    
    # Date range (last X days for relevant news)
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    
    # NewsAPI endpoint with parameters
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"({query} OR {ticker}) AND (stock OR market OR earnings OR shares OR price)",
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

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob
    
    Returns:
        tuple: (sentiment_score, sentiment_category)
    """
    if not text:
        return 0, "Neutral"
    
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    # Categorize sentiment
    if sentiment_score > 0.1:
        category = "Positive"
    elif sentiment_score < -0.1:
        category = "Negative"
    else:
        category = "Neutral"
    
    return sentiment_score, category

def get_sentiment_emoji(category):
    """Return emoji for sentiment category"""
    emojis = {
        "Positive": "🟢",
        "Neutral": "🟡",
        "Negative": "🔴"
    }
    return emojis.get(category, "⚪")

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
    """Return emoji for common news sources"""
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
        "motley fool": "🃏",
        "cnn": "📡",
        "bbc": "🌍",
        "fox": "🦊",
        "nyt": "📰",
        "guardian": "📰"
    }
    
    source_lower = source_name.lower()
    for key, logo in logos.items():
        if key in source_lower:
            return logo
    return "📰"

def create_sentiment_chart(articles):
    """Create a sentiment distribution chart"""
    sentiments = []
    for article in articles:
        # Combine title and description for better sentiment analysis
        text = article.get("title", "") + " " + (article.get("description", "") or "")
        score, _ = analyze_sentiment(text)
        sentiments.append(score)
    
    # Create histogram
    fig = go.Figure(data=[go.Histogram(x=sentiments, nbinsx=20, marker_color='#1E88E5')])
    fig.update_layout(
        title="Sentiment Distribution of News Headlines",
        xaxis_title="Sentiment Score (-1 to 1)",
        yaxis_title="Number of Articles",
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig

# ============================================================================
# MAIN UI
# ============================================================================

def render_news_page():
    """Main function to render the news page"""
    
    # Page header with styling
    st.markdown("""
    <style>
    .news-card {
        padding: 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1.2rem;
        transition: all 0.3s;
        background: darkkgrey;
    }
    .news-card:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        border-color: #1E88E5;
        transform: translateY(-2px);
    }
    .news-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1E88E5;
    }
    .news-title a {
        color: #1E88E5;
        text-decoration: none;
    }
    .news-title a:hover {
        text-decoration: underline;
    }
    .news-meta {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }
    .news-description {
        color: darkgreen;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    .source-badge {
        display: inline-block;
        background-color: #f0f0f0;
        padding: 0.2rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
    .sentiment-positive {
        background-color: #e6f7e6;
        color: #2e7d32;
    }
    .sentiment-negative {
        background-color: #ffebee;
        color: #c62828;
    }
    .sentiment-neutral {
        background-color: #f5f5f5;
        color: #616161;
    }
    .read-more-btn {
        background-color: #1E88E5;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        cursor: pointer;
        font-size: 0.9rem;
        transition: background-color 0.2s;
    }
    .read-more-btn:hover {
        background-color: #1565C0;
    }
    .sentiment-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📰 Stock Market News with Sentiment Analysis")
    st.markdown("Top 10 most relevant news articles with AI-powered sentiment analysis")
    
    # Stock selector with company names
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_ticker = st.selectbox(
            "Select a stock",
            options=list(POPULAR_STOCKS.keys()),
            format_func=lambda x: f"{x} - {POPULAR_STOCKS[x]}",
            key="ticker_selector"
        )
    
    with col2:
        days_back = st.selectbox(
            "Time range",
            options=[1, 3, 7, 14, 30],
            format_func=lambda x: f"Last {x} days",
            index=2,  # Default to 7 days
            key="days_selector"
        )
    
    with col3:
        st.write("")  # Spacing
        st.write("")
        fetch_button = st.button("🔍 Get News", type="primary", use_container_width=True)
    
    # Fetch news when button is clicked
    if fetch_button:
        with st.spinner(f"Fetching and analyzing news for {selected_ticker}..."):
            articles = fetch_stock_news(
                selected_ticker, 
                POPULAR_STOCKS[selected_ticker],
                days_back
            )
            
            if articles:
                # Analyze sentiment for all articles
                for article in articles:
                    text = article.get("title", "") + " " + (article.get("description", "") or "")
                    score, category = analyze_sentiment(text)
                    article["sentiment_score"] = score
                    article["sentiment_category"] = category
                
                # Calculate sentiment summary
                sentiment_counts = Counter([a["sentiment_category"] for a in articles])
                total = len(articles)
                avg_sentiment = sum([a["sentiment_score"] for a in articles]) / total
                
                # Display sentiment summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Articles", total)
                with col2:
                    st.metric("🟢 Positive", sentiment_counts.get("Positive", 0))
                with col3:
                    st.metric("🟡 Neutral", sentiment_counts.get("Neutral", 0))
                with col4:
                    st.metric("🔴 Negative", sentiment_counts.get("Negative", 0))
                
                # Overall sentiment indicator
                if avg_sentiment > 0.1:
                    st.success(f"📈 Overall Sentiment: Positive ({avg_sentiment:.2f})")
                elif avg_sentiment < -0.1:
                    st.error(f"📉 Overall Sentiment: Negative ({avg_sentiment:.2f})")
                else:
                    st.info(f"⚖️ Overall Sentiment: Neutral ({avg_sentiment:.2f})")
                
                # Show sentiment distribution chart
                with st.expander("📊 View Sentiment Distribution"):
                    fig = create_sentiment_chart(articles)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Display articles
                for i, article in enumerate(articles, 1):
                    source_name = article["source"]["name"]
                    source_logo = get_source_logo(source_name)
                    sentiment_score = article["sentiment_score"]
                    sentiment_category = article["sentiment_category"]
                    sentiment_emoji = get_sentiment_emoji(sentiment_category)
                    
                    # Determine sentiment class for styling
                    sentiment_class = f"sentiment-{sentiment_category.lower()}"
                    
                    # Create a card-like container
                    with st.container():
                        st.markdown(f'<div class="news-card">', unsafe_allow_html=True)
                        
                        # Layout with columns for image and content
                        col_img, col_content = st.columns([1, 4])
                        
                        with col_img:
                            if article.get("urlToImage"):
                                # FIXED: Use width parameter instead of deprecated use_column_width
                                st.image(
                                    article["urlToImage"], 
                                    width=200,  # Fixed width instead of use_column_width
                                    caption=f"Source: {source_name}"
                                )
                            else:
                                # Placeholder for articles without images
                                st.markdown(
                                    f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); height:120px; display:flex; align-items:center; justify-content:center; border-radius:8px; color:white; font-size:2rem;">{source_logo}</div>',
                                    unsafe_allow_html=True
                                )
                        
                        with col_content:
                            # Title with link
                            st.markdown(
                                f'<div class="news-title"><a href="{article["url"]}" target="_blank">{article["title"]}</a></div>',
                                unsafe_allow_html=True
                            )
                            
                            # Meta information with sentiment
                            published = format_published_date(article["publishedAt"])
                            st.markdown(
                                f'<div class="news-meta">'
                                f'<span class="source-badge">{source_logo} {source_name}</span> '
                                f'<span class="sentiment-badge {sentiment_class}">{sentiment_emoji} {sentiment_category} ({sentiment_score:.2f})</span> '
                                f'<span>⏱️ {published}</span>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Description/preview
                            if article.get("description"):
                                st.markdown(
                                    f'<div class="news-description">{article["description"][:250]}...</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Read more button
                            st.markdown(
                                f'<a href="{article["url"]}" target="_blank">'
                                f'<button class="read-more-btn">📖 Read Full Article</button>'
                                f'</a>',
                                unsafe_allow_html=True
                            )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add separator between articles (except last)
                        if i < len(articles):
                            st.markdown("<hr style='margin: 0.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
                            
            else:
                st.warning(f"No recent news found for {selected_ticker}. Try a different stock or expand the date range.")

# ============================================================================
# PAGE EXECUTION
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(
        page_title="Stock News with Sentiment",
        page_icon="📰",
        layout="wide"
    )
    
    # Check if API key is configured
    if "NEWS_API_KEY" in st.secrets:
        render_news_page()
    else:
        st.warning("⚠️ News API key not found. Using demo data with sentiment analysis.")
        st.info("To enable live news, add NEWS_API_KEY to your .streamlit/secrets.toml file")
        
        # Demo data with sentiment
        # ... (you can add demo articles here with sentiment pre-calculated)
