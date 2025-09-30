import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from metals_api import MetalsAPIClient, format_price_data, format_historical_data
from forex_api import ForexAPIClient, format_forex_data, format_historical_forex_data
from news_api import NewsAPIClient, get_demo_news_data

# Page configuration
st.set_page_config(
    page_title="Supplier Risk & Recommendation Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title and description
st.title("🏭 Supplier Risk & Recommendation Engine")
st.markdown("**AI-powered decision support for SMEs in exports and manufacturing**")
st.markdown("Monitor raw material costs, score supplier reliability, and get smart sourcing recommendations.")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "Price Monitoring", "News Headlines", "Supplier Analysis", "Risk Alerts", "Recommendations"]
)

# Main content area
if page == "Dashboard":
    st.header("📊 Dashboard")
    st.markdown("Overview of your supply chain health and key metrics")
    
    # Placeholder for dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Suppliers", "12", "2")
    
    with col2:
        st.metric("Price Alerts", "3", "1")
    
    with col3:
        st.metric("Risk Score", "7.2", "0.3")
    
    with col4:
        st.metric("Cost Savings", "$12.5K", "5.2%")

elif page == "Price Monitoring":
    st.header("📈 Price Monitoring")
    st.markdown("Track metal prices and commodity trends in real-time")
    
    # Debug section
    with st.expander("🔧 Debug Information", expanded=False):
        import os
        from dotenv import load_dotenv
        
        load_dotenv(override=True)
        api_key = os.getenv('METALS_API_KEY')
        
        st.write(f"**Environment Variable Status:**")
        st.write(f"- METALS_API_KEY loaded: {api_key is not None}")
        st.write(f"- API key length: {len(api_key) if api_key else 0}")
        st.write(f"- API key starts with NOH4JQ: {api_key.startswith('NOH4JQ') if api_key else False}")
        st.write(f"- Current working directory: {os.getcwd()}")
        st.write(f"- .env file exists: {os.path.exists('.env')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Reload Environment Variables"):
                load_dotenv(override=True)
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset API Client"):
                # Clear all API clients from session state
                if 'metals_client' in st.session_state:
                    del st.session_state.metals_client
                if 'forex_client' in st.session_state:
                    del st.session_state.forex_client
                # Clear any cached data
                if 'imported_data' in st.session_state:
                    del st.session_state.imported_data
                st.rerun()
        
        with col3:
            if st.button("🗑️ Clear All Cache"):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Initialize API clients
    if 'metals_client' not in st.session_state:
        st.session_state.metals_client = MetalsAPIClient()
    
    if 'forex_client' not in st.session_state:
        st.session_state.forex_client = ForexAPIClient()
    
    # Sidebar controls
    st.sidebar.subheader("🔧 Price Monitoring Controls")
    
    # Data type selection
    data_type = st.sidebar.selectbox(
        "Data Type",
        options=['Metals', 'Forex'],
        index=0,
        help="Choose between metals and forex data"
    )
    
    if data_type == 'Metals':
        # Metal selection
        available_metals = ['gold', 'silver', 'copper', 'platinum', 'palladium', 'aluminum', 'zinc', 'lead', 'tin', 'nickel']
        selected_metals = st.sidebar.multiselect(
            "Select Metals to Monitor",
            options=available_metals,
            default=['gold', 'silver', 'copper'],
            help="Choose which metals to display in the dashboard"
        )
        
        # Currency selection for metals
        currency = st.sidebar.selectbox(
            "Currency",
            options=['USD', 'EUR', 'GBP', 'JPY'],
            index=0,
            help="Select the currency for metal price display"
        )
    else:
        # Forex selection
        available_currencies = ['EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'BRL', 'MXN']
        selected_currencies = st.sidebar.multiselect(
            "Select Currencies to Monitor",
            options=available_currencies,
            default=['EUR', 'GBP', 'JPY'],
            help="Choose which currencies to display in the dashboard"
        )
        
        # Base currency selection for forex
        base_currency = st.sidebar.selectbox(
            "Base Currency",
            options=['USD', 'EUR', 'GBP'],
            index=0,
            help="Select the base currency for forex rates"
        )
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Data", type="primary"):
        st.rerun()
    
    # Main content area
    if data_type == 'Metals' and selected_metals:
        # Current prices section
        st.subheader("💰 Current Metal Prices")
        
        with st.spinner("Fetching current metal prices..."):
            prices_data = st.session_state.metals_client.get_metal_prices(selected_metals, currency)
        
        if prices_data:
            # Display prices in a nice format
            price_df = format_price_data(prices_data)
            
            if not price_df.empty:
                # Create columns for better layout
                cols = st.columns(len(selected_metals))
                
                for i, (_, row) in enumerate(price_df.iterrows()):
                    with cols[i % len(cols)]:
                        st.metric(
                            label=row['Metal'],
                            value=row['Price'],
                            delta=None,
                            help=f"Unit: {row['Unit']}"
                        )
                
                # Detailed price table
                st.subheader("📊 Detailed Price Information")
                st.dataframe(price_df, use_container_width=True)
                
                # Historical charts section
                st.subheader("📈 Historical Price Charts")
                
                # Select metal for historical chart
                chart_metal = st.selectbox(
                    "Select metal for historical chart",
                    options=selected_metals,
                    key="chart_metal"
                )
                
                # Date range selection
                days_back = st.selectbox(
                    "Time Period",
                    options=[7, 14, 30, 60, 90, 180, 365],
                    index=2,
                    format_func=lambda x: f"Last {x} days"
                )
                
                if st.button("📊 Generate Chart"):
                    with st.spinner(f"Fetching historical data for {chart_metal}..."):
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                        
                        historical_data = st.session_state.metals_client.get_historical_prices(
                            chart_metal, currency, start_date, end_date
                        )
                        
                        if historical_data and 'prices' in historical_data:
                            hist_df = format_historical_data(historical_data)
                            
                            if not hist_df.empty:
                                # Create interactive chart
                                fig = px.line(
                                    hist_df, 
                                    x='date', 
                                    y='price',
                                    title=f"{chart_metal.title()} Price Trend ({days_back} days)",
                                    labels={'price': f'Price ({currency})', 'date': 'Date'}
                                )
                                
                                fig.update_layout(
                                    xaxis_title="Date",
                                    yaxis_title=f"Price ({currency})",
                                    hovermode='x unified'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Price statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Current Price", f"{hist_df['price'].iloc[-1]:.2f} {currency}")
                                
                                with col2:
                                    price_change = hist_df['price'].iloc[-1] - hist_df['price'].iloc[0]
                                    st.metric("Period Change", f"{price_change:+.2f} {currency}")
                                
                                with col3:
                                    price_change_pct = (price_change / hist_df['price'].iloc[0]) * 100
                                    st.metric("Period Change %", f"{price_change_pct:+.2f}%")
                                
                                with col4:
                                    volatility = hist_df['price'].std()
                                    st.metric("Volatility", f"{volatility:.2f} {currency}")
                            else:
                                st.warning("No historical data available for the selected period.")
                        else:
                            st.error("Failed to fetch historical data.")
            else:
                st.error("No price data available.")
        else:
            st.error("Failed to fetch price data. Please check your API configuration.")
    elif data_type == 'Forex' and selected_currencies:
        # Current forex rates section
        st.subheader("💱 Current Exchange Rates")
        
        with st.spinner("Fetching current exchange rates..."):
            forex_data = st.session_state.forex_client.get_latest_rates(base_currency, selected_currencies)
        
        if forex_data:
            # Display rates in a nice format
            forex_df = format_forex_data(forex_data)
            
            if not forex_df.empty:
                # Create columns for better layout
                cols = st.columns(len(selected_currencies))
                
                for i, (_, row) in enumerate(forex_df.iterrows()):
                    with cols[i % len(cols)]:
                        st.metric(
                            label=f"{row['Currency']}/{row['Base']}",
                            value=row['Rate'],
                            delta=None,
                            help=f"Last updated: {row['Last Updated']}"
                        )
                
                # Detailed rates table
                st.subheader("📊 Detailed Exchange Rates")
                st.dataframe(forex_df, use_container_width=True)
                
                # Historical charts section
                st.subheader("📈 Historical Exchange Rate Charts")
                
                # Select currency for historical chart
                chart_currency = st.selectbox(
                    "Select currency for historical chart",
                    options=selected_currencies,
                    key="forex_chart_currency"
                )
                
                # Date range selection
                days_back = st.selectbox(
                    "Time Period",
                    options=[7, 14, 30, 60, 90, 180, 365],
                    index=2,
                    format_func=lambda x: f"Last {x} days",
                    key="forex_days"
                )
                
                if st.button("📊 Generate Forex Chart"):
                    with st.spinner(f"Fetching historical data for {chart_currency}..."):
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                        
                        historical_forex_data = st.session_state.forex_client.get_time_series(
                            start_date, end_date, base_currency, [chart_currency]
                        )
                        
                        if historical_forex_data and 'rates' in historical_forex_data:
                            hist_df = format_historical_forex_data(historical_forex_data)
                            
                            if not hist_df.empty:
                                # Filter for selected currency
                                currency_df = hist_df[hist_df['currency'] == chart_currency].copy()
                                
                                if not currency_df.empty:
                                    # Create interactive chart
                                    fig = px.line(
                                        currency_df, 
                                        x='date', 
                                        y='rate',
                                        title=f"{chart_currency}/{base_currency} Exchange Rate Trend ({days_back} days)",
                                        labels={'rate': f'Rate ({base_currency})', 'date': 'Date'}
                                    )
                                    
                                    fig.update_layout(
                                        xaxis_title="Date",
                                        yaxis_title=f"Exchange Rate ({base_currency})",
                                        hovermode='x unified'
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Rate statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Current Rate", f"{currency_df['rate'].iloc[-1]:.4f} {base_currency}")
                                    
                                    with col2:
                                        rate_change = currency_df['rate'].iloc[-1] - currency_df['rate'].iloc[0]
                                        st.metric("Period Change", f"{rate_change:+.4f} {base_currency}")
                                    
                                    with col3:
                                        rate_change_pct = (rate_change / currency_df['rate'].iloc[0]) * 100
                                        st.metric("Period Change %", f"{rate_change_pct:+.2f}%")
                                    
                                    with col4:
                                        volatility = currency_df['rate'].std()
                                        st.metric("Volatility", f"{volatility:.4f} {base_currency}")
                                else:
                                    st.warning("No historical data available for the selected currency.")
                            else:
                                st.warning("No historical data available for the selected period.")
                        else:
                            st.error("Failed to fetch historical forex data.")
            else:
                st.error("No forex data available.")
        else:
            st.error("Failed to fetch forex data. Please check your API configuration.")
    
    else:
        if data_type == 'Metals':
            st.info("👆 Please select metals to monitor from the sidebar.")
        else:
            st.info("👆 Please select currencies to monitor from the sidebar.")
    
    # API Configuration Help
    with st.expander("🔧 API Configuration Help"):
        st.markdown("""
        **To use real-time data:**
        
        **Metals Data (metals.dev):**
        1. Get an API key from [metals.dev](https://metals.dev)
        2. Add your API key to the `.env` file:
           ```
           METALS_API_KEY=your_actual_api_key_here
           ```
        
        **Forex Data (Open Exchange Rates):**
        1. Get an App ID from [Open Exchange Rates](https://openexchangerates.org)
        2. Add your App ID to the `.env` file:
           ```
           OPENEXCHANGERATES_APP_ID=your_actual_app_id_here
           ```
        
        3. Restart the Streamlit app
        
        **Current Status:** The app shows demo data when API keys are not configured. Configure your API keys to get real-time data.
        """)

elif page == "News Headlines":
    st.header("📰 News Headlines")
    st.markdown("Stay updated with relevant shipping, metals, and forex news")
    
    # Initialize news client
    try:
        if 'news_client' not in st.session_state:
            st.session_state.news_client = NewsAPIClient()
        news_client = st.session_state.news_client
        api_available = True
    except Exception as e:
        st.warning(f"News API not available: {str(e)}. Showing demo data.")
        news_client = None
        api_available = False
    
    # Sidebar controls
    st.sidebar.subheader("🔧 News Controls")
    
    # Time period selection
    days_back = st.sidebar.selectbox(
        "Time Period",
        options=[1, 3, 7, 14, 30],
        index=2,
        format_func=lambda x: f"Last {x} day{'s' if x > 1 else ''}"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh News", type="primary"):
        st.rerun()
    
    # Main content
    if api_available:
        with st.spinner("Fetching latest news headlines..."):
            # Define example shipping routes
            shipping_routes = ["Asia-Europe", "Trans-Pacific", "Suez Canal", "Panama Canal"]
            
            # Define metals and currencies for news
            metals = ["gold", "silver", "copper", "aluminum", "steel"]
            currencies = ["USD", "EUR", "GBP", "JPY", "CNY"]
            
            # Fetch news data
            shipping_news = news_client.get_shipping_news(shipping_routes, days_back)
            metals_forex_news = news_client.get_metals_forex_news(metals, currencies, days_back)
            business_news = news_client.get_business_headlines(days_back)
            
            # Format news data
            shipping_formatted = news_client.format_news_data(shipping_news)
            metals_forex_formatted = news_client.format_news_data(metals_forex_news)
            business_formatted = news_client.format_news_data(business_news)
    else:
        # Use demo data
        demo_data = get_demo_news_data()
        shipping_formatted = demo_data['shipping_news']
        metals_forex_formatted = demo_data['metals_forex_news']
        business_formatted = demo_data['business_news']
    
    # Display news sections
    tab1, tab2, tab3 = st.tabs(["🚢 Shipping News", "💰 Metals & Forex", "📈 Business News"])
    
    with tab1:
        st.subheader("🚢 Shipping & Logistics News")
        st.markdown(f"*Latest news about shipping routes and logistics*")
        
        if shipping_formatted:
            for i, article in enumerate(shipping_formatted):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{article['title']}**")
                        st.markdown(f"*{article['description']}*")
                        st.markdown(f"📅 {article['published_at']} | 📰 {article['source']}")
                        if article.get('route'):
                            st.markdown(f"🛣️ Route: {article['route']}")
                    
                    with col2:
                        if article['url'] != '#':
                            st.link_button("Read More", article['url'])
                        else:
                            st.info("Demo Article")
                    
                    if i < len(shipping_formatted) - 1:
                        st.markdown("---")
        else:
            st.info("No shipping news available for the selected time period.")
    
    with tab2:
        st.subheader("💰 Metals & Forex News")
        st.markdown(f"*Latest news about metal prices and currency markets*")
        
        if metals_forex_formatted:
            for i, article in enumerate(metals_forex_formatted):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{article['title']}**")
                        st.markdown(f"*{article['description']}*")
                        st.markdown(f"📅 {article['published_at']} | 📰 {article['source']} | 🏷️ {article['category']}")
                    
                    with col2:
                        if article['url'] != '#':
                            st.link_button("Read More", article['url'])
                        else:
                            st.info("Demo Article")
                    
                    if i < len(metals_forex_formatted) - 1:
                        st.markdown("---")
        else:
            st.info("No metals/forex news available for the selected time period.")
    
    with tab3:
        st.subheader("📈 Business & Economy News")
        st.markdown(f"*Latest business news affecting supply chains*")
        
        if business_formatted:
            for i, article in enumerate(business_formatted):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        st.markdown(f"**{article['title']}**")
                        st.markdown(f"*{article['description']}*")
                        st.markdown(f"📅 {article['published_at']} | 📰 {article['source']}")
                    
                    with col2:
                        if article['url'] != '#':
                            st.link_button("Read More", article['url'])
                        else:
                            st.info("Demo Article")
                    
                    if i < len(business_formatted) - 1:
                        st.markdown("---")
        else:
            st.info("No business news available for the selected time period.")
    
    # API Configuration Help
    with st.expander("🔧 News API Configuration Help"):
        st.markdown("""
        **To use real-time news data:**
        
        **News API:**
        1. Get an API key from [NewsAPI](https://newsapi.org)
        2. Add your API key to the `.env` file:
           ```
           NEWS_API_KEY=your_actual_api_key_here
           ```
        
        3. Restart the Streamlit app
        
        **Current Status:** The app shows demo data when the News API key is not configured. Configure your API key to get real-time news headlines.
        """)

elif page == "Supplier Analysis":
    st.header("🏢 Supplier Analysis")
    st.markdown("Analyze supplier performance and reliability scores")
    
    # Placeholder for supplier analysis content
    st.info("Supplier analysis functionality will be implemented here")

elif page == "Risk Alerts":
    st.header("⚠️ Risk Alerts")
    st.markdown("Get notified about potential supply chain disruptions")
    
    # Placeholder for risk alerts content
    st.info("Risk alerts functionality will be implemented here")

elif page == "Recommendations":
    st.header("💡 Recommendations")
    st.markdown("AI-powered sourcing recommendations and alternatives")
    
    # Placeholder for recommendations content
    st.info("Recommendations functionality will be implemented here")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for SMEs in exports and manufacturing")
