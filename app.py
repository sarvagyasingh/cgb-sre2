import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from metals_api import MetalsAPIClient, format_price_data, format_historical_data
from forex_api import ForexAPIClient, format_forex_data, format_historical_forex_data
from news_api import (
    NewsAPIClient,
    generate_demo_country_news,
    get_demo_news_data,
)
from supplier_risk import (
    aggregate_supplier_metrics,
    build_country_risk_table,
    compute_country_news_scores,
    compute_supplier_risk_scores,
    load_supplier_dataset,
)
from openai_client import OpenAIAlertClient
try:
    from forex_forecast import forecast_next_rate, load_model_bundle, ArtefactMissingError
    FORECASTING_AVAILABLE = True
except Exception as e:
    FORECASTING_AVAILABLE = False
    def forecast_next_rate(*args, **kwargs):
        raise ImportError("Forecasting not available - TensorFlow error")
    def load_model_bundle(*args, **kwargs):
        raise ImportError("Forecasting not available - TensorFlow error")
    class ArtefactMissingError(Exception):
        pass

st.set_page_config(
    page_title="Supplier Risk & Recommendation Engine",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏭 Supplier Risk & Recommendation Engine")
st.markdown("**AI-powered decision support for SMEs in exports and manufacturing**")
st.markdown("Monitor raw material costs, score supplier reliability, and get smart sourcing recommendations.")

st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Dashboard", "Price Monitoring", "News Headlines", "Supplier Analysis", "Risk Alerts", "Recommendations", "Forex Forecasting"]
)

if page == "Dashboard":
    st.header("📊 Dashboard")
    st.markdown("Overview of your supply chain health and key metrics")
    
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
    

elif page == "Supplier Analysis":
    st.header("🏢 Supplier Analysis")
    st.markdown("Analyze supplier performance, risk scores, and country-level context")

    @st.cache_data(show_spinner=False)
    def load_supplier_data() -> pd.DataFrame:
        return load_supplier_dataset("data/raw_material_suppliers_timeseries.csv")

    supplier_df = load_supplier_data()

    if supplier_df.empty:
        st.warning("Supplier dataset is empty. Please upload data to view analysis.")
    else:
        min_date = supplier_df["order_date"].min().date()
        max_date = supplier_df["order_date"].max().date()

        st.sidebar.subheader("📅 Supplier Filters")
        date_range = st.sidebar.date_input(
            "Order date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        country_options = sorted(supplier_df["country"].dropna().unique())
        selected_countries = st.sidebar.multiselect(
            "Supplier countries",
            options=country_options,
            default=country_options,
        )

        material_options = sorted(supplier_df["material"].dropna().unique())
        selected_materials = st.sidebar.multiselect(
            "Materials",
            options=material_options,
            default=material_options[:5] if len(material_options) > 5 else material_options,
        )

        news_lookback = st.sidebar.slider(
            "News lookback (days)",
            min_value=3,
            max_value=30,
            value=10,
            help="Number of days of headlines to include in the country risk signal",
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = min_date
            end_date = max_date

        filtered_df = supplier_df[
            (supplier_df["order_date"] >= pd.Timestamp(start_date))
            & (supplier_df["order_date"] <= pd.Timestamp(end_date))
        ]

        if selected_countries:
            filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]

        if selected_materials:
            filtered_df = filtered_df[filtered_df["material"].isin(selected_materials)]

        if filtered_df.empty:
            st.warning("No supplier records match the selected filters.")
        else:
            st.subheader("Supplier performance overview")
            metrics_df = aggregate_supplier_metrics(filtered_df)

            countries = metrics_df["country"].dropna().unique().tolist()
            news_client = st.session_state.get("country_news_client")
            news_error = None

            if news_client is None and countries:
                try:
                    news_client = NewsAPIClient()
                    st.session_state["country_news_client"] = news_client
                except Exception as exc:  # pragma: no cover - depends on env
                    news_error = str(exc)
                    news_client = None

            if countries:
                with st.spinner("Evaluating country-level news signals..."):
                    if news_client:
                        country_news = news_client.get_supply_chain_news_for_countries(
                            countries, days_back=news_lookback
                        )
                    else:
                        country_news = generate_demo_country_news(countries)
            else:
                country_news = {}

            country_news_scores = compute_country_news_scores(country_news)
            supplier_scores, supplier_results = compute_supplier_risk_scores(
                metrics_df, country_news_scores
            )

            avg_risk = supplier_scores["risk_score"].mean()
            high_risk_count = (supplier_scores["risk_level"] == "High").sum()
            total_spend = filtered_df["total_price"].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("Average risk score", f"{avg_risk:.1f} / 100")
            col2.metric("High risk suppliers", int(high_risk_count))
            col3.metric("Spend in scope", f"${total_spend:,.0f}")

            risk_table = supplier_scores[
                [
                    "supplier_name",
                    "country",
                    "risk_score",
                    "risk_level",
                    "top_factor",
                    "total_orders",
                    "total_spend",
                    "avg_delivery_time",
                    "on_time_rate",
                    "defect_rate",
                    "news_severity",
                ]
            ].sort_values("risk_score", ascending=False)

            st.dataframe(
                risk_table,
                use_container_width=True,
                column_config={
                    "risk_score": st.column_config.NumberColumn("Risk score", format="%.1f"),
                    "on_time_rate": st.column_config.ProgressColumn(
                        "On-time rate", format="%.0f%%", min_value=0, max_value=1
                    ),
                    "defect_rate": st.column_config.NumberColumn("Defect rate", format="%.2f"),
                },
            )

            st.markdown("---")

            st.subheader("Risk breakdown by supplier")
            supplier_options = [
                f"{row.supplier_name} ({row.country})" for _, row in supplier_scores.iterrows()
            ]
            selection = st.selectbox(
                "Select a supplier for detailed explainability",
                options=supplier_options,
            )

            if selection:
                selected_index = supplier_options.index(selection)
                selected_result = supplier_results[selected_index]
                contributions = selected_result.top_factors

                contrib_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=[factor for factor, _ in contributions],
                            y=[value for _, value in contributions],
                            marker_color="#FF7F50",
                        )
                    ]
                )
                contrib_fig.update_layout(
                    yaxis_title="Contribution to risk score",
                    xaxis_title="Risk factor",
                    height=400,
                    margin=dict(t=40, b=40, l=20, r=20),
                )
                st.plotly_chart(contrib_fig, use_container_width=True)

                st.markdown(
                    f"**Overall risk score:** {selected_result.risk_score:.1f} / 100 ({selected_result.risk_level})"
                )

                detail_cols = st.columns(3)
                detail_cols[0].metric(
                    "Avg. delivery time (days)", f"{selected_result.metrics['avg_delivery_time']:.1f}"
                )
                detail_cols[1].metric(
                    "On-time delivery",
                    f"{selected_result.metrics['on_time_rate'] * 100:.1f}%",
                )
                detail_cols[2].metric(
                    "Defect rate", f"{selected_result.metrics['defect_rate'] * 100:.2f}%"
                )

                st.markdown("**Top contributing risk factors:**")
                for factor, value in contributions[:3]:
                    st.markdown(f"- {factor}: {value:.1f} points")

            st.markdown("---")
            st.subheader("Country-level news signal")

            if news_error:
                st.info(
                    "Real-time news could not be loaded: "
                    f"{news_error}. Displaying demo news stories instead."
                )

            country_summary = build_country_risk_table(country_news_scores)
            if not country_summary.empty:
                st.dataframe(country_summary, use_container_width=True)

            for country in countries:
                articles = country_news.get(country, [])
                if not articles:
                    continue
                with st.expander(f"{country} headlines", expanded=False):
                    for article in articles[:3]:
                        st.markdown(f"**{article.get('title')}**")
                        if article.get('description'):
                            st.write(article['description'])
                        meta_bits = []
                        if article.get('source'):
                            meta_bits.append(article['source'] if isinstance(article['source'], str) else article['source'].get('name', ''))
                        if article.get('publishedAt'):
                            meta_bits.append(article['publishedAt'])
                        if meta_bits:
                            st.caption(" • ".join([bit for bit in meta_bits if bit]))
                        if article.get('url') and article.get('url') != '#':
                            st.link_button("Read more", article['url'])

            with st.expander("ℹ️ How the risk score is calculated"):
                st.markdown(
                    """
                    **Explainable supplier risk score**

                    The score ranges from 0 (low risk) to 100 (high risk) and blends five observable
                    performance signals with a country news adjustment:

                    - Delivery performance (25%) — slower average delivery times increase risk.
                    - On-time reliability (20%) — missed delivery commitments add risk.
                    - Quality issues (20%) — higher defect rates raise the score.
                    - Price volatility (15%) — volatile pricing indicates potential instability.
                    - Spend concentration (10%) — large wallet share concentrates exposure.
                    - Country news (10%) — supply-chain related headlines can increase country-level risk.

                    Each component is normalised across the filtered suppliers and converted into
                    an additive score so you can see which factors drive risk for each supplier.
                    """
                )

elif page == "Risk Alerts":
    st.header("⚠️ Risk Alerts")
    st.markdown("Get notified about potential supply chain disruptions")

    # Controls
    st.sidebar.subheader("🔧 Alerts Controls")
    news_lookback = st.sidebar.slider(
        "News lookback (days)", min_value=3, max_value=30, value=10,
        help="Days of headlines included in country risk signal",
    )
    supplier_top_n = st.sidebar.slider(
        "Max suppliers in context", min_value=3, max_value=20, value=8,
        help="How many highest-risk suppliers to include in alert context",
    )
    fx_abs_threshold = st.sidebar.number_input(
        "FX day-over-day move threshold (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="Minimum absolute % change to flag FX moves",
    )
    include_forecast = st.sidebar.checkbox(
        "Include LSTM forecast deltas (if available)", value=False,
        help="Adds short-horizon forecast moves when model artifacts exist",
    )

    # Build structured context
    context = {
        "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "base_currency": "EUR",
    }

    # 1) Supplier risk context
    try:
        supplier_df = load_supplier_dataset("data/raw_material_suppliers_timeseries.csv")
        if supplier_df is not None and not supplier_df.empty:
            metrics_df = aggregate_supplier_metrics(supplier_df)

            # Country news scores feeding into supplier scoring
            countries = metrics_df["country"].dropna().unique().tolist()
            try:
                news_client = NewsAPIClient()
                country_news = news_client.get_supply_chain_news_for_countries(
                    countries, days_back=news_lookback
                )
            except Exception:
                country_news = generate_demo_country_news(countries)

            country_news_scores = compute_country_news_scores(country_news)
            supplier_scores, _supplier_results = compute_supplier_risk_scores(
                metrics_df, country_news_scores
            )

            # Top high-risk suppliers
            top_suppliers = (
                supplier_scores
                .sort_values("risk_score", ascending=False)
                .head(supplier_top_n)
            )
            context["suppliers_high_risk"] = [
                {
                    "supplier_name": row.supplier_name,
                    "country": row.country,
                    "risk_score": float(row.risk_score),
                    "risk_level": row.risk_level,
                    "top_factor": row.top_factor,
                    # Provide both numeric news score and label
                    "news_score": float(country_news_scores.get(row.country, {}).get("score", 0.0)),
                    "news_severity_label": str(row.news_severity),
                }
                for _, row in top_suppliers.iterrows()
            ]

            # Country hotspots from news
            hotspots = [
                {
                    "country": c,
                    "news_score": float(s.get("score", 0.0)),
                    "severity_label": s.get("severity", "Low"),
                }
                for c, s in country_news_scores.items()
            ]
            hotspots = sorted(hotspots, key=lambda x: x["news_score"], reverse=True)[:5]
            context["country_news_hotspots"] = hotspots
        else:
            context["suppliers_high_risk"] = []
            context["country_news_hotspots"] = []
    except Exception as exc:
        st.warning(f"Supplier context error: {exc}")
        context["suppliers_high_risk"] = []
        context["country_news_hotspots"] = []

    # 2) FX spikes from latest historical file and optional forecast deltas
    fx_spikes = []
    try:
        hist = pd.read_csv("data/daily_forex_rates.csv", parse_dates=["date"])  # columns: date, base_currency, currency, exchange_rate
        # Focus on latest two dates per currency for EUR base
        eur_hist = hist[(hist["base_currency"] == "EUR")].copy()
        latest_date = eur_hist["date"].max()
        prior_date = eur_hist[eur_hist["date"] < latest_date]["date"].max()
        latest = eur_hist[eur_hist["date"] == latest_date]
        prior = eur_hist[eur_hist["date"] == prior_date][["currency", "exchange_rate"]].rename(columns={"exchange_rate": "exchange_rate_prior"})
        joined = latest.merge(prior, on="currency", how="left")
        joined["pct_change_1d"] = (joined["exchange_rate"] - joined["exchange_rate_prior"]) / joined["exchange_rate_prior"] * 100.0
        moved = joined[joined["pct_change_1d"].abs() >= fx_abs_threshold]
        for _, r in moved.sort_values("pct_change_1d", key=lambda s: s.abs(), ascending=False).head(8).iterrows():
            fx_spikes.append({
                "currency": r["currency"],
                "base_currency": "EUR",
                "pct_change_1d": float(r["pct_change_1d"]),
                "last_rate": float(r["exchange_rate"]),
                "date": latest_date.strftime('%Y-%m-%d'),
            })
    except Exception as exc:
        st.info(f"FX spike detection skipped: {exc}")

    # Optional: include LSTM forecast delta vs last observation (short horizon)
    if include_forecast:
        try:
            if 'FORECASTING_AVAILABLE' in globals() and FORECASTING_AVAILABLE:
                # Try a short 3-day horizon for a few common currencies
                tracked = ["USD", "GBP", "JPY"]
                for cur in tracked:
                    try:
                        result = forecast_next_rate(steps=3)
                        if result and result.get('currency') == cur:
                            last_obs = float(result.get('last_observation', 0.0))
                            if result.get('predictions'):
                                day3 = float(result['predictions'][-1])
                                pct = (day3 - last_obs) / last_obs * 100.0 if last_obs else 0.0
                                fx_spikes.append({
                                    "currency": cur,
                                    "base_currency": "EUR",
                                    "forecast_pct_change_3d": pct,
                                })
                    except Exception:
                        continue
        except Exception:
            pass

    context["forex_spikes"] = fx_spikes

    # Generate alerts
    client = OpenAIAlertClient()
    alerts = client.generate_alerts(context, max_alerts=10)

    if alerts:
        st.subheader("📣 Alerts")
        for a in alerts:
            st.markdown(f"- {a}")
    else:
        st.info("No high-priority alerts based on current context.")

    # Indicate whether OpenAI API was used
    diag = client.diagnostics()
    if diag.get("last_used_openai"):
        st.caption("✅ OpenAI API used to generate alerts")
    else:
        st.caption("ℹ️ Rule-based fallback used (OpenAI not configured or unavailable)")



elif page == "Recommendations":
    st.header("💡 Recommendations")
    st.markdown("AI-powered sourcing recommendations and diversification strategies")
    
    # Load supplier data
    @st.cache_data(show_spinner=False)
    def load_supplier_data() -> pd.DataFrame:
        return load_supplier_dataset("data/raw_material_suppliers_timeseries.csv")

    supplier_df = load_supplier_data()

    if supplier_df.empty:
        st.warning("Supplier dataset is empty. Please upload data to view recommendations.")
    else:
        # Sidebar controls
        st.sidebar.subheader("🔧 Recommendation Filters")
        
        # Material selection
        material_options = sorted(supplier_df["material"].dropna().unique())
        selected_material = st.sidebar.selectbox(
            "Select Material",
            options=material_options,
            help="Choose the material to get recommendations for"
        )
        
        # Country selection
        country_options = sorted(supplier_df["country"].dropna().unique())
        selected_countries = st.sidebar.multiselect(
            "Include Countries",
            options=country_options,
            default=country_options,
            help="Select countries to include in recommendations"
        )
        
        # Ranking weights
        st.sidebar.subheader("⚖️ Ranking Weights")
        cost_weight = st.sidebar.slider(
            "Cost Importance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.4, 
            step=0.1,
            help="How much to weight cost in the ranking"
        )
        reliability_weight = st.sidebar.slider(
            "Reliability Importance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="How much to weight reliability (on-time delivery, quality)"
        )
        risk_weight = st.sidebar.slider(
            "Risk Importance", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.1,
            help="How much to weight risk factors"
        )
        
        # Normalize weights
        total_weight = cost_weight + reliability_weight + risk_weight
        if total_weight > 0:
            cost_weight /= total_weight
            reliability_weight /= total_weight
            risk_weight /= total_weight
        
        # Diversification options
        st.sidebar.subheader("🌍 Diversification Options")
        max_suppliers = st.sidebar.slider(
            "Max Suppliers per Strategy",
            min_value=2,
            max_value=10,
            value=5,
            help="Maximum number of suppliers to include in diversification strategies"
        )
        
        min_countries = st.sidebar.slider(
            "Min Countries for Diversification",
            min_value=2,
            max_value=5,
            value=2,
            help="Minimum number of countries to include in diversification strategies"
        )
        
        # Filter data
        filtered_df = supplier_df[supplier_df["material"] == selected_material]
        if selected_countries:
            filtered_df = filtered_df[filtered_df["country"].isin(selected_countries)]
        
        if filtered_df.empty:
            st.warning(f"No suppliers found for {selected_material} in selected countries.")
        else:
            # Calculate supplier metrics
            metrics_df = aggregate_supplier_metrics(filtered_df)
            
            # Get country news scores for risk assessment
            countries = metrics_df["country"].dropna().unique().tolist()
            try:
                news_client = NewsAPIClient()
                country_news = news_client.get_supply_chain_news_for_countries(
                    countries, days_back=10
                )
            except Exception:
                country_news = generate_demo_country_news(countries)
            
            country_news_scores = compute_country_news_scores(country_news)
            supplier_scores, supplier_results = compute_supplier_risk_scores(
                metrics_df, country_news_scores
            )
            
            # Calculate composite ranking scores
            def calculate_composite_score(row):
                # Cost score (lower is better) - normalize and invert
                cost_score = 1 - (row['avg_price_per_unit'] - metrics_df['avg_price_per_unit'].min()) / (metrics_df['avg_price_per_unit'].max() - metrics_df['avg_price_per_unit'].min())
                
                # Reliability score (higher is better)
                reliability_score = (row['on_time_rate'] + (1 - row['defect_rate'])) / 2
                
                # Risk score (lower is better) - invert the risk score
                risk_score = 1 - (row['risk_score'] / 100)
                
                # Weighted composite score
                composite = (cost_weight * cost_score + 
                           reliability_weight * reliability_score + 
                           risk_weight * risk_score)
                
                return composite
            
            supplier_scores['composite_score'] = supplier_scores.apply(calculate_composite_score, axis=1)
            supplier_scores['rank'] = supplier_scores['composite_score'].rank(ascending=False, method='dense').astype(int)
            
            # Sort by composite score
            ranked_suppliers = supplier_scores.sort_values('composite_score', ascending=False)
            
            # Display top suppliers
            st.subheader("🏆 Top Supplier Rankings")
            st.markdown(f"*Ranked by cost, reliability, and risk for {selected_material}*")
            
            # Create display columns
            display_cols = [
                'rank', 'supplier_name', 'country', 'composite_score', 
                'avg_price_per_unit', 'on_time_rate', 'defect_rate', 'risk_score', 'risk_level'
            ]
            
            display_df = ranked_suppliers[display_cols].copy()
            display_df['composite_score'] = display_df['composite_score'].round(3)
            display_df['avg_price_per_unit'] = display_df['avg_price_per_unit'].round(2)
            display_df['on_time_rate'] = (display_df['on_time_rate'] * 100).round(1)
            display_df['defect_rate'] = (display_df['defect_rate'] * 100).round(2)
            display_df['risk_score'] = display_df['risk_score'].round(1)
            
            # Rename columns for display
            display_df.columns = [
                'Rank', 'Supplier', 'Country', 'Score', 
                'Avg Price/Unit', 'On-Time %', 'Defect %', 'Risk Score', 'Risk Level'
            ]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", format="%.3f", min_value=0, max_value=1
                    ),
                    "On-Time %": st.column_config.ProgressColumn(
                        "On-Time %", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "Defect %": st.column_config.NumberColumn("Defect %", format="%.2f%%"),
                    "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.1f"),
                }
            )
            
            # Detailed analysis for top supplier
            st.subheader("🔍 Top Supplier Analysis")
            top_supplier = ranked_suppliers.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Supplier", top_supplier['supplier_name'])
            with col2:
                st.metric("Country", top_supplier['country'])
            with col3:
                st.metric("Composite Score", f"{top_supplier['composite_score']:.3f}")
            with col4:
                st.metric("Risk Level", top_supplier['risk_level'])
            
            # Cost vs Quality scatter plot
            st.subheader("📊 Cost vs Quality Analysis")
            
            fig = px.scatter(
                ranked_suppliers,
                x='avg_price_per_unit',
                y='on_time_rate',
                size='total_spend',
                color='risk_level',
                hover_data=['supplier_name', 'country', 'defect_rate', 'risk_score'],
                title=f"Cost vs Quality for {selected_material} Suppliers",
                labels={
                    'avg_price_per_unit': 'Average Price per Unit (USD)',
                    'on_time_rate': 'On-Time Delivery Rate',
                    'total_spend': 'Total Spend (USD)'
                }
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Diversification strategies
            st.subheader("🌍 Diversification Strategies")
            st.markdown("Explore different sourcing strategies to reduce risk and improve resilience")
            
            # Strategy 1: Geographic Diversification
            st.markdown("### 🌏 Geographic Diversification")
            
            # Group by country and get top suppliers per country
            country_suppliers = {}
            for country in ranked_suppliers['country'].unique():
                country_data = ranked_suppliers[ranked_suppliers['country'] == country]
                country_suppliers[country] = country_data.head(2)  # Top 2 per country
            
            # Create diversification table
            diversification_data = []
            for country, suppliers in country_suppliers.items():
                for _, supplier in suppliers.iterrows():
                    diversification_data.append({
                        'Country': country,
                        'Supplier': supplier['supplier_name'],
                        'Score': supplier['composite_score'],
                        'Price': supplier['avg_price_per_unit'],
                        'Risk': supplier['risk_level'],
                        'On-Time %': supplier['on_time_rate'] * 100
                    })
            
            if diversification_data:
                geo_df = pd.DataFrame(diversification_data)
                geo_df = geo_df.sort_values(['Country', 'Score'], ascending=[True, False])
                
                st.markdown("**Top suppliers by country:**")
                st.dataframe(
                    geo_df,
                    use_container_width=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", format="%.3f", min_value=0, max_value=1
                        ),
                        "On-Time %": st.column_config.ProgressColumn(
                            "On-Time %", format="%.1f%%", min_value=0, max_value=100
                        ),
                    }
                )
                
                # Calculate diversification metrics
                countries_included = geo_df['Country'].nunique()
                avg_score = geo_df['Score'].mean()
                price_range = geo_df['Price'].max() - geo_df['Price'].min()
                price_cv = geo_df['Price'].std() / geo_df['Price'].mean()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Countries", countries_included)
                with col2:
                    st.metric("Avg Score", f"{avg_score:.3f}")
                with col3:
                    st.metric("Price Range", f"${price_range:.0f}")
                with col4:
                    st.metric("Price CV", f"{price_cv:.2f}")
                
                # Trade-offs analysis
                st.markdown("**Trade-offs of Geographic Diversification:**")
                
                if countries_included >= min_countries:
                    st.success("✅ **Pros:**")
                    st.markdown("- Reduced country-specific risk exposure")
                    st.markdown("- Better resilience to regional disruptions")
                    st.markdown("- Access to different cost structures")
                    
                    if price_cv > 0.1:
                        st.warning("⚠️ **Cons:**")
                        st.markdown("- Price variability across suppliers")
                        st.markdown("- Increased complexity in supplier management")
                        st.markdown("- Potential quality inconsistencies")
                    else:
                        st.info("ℹ️ **Cons:**")
                        st.markdown("- Increased complexity in supplier management")
                        st.markdown("- Potential quality inconsistencies")
                else:
                    st.warning("⚠️ **Limited diversification:** Not enough countries available for effective geographic diversification.")
            
            # Strategy 2: Risk-Based Diversification
            st.markdown("### ⚖️ Risk-Based Diversification")
            
            # Categorize suppliers by risk level
            low_risk = ranked_suppliers[ranked_suppliers['risk_level'] == 'Low'].head(3)
            medium_risk = ranked_suppliers[ranked_suppliers['risk_level'] == 'Medium'].head(2)
            high_risk = ranked_suppliers[ranked_suppliers['risk_level'] == 'High'].head(1)
            
            risk_strategy_data = []
            for risk_level, suppliers in [('Low Risk', low_risk), ('Medium Risk', medium_risk), ('High Risk', high_risk)]:
                for _, supplier in suppliers.iterrows():
                    risk_strategy_data.append({
                        'Risk Category': risk_level,
                        'Supplier': supplier['supplier_name'],
                        'Country': supplier['country'],
                        'Score': supplier['composite_score'],
                        'Price': supplier['avg_price_per_unit'],
                        'Risk Score': supplier['risk_score']
                    })
            
            if risk_strategy_data:
                risk_df = pd.DataFrame(risk_strategy_data)
                
                st.markdown("**Balanced risk portfolio:**")
                st.dataframe(
                    risk_df,
                    use_container_width=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", format="%.3f", min_value=0, max_value=1
                        ),
                        "Risk Score": st.column_config.NumberColumn("Risk Score", format="%.1f"),
                    }
                )
                
                # Risk diversification metrics
                risk_categories = risk_df['Risk Category'].nunique()
                avg_risk_score = risk_df['Risk Score'].mean()
                price_std = risk_df['Price'].std()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Categories", risk_categories)
                with col2:
                    st.metric("Avg Risk Score", f"{avg_risk_score:.1f}")
                with col3:
                    st.metric("Price Std Dev", f"${price_std:.0f}")
                
                st.markdown("**Trade-offs of Risk-Based Diversification:**")
                st.success("✅ **Pros:**")
                st.markdown("- Balanced exposure across risk levels")
                st.markdown("- Maintains some high-performing suppliers")
                st.markdown("- Reduces overall portfolio risk")
                
                st.info("ℹ️ **Cons:**")
                st.markdown("- Includes some higher-risk suppliers")
                st.markdown("- May increase average cost")
                st.markdown("- Requires careful monitoring of high-risk suppliers")
            
            # Strategy 3: Cost-Optimized Diversification
            st.markdown("### 💰 Cost-Optimized Diversification")
            
            # Get suppliers with good balance of cost and quality
            cost_optimized = ranked_suppliers[
                (ranked_suppliers['composite_score'] >= ranked_suppliers['composite_score'].quantile(0.3)) &
                (ranked_suppliers['avg_price_per_unit'] <= ranked_suppliers['avg_price_per_unit'].quantile(0.7))
            ].head(max_suppliers)
            
            if not cost_optimized.empty:
                st.markdown("**Cost-effective suppliers with good quality:**")
                cost_display = cost_optimized[['supplier_name', 'country', 'composite_score', 'avg_price_per_unit', 'on_time_rate', 'risk_level']].copy()
                cost_display['on_time_rate'] = (cost_display['on_time_rate'] * 100).round(1)
                cost_display.columns = ['Supplier', 'Country', 'Score', 'Price/Unit', 'On-Time %', 'Risk Level']
                
                st.dataframe(
                    cost_display,
                    use_container_width=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            "Score", format="%.3f", min_value=0, max_value=1
                        ),
                        "On-Time %": st.column_config.ProgressColumn(
                            "On-Time %", format="%.1f%%", min_value=0, max_value=100
                        ),
                    }
                )
                
                # Cost optimization metrics
                avg_cost = cost_optimized['avg_price_per_unit'].mean()
                cost_savings = ranked_suppliers['avg_price_per_unit'].mean() - avg_cost
                cost_savings_pct = (cost_savings / ranked_suppliers['avg_price_per_unit'].mean()) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Cost", f"${avg_cost:.0f}")
                with col2:
                    st.metric("Cost Savings", f"${cost_savings:.0f}")
                with col3:
                    st.metric("Savings %", f"{cost_savings_pct:.1f}%")
                
                st.markdown("**Trade-offs of Cost-Optimized Diversification:**")
                st.success("✅ **Pros:**")
                st.markdown("- Significant cost savings potential")
                st.markdown("- Maintains quality standards")
                st.markdown("- Good value for money")
                
                st.warning("⚠️ **Cons:**")
                st.markdown("- May have higher risk exposure")
                st.markdown("- Potential quality variability")
                st.markdown("- May require more supplier management")
            
            # Implementation recommendations
            st.subheader("🚀 Implementation Recommendations")
            
            st.markdown("### Next Steps:")
            st.markdown("1. **Start with the top-ranked supplier** for immediate implementation")
            st.markdown("2. **Develop backup suppliers** from the diversification strategies")
            st.markdown("3. **Monitor performance** using the risk scoring system")
            st.markdown("4. **Adjust strategy** based on market conditions and supplier performance")
            
            # Export recommendations
            if st.button("📥 Export Recommendations", type="primary"):
                # Create comprehensive recommendations export
                export_data = []
                for _, supplier in ranked_suppliers.iterrows():
                    export_data.append({
                        'Rank': supplier['rank'],
                        'Supplier': supplier['supplier_name'],
                        'Country': supplier['country'],
                        'Material': selected_material,
                        'Composite Score': supplier['composite_score'],
                        'Price per Unit': supplier['avg_price_per_unit'],
                        'On-Time Rate': supplier['on_time_rate'],
                        'Defect Rate': supplier['defect_rate'],
                        'Risk Score': supplier['risk_score'],
                        'Risk Level': supplier['risk_level'],
                        'Total Orders': supplier['total_orders'],
                        'Total Spend': supplier['total_spend']
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"supplier_recommendations_{selected_material}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

elif page == "Forex Forecasting":
    st.header("🔮 Forex Forecasting")
    st.markdown("AI-powered exchange rate predictions using LSTM neural networks")
    
    # Check if forecasting is available
    if not FORECASTING_AVAILABLE:
        st.error("❌ Forecasting not available")
        st.warning("""
        **TensorFlow is required for forecasting functionality.**
        
        There seems to be an issue with TensorFlow. To fix this:
        1. Try reinstalling TensorFlow:
           ```bash
           pip uninstall tensorflow
           pip install tensorflow
           ```
        2. Or try the Apple Silicon optimized version:
           ```bash
           pip install tensorflow-macos tensorflow-metal
           ```
        3. Restart the Streamlit app
        """)
        st.stop()
    
    # Sidebar controls for forecasting
    st.sidebar.subheader("🔧 Forecasting Controls")
    
    # Currency selection
    available_currencies = ['USD', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY', 'INR', 'BRL', 'MXN']
    selected_currency = st.sidebar.selectbox(
        "Target Currency",
        options=available_currencies,
        index=0,
        help="Currency to forecast against EUR base"
    )
    
    # Forecast horizon
    forecast_days = st.sidebar.slider(
        "Forecast Horizon (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days to forecast ahead"
    )
    
    # Confidence level
    confidence_level = st.sidebar.selectbox(
        "Confidence Level",
        options=[0.68, 0.80, 0.90, 0.95],
        index=2,
        format_func=lambda x: f"{int(x*100)}%",
        help="Statistical confidence level for prediction intervals"
    )
    
    # Model retraining option
    retrain_model = st.sidebar.checkbox(
        "Retrain Model",
        value=False,
        help="Retrain the LSTM model with latest data (may take a few minutes)"
    )
    
    # Main forecasting content
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"📈 {selected_currency}/EUR Forecast")
    
    with col2:
        if st.button("🔄 Generate Forecast", type="primary"):
            st.rerun()
    
    with col3:
        if st.button("📊 Model Info"):
            st.rerun()
    
    # Check if model artifacts exist
    from pathlib import Path
    model_path = Path("models/forex_lstm.keras")
    scaler_path = Path("models/forex_scaler.pkl")
    
    try:
        # Load model bundle to check if artifacts exist
        model, scaler, metadata = load_model_bundle(
            model_path=model_path,
            scaler_path=scaler_path,
            auto_train=False
        )
        
        # Display model information
        with st.expander("ℹ️ Model Information", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Lookback Window", f"{metadata.get('lookback', 30)} days")
            with col2:
                st.metric("Training Samples", f"{metadata.get('training_rows', 'Unknown')}")
            with col3:
                st.metric("Base Currency", metadata.get('base_currency', 'EUR'))
        
        # Generate forecast
        with st.spinner(f"Generating {forecast_days}-day forecast for {selected_currency}/EUR..."):
            try:
                forecast_result = forecast_next_rate(
                    steps=forecast_days,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    auto_train=retrain_model
                )
                
                # Check if the forecast is for the selected currency
                if forecast_result.get('currency') != selected_currency:
                    st.warning(f"⚠️ Model was trained for {forecast_result.get('currency')}, but you selected {selected_currency}. Consider retraining the model for {selected_currency}.")
                
                # Display current rate and immediate forecast
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Rate",
                        f"{forecast_result['last_observation']:.4f}",
                        help=f"Last observed rate on {forecast_result['last_observation_date']}"
                    )
                
                with col2:
                    next_day_forecast = forecast_result['predictions'][0]
                    change = next_day_forecast - forecast_result['last_observation']
                    change_pct = (change / forecast_result['last_observation']) * 100
                    st.metric(
                        "Next Day Forecast",
                        f"{next_day_forecast:.4f}",
                        delta=f"{change:+.4f} ({change_pct:+.2f}%)"
                    )
                
                with col3:
                    if len(forecast_result['predictions']) > 6:
                        week_forecast = forecast_result['predictions'][6]
                        week_change = week_forecast - forecast_result['last_observation']
                        week_change_pct = (week_change / forecast_result['last_observation']) * 100
                        st.metric(
                            "1 Week Forecast",
                            f"{week_forecast:.4f}",
                            delta=f"{week_change:+.4f} ({week_change_pct:+.2f}%)"
                        )
                    else:
                        st.metric("1 Week Forecast", "N/A")
                
                with col4:
                    final_forecast = forecast_result['predictions'][-1]
                    total_change = final_forecast - forecast_result['last_observation']
                    total_change_pct = (total_change / forecast_result['last_observation']) * 100
                    st.metric(
                        f"{forecast_days}-Day Forecast",
                        f"{final_forecast:.4f}",
                        delta=f"{total_change:+.4f} ({total_change_pct:+.2f}%)"
                    )
                
                # Create forecast chart
                st.subheader("📊 Forecast Visualization")
                
                # Generate dates for forecast
                from datetime import datetime, timedelta
                
                # Handle both string and datetime types for last_observation_date
                if isinstance(forecast_result['last_observation_date'], str):
                    last_date = datetime.strptime(forecast_result['last_observation_date'], '%Y-%m-%d')
                else:
                    # If it's already a datetime object, use it directly
                    last_date = forecast_result['last_observation_date']
                    if hasattr(last_date, 'date'):
                        last_date = last_date.date()
                    if not isinstance(last_date, datetime):
                        last_date = datetime.combine(last_date, datetime.min.time())
                
                # Generate forecast dates using Python datetime and convert to pandas
                forecast_dates = []
                for i in range(forecast_days):
                    forecast_date = last_date + timedelta(days=i+1)
                    forecast_dates.append(forecast_date)
                
                # Convert to pandas datetime
                forecast_dates = pd.to_datetime(forecast_dates)
                
                # Create DataFrame for plotting
                forecast_df = pd.DataFrame({
                    'date': pd.to_datetime(forecast_dates),  # Convert to pandas datetime
                    'predicted_rate': forecast_result['predictions'],
                    'forecast_type': 'Forecast'
                })
                
                # Add historical context (last 30 days)
                try:
                    historical_df = pd.read_csv("data/daily_forex_rates.csv", parse_dates=["date"])
                    hist_filtered = historical_df[
                        (historical_df["currency"] == selected_currency) & 
                        (historical_df["base_currency"] == "EUR")
                    ].sort_values("date").tail(30)
                    
                    if not hist_filtered.empty:
                        hist_df = pd.DataFrame({
                            'date': hist_filtered['date'],
                            'predicted_rate': hist_filtered['exchange_rate'],
                            'forecast_type': 'Historical'
                        })
                        
                        # Ensure both DataFrames have the same datetime type and timezone
                        hist_df['date'] = pd.to_datetime(hist_df['date']).dt.tz_localize(None)
                        forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.tz_localize(None)
                        
                        # Combine historical and forecast data
                        combined_df = pd.concat([hist_df, forecast_df], ignore_index=True)
                        
                        # Create interactive chart
                        fig = px.line(
                            combined_df,
                            x='date',
                            y='predicted_rate',
                            color='forecast_type',
                            title=f"{selected_currency}/EUR Exchange Rate Forecast",
                            labels={'predicted_rate': 'Exchange Rate', 'date': 'Date'},
                            color_discrete_map={'Historical': '#1f77b4', 'Forecast': '#ff7f0e'}
                        )
                        
                        # Add vertical line to separate historical from forecast
                        try:
                            # Convert to string format for Plotly to avoid Timestamp arithmetic issues
                            last_date_str = last_date.strftime('%Y-%m-%d')
                            fig.add_vline(
                                x=last_date_str,
                                line_dash="dash",
                                line_color="red",
                                annotation_text="Forecast Start"
                            )
                        except Exception:
                            # Skip vertical line if it fails
                            pass
                        
                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title=f"{selected_currency}/EUR Rate",
                            hovermode='x unified',
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Just show forecast if no historical data
                        fig = px.line(
                            forecast_df,
                            x='date',
                            y='predicted_rate',
                            title=f"{selected_currency}/EUR Exchange Rate Forecast",
                            labels={'predicted_rate': 'Exchange Rate', 'date': 'Date'}
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"Could not load historical data: {e}")
                    # Just show forecast
                    fig = px.line(
                        forecast_df,
                        x='date',
                        y='predicted_rate',
                        title=f"{selected_currency}/EUR Exchange Rate Forecast",
                        labels={'predicted_rate': 'Exchange Rate', 'date': 'Date'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed forecast table
                st.subheader("📋 Detailed Forecast Table")
                
                # Create detailed forecast table
                detailed_df = pd.DataFrame({
                    'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
                    'Day': [f"Day {i+1}" for i in range(forecast_days)],
                    'Predicted Rate': [f"{rate:.4f}" for rate in forecast_result['predictions']],
                    'Change from Current': [f"{rate - forecast_result['last_observation']:+.4f}" for rate in forecast_result['predictions']],
                    'Change %': [f"{(rate - forecast_result['last_observation']) / forecast_result['last_observation'] * 100:+.2f}%" for rate in forecast_result['predictions']]
                })
                
                st.dataframe(detailed_df, use_container_width=True)
                
                # Model confidence and limitations
                st.subheader("⚠️ Forecast Limitations")
                st.info("""
                **Important Notes:**
                - This forecast is based on historical patterns and may not account for unexpected events
                - Exchange rates are influenced by many factors including economic news, political events, and market sentiment
                - Past performance does not guarantee future results
                - Use this forecast as one input among many for decision-making
                """)
                
            except Exception as e:
                st.error(f"Forecasting failed: {str(e)}")
                st.info("Try retraining the model or check if the required data files exist.")
    
    except ArtefactMissingError as e:
        st.error("❌ Model artifacts not found")
        st.warning(str(e))
        
        if st.button("🚀 Train Model Now", type="primary"):
            with st.spinner("Training LSTM model... This may take a few minutes."):
                try:
                    from scripts.train_forex_model import train_and_export_model
                    train_and_export_model(currency=selected_currency)
                    st.success("✅ Model trained successfully! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model artifacts exist or retrain the model.")
        

# Footer
st.markdown("---")
st.markdown("Built with ❤️ for SMEs in exports and manufacturing")
