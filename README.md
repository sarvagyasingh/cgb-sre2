# Supplier Risk & Recommendation Engine

An AI-powered decision support tool for SMEs engaged in exports and manufacturing. It predicts raw material cost trends, scores supplier reliability, and recommends alternative sourcing options.

## Features

### 📈 Price Monitoring
- **Metals**: Real-time metal price tracking via metals.dev API
- **Forex**: Live exchange rates via Open Exchange Rates API
- Interactive historical price charts for both metals and currencies
- Multi-currency support (USD, EUR, GBP, JPY)
- Price volatility analysis
- Customizable metal and currency selection

### 📰 News Headlines
- **Shipping News**: Latest news about major shipping routes (Asia-Europe, Trans-Pacific, Suez Canal, Panama Canal)
- **Metals & Forex News**: Real-time news about metal prices and currency markets
- **Business News**: Supply chain and manufacturing industry updates
- Time-based filtering (1, 3, 7, 14, 30 days)
- Direct links to full articles
- Demo mode with sample news when API not configured

### 📊 Dashboard
- Supply chain health overview
- Key performance indicators
- Risk metrics and alerts

### 🏢 Supplier Analysis
- Supplier performance scoring
- Reliability analysis
- Cost-benefit evaluation

### ⚠️ Risk Alerts
- Supply chain disruption notifications
- Price volatility warnings
- Supplier risk alerts

### 💡 Recommendations
- AI-powered sourcing suggestions
- Alternative supplier recommendations
- Cost optimization strategies

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the project root:
```env
# Metals.dev API Configuration
# No quotes needed around the API key
METALS_API_KEY=your_api_key_here

# Open Exchange Rates API Configuration
OPENEXCHANGERATES_APP_ID=your_app_id_here

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Optional: API Configuration
METALS_BASE_URL=https://api.metals.dev/v1
OPENEXCHANGERATES_BASE_URL=https://openexchangerates.org/api
```

### 3. Get API Keys
**For Metals Data:**
1. Visit [metals.dev](https://metals.dev)
2. Sign up for an account
3. Get your API key
4. Add it to the `.env` file

**For Forex Data:**
1. Visit [Open Exchange Rates](https://openexchangerates.org)
2. Sign up for an account
3. Get your App ID
4. Add it to the `.env` file

**For News Data:**
1. Visit [NewsAPI](https://newsapi.org)
2. Sign up for an account
3. Get your API key
4. Add it to the `.env` file

### 4. Run the Application
```bash
streamlit run app.py
```

## Regenerating the Forex Forecast Artefacts

The forecasting utilities in `forex_forecast.py` load a persisted LSTM model
and scaler from `models/forex_lstm.keras` and `models/forex_scaler.pkl`. These
files are ignored by Git so every contributor should recreate them locally
before running features that rely on forex predictions.

1. Install the project dependencies (TensorFlow and scikit-learn are listed in
   `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

2. Run the deterministic training script. The example below rebuilds the
   default EUR→USD model and scaler:

   ```bash
   python scripts/train_forex_model.py --currency USD
   ```

   You can adjust the lookback window, number of epochs, or target currency via
   the CLI flags (see `python scripts/train_forex_model.py --help`).

3. The regenerated artefacts are written to the `models/` directory. Subsequent
   calls to `forecast_next_rate()` in `forex_forecast.py` will load them
   automatically. Passing `auto_train=True` will trigger the same export
   routine if the files are missing.

## Usage

### Price Monitoring
1. Navigate to the "Price Monitoring" page
2. Choose between "Metals" or "Forex" data type
3. Select metals or currencies to monitor from the sidebar
4. Choose your preferred base currency
5. Click "Generate Chart" to view historical data
6. Use the refresh button to get latest prices/rates

### News Headlines
1. Navigate to the "News Headlines" page
2. Select time period from the sidebar (1, 3, 7, 14, or 30 days)
3. Browse news in three categories:
   - **Shipping News**: Updates about major shipping routes
   - **Metals & Forex**: News about metal prices and currency markets
   - **Business News**: General supply chain and manufacturing updates
4. Click "Read More" to view full articles
5. Use the refresh button to get latest news

### Demo Mode
If no API key is configured, the app will run in demo mode with sample data.

## API Integration

The app integrates with:
- **metals.dev API**: Real-time metal prices and historical data
- **Open Exchange Rates API**: Live foreign exchange rates and historical data
- **NewsAPI**: Real-time news headlines for shipping, metals, forex, and business
- **Multiple currencies**: USD, EUR, GBP, JPY, CAD, AUD, CHF, CNY, INR, BRL, MXN support
- **Historical data**: Up to 1 year of price/rate history
- **Multiple metals**: Gold, silver, copper, platinum, palladium, aluminum, zinc, lead, tin, nickel
- **News categories**: Shipping routes, metals/forex markets, business/supply chain news

## File Structure

```
├── app.py                  # Main Streamlit application
├── forex_api.py            # Open Exchange Rates API client
├── forex_forecast.py       # Forex LSTM inference helpers
├── metals_api.py           # Metals.dev API client
├── news_api.py             # NewsAPI client
├── requirements.txt        # Python dependencies
├── scripts/
│   └── train_forex_model.py  # Deterministic training and export CLI
├── models/                 # Generated model artefacts (gitignored)
├── data/                   # Sample datasets
├── .env                    # Environment variables (API keys)
└── README.md               # This file
```

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Plotly
- Requests
- python-dotenv
- newsapi-python

## License

Built for SMEs in exports and manufacturing.
# cgb-sre2
