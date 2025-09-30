import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class MetalsAPIClient:
    """
    Client for interacting with the metals.dev API to retrieve metal prices and data.
    """
    
    def __init__(self):
        # Force reload environment variables
        load_dotenv(override=True)
        
        self.api_key = os.getenv('METALS_API_KEY')
        self.base_url = os.getenv('METALS_BASE_URL', 'https://api.metals.dev/v1')
        self.session = requests.Session()
        
        # Debug information
        st.write(f"🔍 Debug: API key loaded: {self.api_key[:10] if self.api_key else 'None'}...")
        st.write(f"🔍 Debug: API key length: {len(self.api_key) if self.api_key else 0}")
        
        if self.api_key and self.api_key != 'your_api_key_here':
            self.session.headers.update({
                'Accept': 'application/json'
            })
            st.success("✅ API key loaded successfully!")
        else:
            st.warning("⚠️ API key not configured. Please set METALS_API_KEY in your .env file")
    
    def get_metal_prices(self, metals=None, currency='USD'):
        """
        Get current metal prices.
        
        Args:
            metals (list): List of metal symbols (e.g., ['gold', 'silver', 'copper'])
            currency (str): Currency for prices (default: USD)
            
        Returns:
            dict: Metal prices data
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            return self._get_demo_data()
        
        try:
            # Default metals if none specified
            if metals is None:
                metals = ['gold', 'silver', 'copper', 'platinum', 'palladium', 'aluminum', 'zinc', 'lead', 'tin', 'nickel']
            
            # Convert to comma-separated string
            metals_str = ','.join(metals)
            
            url = f"{self.base_url}/latest"
            params = {
                'api_key': self.api_key,
                'currency': currency,
                'unit': 'toz'  # troy ounce for precious metals
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response to match our expected structure
            formatted_data = {}
            if 'metals' in data and data['status'] == 'success':
                metals_dict = data['metals']
                for metal_symbol in metals:
                    if metal_symbol.lower() in metals_dict:
                        price = metals_dict[metal_symbol.lower()]
                        formatted_data[metal_symbol.lower()] = {
                            'price': price,
                            'currency': currency,
                            'unit': 'oz' if metal_symbol.lower() in ['gold', 'silver', 'platinum', 'palladium'] else 'lb'
                        }
            
            return formatted_data
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching metal prices: {str(e)}")
            return self._get_demo_data()
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return self._get_demo_data()
    
    def get_historical_prices(self, metal, currency='USD', start_date=None, end_date=None):
        """
        Get historical metal prices.
        
        Args:
            metal (str): Metal symbol (e.g., 'gold', 'silver')
            currency (str): Currency for prices (default: USD)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Historical prices data
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            return self._get_demo_historical_data(metal)
        
        try:
            # Default to last 30 days if no dates provided
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/timeseries"
            params = {
                'api_key': self.api_key,
                'symbols': metal.upper(),
                'currency': currency,
                'start_date': start_date,
                'end_date': end_date,
                'unit': 'toz' if metal.lower() in ['gold', 'silver', 'platinum', 'palladium'] else 'lb'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response to match our expected structure
            if 'data' in data and data['data']:
                prices = []
                for item in data['data']:
                    prices.append({
                        'date': item['date'],
                        'price': item['price'],
                        'currency': currency
                    })
                
                return {
                    'metal': metal,
                    'currency': currency,
                    'prices': prices
                }
            else:
                return self._get_demo_historical_data(metal)
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching historical prices for {metal}: {str(e)}")
            return self._get_demo_historical_data(metal)
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return self._get_demo_historical_data(metal)
    
    def get_metal_info(self, metal):
        """
        Get information about a specific metal.
        
        Args:
            metal (str): Metal symbol
            
        Returns:
            dict: Metal information
        """
        if not self.api_key or self.api_key == 'your_api_key_here':
            return self._get_demo_metal_info(metal)
        
        try:
            url = f"{self.base_url}/metals/{metal}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching metal info for {metal}: {str(e)}")
            return self._get_demo_metal_info(metal)
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return self._get_demo_metal_info(metal)
    
    def _get_demo_data(self):
        """
        Return demo data when API key is not configured.
        """
        return {
            'gold': {'price': 1950.50, 'currency': 'USD', 'unit': 'oz'},
            'silver': {'price': 24.75, 'currency': 'USD', 'unit': 'oz'},
            'copper': {'price': 3.85, 'currency': 'USD', 'unit': 'lb'},
            'platinum': {'price': 950.25, 'currency': 'USD', 'unit': 'oz'},
            'palladium': {'price': 1050.75, 'currency': 'USD', 'unit': 'oz'},
            'aluminum': {'price': 1.15, 'currency': 'USD', 'unit': 'lb'},
            'zinc': {'price': 1.25, 'currency': 'USD', 'unit': 'lb'},
            'lead': {'price': 0.95, 'currency': 'USD', 'unit': 'lb'},
            'tin': {'price': 12.50, 'currency': 'USD', 'unit': 'lb'},
            'nickel': {'price': 8.75, 'currency': 'USD', 'unit': 'lb'}
        }
    
    def _get_demo_historical_data(self, metal):
        """
        Return demo historical data when API key is not configured.
        """
        # Generate demo historical data for the last 30 days
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        # Base prices for different metals
        base_prices = {
            'gold': 1950,
            'silver': 24,
            'copper': 3.8,
            'platinum': 950,
            'palladium': 1050,
            'aluminum': 1.15,
            'zinc': 1.25,
            'lead': 0.95,
            'tin': 12.5,
            'nickel': 8.75
        }
        
        base_price = base_prices.get(metal, 100)
        
        # Generate realistic price movements
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # Add some random variation
            variation = (i % 7 - 3) * 0.02 + (i % 3 - 1) * 0.01
            current_price = current_price * (1 + variation)
            prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'price': round(current_price, 2),
                'currency': 'USD'
            })
        
        return {
            'metal': metal,
            'currency': 'USD',
            'prices': prices
        }
    
    def _get_demo_metal_info(self, metal):
        """
        Return demo metal information when API key is not configured.
        """
        metal_info = {
            'gold': {
                'name': 'Gold',
                'symbol': 'AU',
                'description': 'Precious metal used in jewelry and investment',
                'unit': 'oz',
                'category': 'precious_metals'
            },
            'silver': {
                'name': 'Silver',
                'symbol': 'AG',
                'description': 'Precious metal with industrial applications',
                'unit': 'oz',
                'category': 'precious_metals'
            },
            'copper': {
                'name': 'Copper',
                'symbol': 'CU',
                'description': 'Industrial metal used in electrical applications',
                'unit': 'lb',
                'category': 'base_metals'
            }
        }
        
        return metal_info.get(metal, {
            'name': metal.title(),
            'symbol': metal.upper(),
            'description': f'Metal commodity: {metal}',
            'unit': 'lb',
            'category': 'base_metals'
        })

def format_price_data(prices_data):
    """
    Format price data for display in Streamlit.
    
    Args:
        prices_data (dict): Raw price data from API
        
    Returns:
        pd.DataFrame: Formatted price data
    """
    if not prices_data:
        return pd.DataFrame()
    
    formatted_data = []
    
    for metal, data in prices_data.items():
        if isinstance(data, dict) and 'price' in data:
            formatted_data.append({
                'Metal': metal.title(),
                'Price': f"${data['price']:,.2f}",
                'Currency': data.get('currency', 'USD'),
                'Unit': data.get('unit', 'oz'),
                'Last Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return pd.DataFrame(formatted_data)

def format_historical_data(historical_data):
    """
    Format historical price data for plotting.
    
    Args:
        historical_data (dict): Raw historical data from API
        
    Returns:
        pd.DataFrame: Formatted historical data
    """
    if not historical_data or 'prices' not in historical_data:
        return pd.DataFrame()
    
    prices = historical_data['prices']
    df = pd.DataFrame(prices)
    
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['price'] = pd.to_numeric(df['price'])
    
    return df
