import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class ForexAPIClient:
    """
    Client for interacting with the Open Exchange Rates API to retrieve foreign exchange rates.
    """
    
    def __init__(self):
        # Force reload environment variables
        load_dotenv(override=True)
        
        self.app_id = os.getenv('OPENEXCHANGERATES_APP_ID')
        self.base_url = os.getenv('OPENEXCHANGERATES_BASE_URL', 'https://openexchangerates.org/api')
        self.session = requests.Session()
        
    
    def get_latest_rates(self, base_currency='USD', symbols=None):
        """
        Get latest exchange rates.
        
        Args:
            base_currency (str): Base currency (default: USD)
            symbols (list): List of currency symbols to retrieve (e.g., ['EUR', 'GBP', 'JPY'])
            
        Returns:
            dict: Exchange rates data
        """
        if not self.app_id or self.app_id == 'your_app_id_here':
            return self._get_demo_forex_data()
        
        try:
            url = f"{self.base_url}/latest.json"
            params = {
                'app_id': self.app_id,
                'base': base_currency
            }
            
            if symbols:
                params['symbols'] = ','.join(symbols)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            if 'rates' in data and data.get('base'):
                return {
                    'base': data['base'],
                    'timestamp': data.get('timestamp'),
                    'rates': data['rates'],
                    'disclaimer': data.get('disclaimer', ''),
                    'license': data.get('license', '')
                }
            else:
                return self._get_demo_forex_data()
            
        except requests.exceptions.RequestException as e:
            return self._get_demo_forex_data()
        except Exception as e:
            return self._get_demo_forex_data()
    
    def get_historical_rates(self, date, base_currency='USD', symbols=None):
        """
        Get historical exchange rates for a specific date.
        
        Args:
            date (str): Date in YYYY-MM-DD format
            base_currency (str): Base currency (default: USD)
            symbols (list): List of currency symbols to retrieve
            
        Returns:
            dict: Historical exchange rates data
        """
        if not self.app_id or self.app_id == 'your_app_id_here':
            return self._get_demo_historical_forex_data(date)
        
        try:
            url = f"{self.base_url}/historical/{date}.json"
            params = {
                'app_id': self.app_id,
                'base': base_currency
            }
            
            if symbols:
                params['symbols'] = ','.join(symbols)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            if 'rates' in data and data.get('base'):
                return {
                    'base': data['base'],
                    'date': date,
                    'timestamp': data.get('timestamp'),
                    'rates': data['rates'],
                    'disclaimer': data.get('disclaimer', ''),
                    'license': data.get('license', '')
                }
            else:
                return self._get_demo_historical_forex_data(date)
            
        except requests.exceptions.RequestException as e:
            return self._get_demo_historical_forex_data(date)
        except Exception as e:
            return self._get_demo_historical_forex_data(date)
    
    def get_time_series(self, start_date, end_date, base_currency='USD', symbols=None):
        """
        Get time series data for exchange rates.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            base_currency (str): Base currency (default: USD)
            symbols (list): List of currency symbols to retrieve
            
        Returns:
            dict: Time series data
        """
        if not self.app_id or self.app_id == 'your_app_id_here':
            return self._get_demo_time_series_data(start_date, end_date)
        
        try:
            url = f"{self.base_url}/time-series.json"
            params = {
                'app_id': self.app_id,
                'start': start_date,
                'end': end_date,
                'base': base_currency
            }
            
            if symbols:
                params['symbols'] = ','.join(symbols)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Format the response
            if 'rates' in data:
                return {
                    'base': data.get('base', base_currency),
                    'start_date': start_date,
                    'end_date': end_date,
                    'rates': data['rates'],
                    'disclaimer': data.get('disclaimer', ''),
                    'license': data.get('license', '')
                }
            else:
                return self._get_demo_time_series_data(start_date, end_date)
            
        except requests.exceptions.RequestException as e:
            return self._get_demo_time_series_data(start_date, end_date)
        except Exception as e:
            return self._get_demo_time_series_data(start_date, end_date)
    
    def convert_currency(self, amount, from_currency, to_currency, date=None):
        """
        Convert currency amount from one currency to another.
        
        Args:
            amount (float): Amount to convert
            from_currency (str): Source currency
            to_currency (str): Target currency
            date (str): Date for historical conversion (optional)
            
        Returns:
            dict: Conversion result
        """
        if not self.app_id or self.app_id == 'your_app_id_here':
            return self._get_demo_conversion_data(amount, from_currency, to_currency)
        
        try:
            url = f"{self.base_url}/convert/{amount}/{from_currency}/{to_currency}"
            params = {
                'app_id': self.app_id
            }
            
            if date:
                params['date'] = date
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'amount': amount,
                'from_currency': from_currency,
                'to_currency': to_currency,
                'result': data.get('result'),
                'rate': data.get('rate'),
                'date': data.get('date', date)
            }
            
        except requests.exceptions.RequestException as e:
            return self._get_demo_conversion_data(amount, from_currency, to_currency)
        except Exception as e:
            return self._get_demo_conversion_data(amount, from_currency, to_currency)
    
    def get_currencies(self):
        """
        Get list of supported currencies.
        
        Returns:
            dict: Currencies data
        """
        if not self.app_id or self.app_id == 'your_app_id_here':
            return self._get_demo_currencies_data()
        
        try:
            url = f"{self.base_url}/currencies.json"
            params = {
                'app_id': self.app_id
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'currencies': data,
                'count': len(data)
            }
            
        except requests.exceptions.RequestException as e:
            return self._get_demo_currencies_data()
        except Exception as e:
            return self._get_demo_currencies_data()
    
    def _get_demo_forex_data(self):
        """
        Return demo forex data when App ID is not configured.
        """
        return {
            'base': 'USD',
            'timestamp': int(time.time()),
            'rates': {
                'EUR': 0.85,
                'GBP': 0.73,
                'JPY': 110.0,
                'CAD': 1.25,
                'AUD': 1.35,
                'CHF': 0.92,
                'CNY': 6.45,
                'INR': 74.0,
                'BRL': 5.2,
                'MXN': 20.0
            },
            'disclaimer': 'Demo data - configure API key for real rates',
            'license': 'Demo license'
        }
    
    def _get_demo_historical_forex_data(self, date):
        """
        Return demo historical forex data when App ID is not configured.
        """
        return {
            'base': 'USD',
            'date': date,
            'timestamp': int(time.time()),
            'rates': {
                'EUR': 0.85,
                'GBP': 0.73,
                'JPY': 110.0,
                'CAD': 1.25,
                'AUD': 1.35,
                'CHF': 0.92,
                'CNY': 6.45,
                'INR': 74.0,
                'BRL': 5.2,
                'MXN': 20.0
            },
            'disclaimer': 'Demo data - configure API key for real rates',
            'license': 'Demo license'
        }
    
    def _get_demo_time_series_data(self, start_date, end_date):
        """
        Return demo time series data when App ID is not configured.
        """
        # Generate demo time series data
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        rates = {}
        current_date = start
        
        while current_date <= end:
            date_str = current_date.strftime('%Y-%m-%d')
            rates[date_str] = {
                'EUR': 0.85 + (current_date.day % 10) * 0.01,
                'GBP': 0.73 + (current_date.day % 7) * 0.01,
                'JPY': 110.0 + (current_date.day % 5) * 0.5
            }
            current_date += timedelta(days=1)
        
        return {
            'base': 'USD',
            'start_date': start_date,
            'end_date': end_date,
            'rates': rates,
            'disclaimer': 'Demo data - configure API key for real rates',
            'license': 'Demo license'
        }
    
    def _get_demo_conversion_data(self, amount, from_currency, to_currency):
        """
        Return demo conversion data when App ID is not configured.
        """
        # Demo conversion rates
        demo_rates = {
            'USD': 1.0,
            'EUR': 0.85,
            'GBP': 0.73,
            'JPY': 110.0,
            'CAD': 1.25,
            'AUD': 1.35
        }
        
        from_rate = demo_rates.get(from_currency, 1.0)
        to_rate = demo_rates.get(to_currency, 1.0)
        
        result = (amount / from_rate) * to_rate
        rate = to_rate / from_rate
        
        return {
            'amount': amount,
            'from_currency': from_currency,
            'to_currency': to_currency,
            'result': round(result, 4),
            'rate': round(rate, 6),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
    
    def _get_demo_currencies_data(self):
        """
        Return demo currencies data when App ID is not configured.
        """
        return {
            'currencies': {
                'USD': 'United States Dollar',
                'EUR': 'Euro',
                'GBP': 'British Pound Sterling',
                'JPY': 'Japanese Yen',
                'CAD': 'Canadian Dollar',
                'AUD': 'Australian Dollar',
                'CHF': 'Swiss Franc',
                'CNY': 'Chinese Yuan',
                'INR': 'Indian Rupee',
                'BRL': 'Brazilian Real',
                'MXN': 'Mexican Peso'
            },
            'count': 11
        }

def format_forex_data(rates_data):
    """
    Format forex data for display in Streamlit.
    
    Args:
        rates_data (dict): Raw forex data from API
        
    Returns:
        pd.DataFrame: Formatted forex data
    """
    if not rates_data or 'rates' not in rates_data:
        return pd.DataFrame()
    
    formatted_data = []
    
    for currency, rate in rates_data['rates'].items():
        formatted_data.append({
            'Currency': currency,
            'Rate': f"{rate:.4f}",
            'Base': rates_data.get('base', 'USD'),
            'Last Updated': datetime.fromtimestamp(rates_data.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')
        })
    
    return pd.DataFrame(formatted_data)

def format_historical_forex_data(historical_data):
    """
    Format historical forex data for plotting.
    
    Args:
        historical_data (dict): Raw historical data from API
        
    Returns:
        pd.DataFrame: Formatted historical data
    """
    if not historical_data or 'rates' not in historical_data:
        return pd.DataFrame()
    
    # For time series data
    if isinstance(historical_data['rates'], dict) and any(isinstance(v, dict) for v in historical_data['rates'].values()):
        # Time series format
        data = []
        for date, rates in historical_data['rates'].items():
            for currency, rate in rates.items():
                data.append({
                    'date': pd.to_datetime(date),
                    'currency': currency,
                    'rate': rate
                })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['currency', 'date'])
        
        return df
    else:
        # Single date format
        data = []
        for currency, rate in historical_data['rates'].items():
            data.append({
                'date': pd.to_datetime(historical_data.get('date', datetime.now().strftime('%Y-%m-%d'))),
                'currency': currency,
                'rate': rate
            })
        
        return pd.DataFrame(data)
