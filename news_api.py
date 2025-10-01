import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    from newsapi import NewsApiClient
    NEWSAPI_AVAILABLE = True
except ImportError:
    NEWSAPI_AVAILABLE = False
    NewsApiClient = None

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    load_dotenv = None

COUNTRY_KEYWORDS = [
    "supply chain",
    "logistics",
    "manufacturing",
    "factory",
    "port",
    "strike",
    "protest",
    "inflation",
    "tariff",
    "shutdown",
    "disruption",
]

COUNTRY_TO_ISO = {
    "united states": "us",
    "usa": "us",
    "canada": "ca",
    "germany": "de",
    "brazil": "br",
    "china": "cn",
    "south africa": "za",
    "india": "in",
    "australia": "au",
    "mexico": "mx",
    "russia": "ru",
    "united kingdom": "gb",
}


class NewsAPIClient:
    """
    Client for fetching news headlines using NewsAPI
    """
    
    def __init__(self):
        """Initialize the NewsAPI client with API key from environment variables"""
        if not NEWSAPI_AVAILABLE:
            raise ImportError("newsapi-python package not installed. Please install it with: pip install newsapi-python")
        
        if DOTENV_AVAILABLE:
            load_dotenv()
        
        self.api_key = os.getenv('NEWS_API_KEY')
        
        if not self.api_key:
            raise ValueError("NEWS_API_KEY not found in environment variables")
        
        self.client = NewsApiClient(api_key=self.api_key)
    
    def get_shipping_news(self, routes: List[str], days_back: int = 7) -> List[Dict]:
        """
        Fetch news headlines related to shipping routes
        
        Args:
            routes: List of shipping routes to search for
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles related to shipping
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Create search query for shipping routes
            shipping_queries = []
            for route in routes:
                shipping_queries.extend([
                    f'"{route}" shipping',
                    f'"{route}" logistics',
                    f'"{route}" freight',
                    f'"{route}" cargo',
                    f'"{route}" trade route'
                ])
            
            all_articles = []
            
            # Search for each shipping route
            for route in routes:
                try:
                    # Search for shipping-related news
                    articles = self.client.get_everything(
                        q=f'"{route}" shipping logistics freight cargo',
                        from_param=from_date.strftime('%Y-%m-%d'),
                        to=to_date.strftime('%Y-%m-%d'),
                        language='en',
                        sort_by='relevancy',
                        page_size=5
                    )
                    
                    if articles and 'articles' in articles:
                        for article in articles['articles']:
                            if article and article.get('title'):
                                article['route'] = route
                                article['category'] = 'Shipping'
                                all_articles.append(article)
                
                except Exception as e:
                    print(f"Error fetching news for route {route}: {str(e)}")
                    continue
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_articles = []
            for article in all_articles:
                if article.get('title') not in seen_titles:
                    seen_titles.add(article.get('title'))
                    unique_articles.append(article)
            
            return unique_articles[:10]  # Return top 10 articles
            
        except Exception as e:
            print(f"Error fetching shipping news: {str(e)}")
            return []
    
    def get_metals_forex_news(self, metals: List[str], currencies: List[str], days_back: int = 7) -> List[Dict]:
        """
        Fetch news headlines related to metals and forex prices
        
        Args:
            metals: List of metals to search for
            currencies: List of currencies to search for
            days_back: Number of days to look back for news
            
        Returns:
            List of news articles related to metals and forex
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            all_articles = []
            
            # Search for metals news
            metals_query = ' OR '.join([f'"{metal}" price' for metal in metals])
            metals_query += ' OR metals commodity trading'
            
            try:
                metals_articles = self.client.get_everything(
                    q=metals_query,
                    from_param=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=8
                )
                
                if metals_articles and 'articles' in metals_articles:
                    for article in metals_articles['articles']:
                        if article and article.get('title'):
                            article['category'] = 'Metals'
                            all_articles.append(article)
            
            except Exception as e:
                print(f"Error fetching metals news: {str(e)}")
            
            # Search for forex news
            forex_query = ' OR '.join([f'"{currency}" exchange rate' for currency in currencies])
            forex_query += ' OR forex currency trading'
            
            try:
                forex_articles = self.client.get_everything(
                    q=forex_query,
                    from_param=from_date.strftime('%Y-%m-%d'),
                    to=to_date.strftime('%Y-%m-%d'),
                    language='en',
                    sort_by='relevancy',
                    page_size=8
                )
                
                if forex_articles and 'articles' in forex_articles:
                    for article in forex_articles['articles']:
                        if article and article.get('title'):
                            article['category'] = 'Forex'
                            all_articles.append(article)
            
            except Exception as e:
                print(f"Error fetching forex news: {str(e)}")
            
            # Remove duplicates based on title
            seen_titles = set()
            unique_articles = []
            for article in all_articles:
                if article.get('title') not in seen_titles:
                    seen_titles.add(article.get('title'))
                    unique_articles.append(article)
            
            return unique_articles[:15]  # Return top 15 articles
            
        except Exception as e:
            print(f"Error fetching metals/forex news: {str(e)}")
            return []
    
    def get_business_headlines(self, days_back: int = 7) -> List[Dict]:
        """
        Fetch general business headlines
        
        Args:
            days_back: Number of days to look back for news
            
        Returns:
            List of business news articles
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            articles = self.client.get_everything(
                q='business economy trade manufacturing supply chain',
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            if articles and 'articles' in articles:
                for article in articles['articles']:
                    if article:
                        article['category'] = 'Business'
                return articles['articles']

            return []

        except Exception as e:
            print(f"Error fetching business headlines: {str(e)}")
            return []

    def get_country_supply_chain_news(self, country: str, days_back: int = 7) -> List[Dict]:
        """Fetch supply-chain related news for a specific country."""

        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            iso_code = COUNTRY_TO_ISO.get(country.lower())
            keywords = " OR ".join(COUNTRY_KEYWORDS)
            search_terms = [f'"{country}" ({keywords})']

            if iso_code:
                search_terms.append(f'"{iso_code.upper()}" ({keywords})')

            query = " OR ".join(search_terms)
            response = self.client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=10,
            )

            articles = response.get('articles', []) if response else []
            for article in articles:
                article['category'] = 'Country'
                article['country'] = country
            return articles
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"Error fetching country news for {country}: {exc}")
            return []

    def get_supply_chain_news_for_countries(
        self, countries: List[str], days_back: int = 7
    ) -> Dict[str, List[Dict]]:
        """Fetch supply-chain news for a list of countries."""

        results: Dict[str, List[Dict]] = {}
        for country in countries:
            results[country] = self.get_country_supply_chain_news(country, days_back)
        return results
    
    def format_news_data(self, articles: List[Dict]) -> List[Dict]:
        """
        Format news articles for display
        
        Args:
            articles: List of raw news articles
            
        Returns:
            List of formatted news articles
        """
        formatted_articles = []
        
        for article in articles:
            if not article or not article.get('title'):
                continue
                
            formatted_article = {
                'title': article.get('title', 'No title'),
                'description': article.get('description', 'No description available'),
                'url': article.get('url', '#'),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', 'Unknown source'),
                'category': article.get('category', 'General'),
                'route': article.get('route', '')  # For shipping news
            }
            
            # Format published date
            if formatted_article['published_at']:
                try:
                    pub_date = datetime.fromisoformat(formatted_article['published_at'].replace('Z', '+00:00'))
                    formatted_article['published_at'] = pub_date.strftime('%Y-%m-%d %H:%M')
                except:
                    formatted_article['published_at'] = 'Unknown date'
            
            formatted_articles.append(formatted_article)
        
        return formatted_articles

def get_demo_news_data() -> Dict:
    """
    Return demo news data when API is not available
    """
    return {
        'shipping_news': [
            {
                'title': 'Major Shipping Route Disruption: Suez Canal Congestion',
                'description': 'Heavy congestion reported in the Suez Canal affecting Asia-Europe shipping routes, with delays of up to 48 hours.',
                'url': '#',
                'published_at': '2024-01-15 10:30',
                'source': 'Maritime News',
                'category': 'Shipping',
                'route': 'Asia-Europe'
            },
            {
                'title': 'Trans-Pacific Shipping Rates Rise 15%',
                'description': 'Container shipping rates on major trans-Pacific routes have increased due to strong demand and capacity constraints.',
                'url': '#',
                'published_at': '2024-01-14 14:20',
                'source': 'Shipping Weekly',
                'category': 'Shipping',
                'route': 'Trans-Pacific'
            }
        ],
        'metals_forex_news': [
            {
                'title': 'Gold Prices Surge Amid Economic Uncertainty',
                'description': 'Gold prices reached a new high as investors seek safe-haven assets amid global economic concerns.',
                'url': '#',
                'published_at': '2024-01-15 09:15',
                'source': 'Financial Times',
                'category': 'Metals'
            },
            {
                'title': 'Copper Prices Fall on China Demand Concerns',
                'description': 'Copper futures declined as concerns about Chinese economic growth weigh on industrial metal demand.',
                'url': '#',
                'published_at': '2024-01-14 16:45',
                'source': 'Reuters',
                'category': 'Metals'
            },
            {
                'title': 'USD/EUR Exchange Rate Hits 3-Month High',
                'description': 'The US dollar strengthened against the euro as Federal Reserve signals continued monetary tightening.',
                'url': '#',
                'published_at': '2024-01-15 11:30',
                'source': 'Bloomberg',
                'category': 'Forex'
            }
        ],
        'business_news': [
            {
                'title': 'Global Supply Chain Resilience in Focus',
                'description': 'Companies are investing heavily in supply chain diversification and resilience measures following recent disruptions.',
                'url': '#',
                'published_at': '2024-01-15 08:00',
                'source': 'Wall Street Journal',
                'category': 'Business'
            }
        ]
    }


def generate_demo_country_news(countries: List[str]) -> Dict[str, List[Dict]]:
    """Generate structured demo country news items for offline usage."""

    template_headlines = {
        'High': [
            'Nationwide port workers strike threatens exports',
            'Government announces new tariffs impacting metal imports',
        ],
        'Medium': [
            'Currency volatility raises import costs for manufacturers',
            'Logistics delays reported amid severe weather warnings',
        ],
        'Low': [
            'Infrastructure investment aims to boost supply chain resilience',
            'Manufacturing sector reports stable output amid reforms',
        ],
    }

    demo_news: Dict[str, List[Dict]] = {}
    for idx, country in enumerate(countries):
        if idx % 3 == 0:
            severity = 'High'
        elif idx % 3 == 1:
            severity = 'Medium'
        else:
            severity = 'Low'

        entries: List[Dict] = []
        for headline in template_headlines[severity]:
            entries.append(
                {
                    'title': f'{country}: {headline}',
                    'description': headline,
                    'country': country,
                    'category': 'Country',
                    'publishedAt': '2024-01-15T00:00:00Z',
                    'severity': severity,
                    'source': 'Demo News',
                }
            )
        demo_news[country] = entries

    return demo_news
