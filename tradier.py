import requests
from functools import partial
from wheel_enums import StrategyType

# Replace with your actual Tradier API token
API_KEY = "zLPdCfGiXbrv6SEbIxIrpXbbEqXH"

def get_stock_quote(symbol):
  """
  Fetches a real-time stock quote for a given stock symbol.

  Args:
    symbol: The stock ticker symbol (e.g., 'AAPL')

  Returns:
    A JSON dictionary.
  """
  response = requests.get('https://api.tradier.com/v1/markets/quotes',
    params={'symbols': symbol, 'greeks': 'false'},
    headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}
  )
  response.raise_for_status()
  data = response.json()
  return data['quotes']['quote']

def get_option_quotes(optionType, symbol, expiration):
  """
  Fetches real-time option quotes for a given stock symbol.

  Args:
    symbol: The stock ticker symbol (e.g., 'AAPL')

  Returns:
    A list of dictionaries, where each dictionary represents an option quote.
  """

  response = requests.get('https://api.tradier.com/v1/markets/options/chains',
      params={'symbol': symbol, 'expiration': f'{expiration}', 'greeks': 'true'},
      headers={'Authorization': f'Bearer {API_KEY}', 'Accept': 'application/json'}
  )
  response.raise_for_status()

  data = response.json()
  if len(data) == 0 or data['options'] == None:
    return []
  option_quotes = data['options']['option']

  filtered_quotes = []
  for quote in option_quotes:
    filtered_quote = {
        'symbol': quote['underlying'],
        'description': quote['description'],
        'strike': quote['strike'],
        'bid': quote['bid'],
        'ask': quote['ask'],
        'expiration_date': quote['expiration_date'],
        'option_type': quote['option_type'],
        'greeks': {
            'delta': quote['greeks']['delta'],
            'gamma': quote['greeks']['gamma'],
            'theta': quote['greeks']['theta'],
            'vega': quote['greeks']['vega'],
            'iv': quote['greeks']['mid_iv']  # Use mid_iv for implied volatility
        },
        'volume': quote['volume'],
        'open_interest': quote['open_interest'],
    }
    if filtered_quote['option_type'] == optionType:
      filtered_quotes.append(filtered_quote)
  return filtered_quotes

get_call_quotes = partial(get_option_quotes, StrategyType.CALL.value)
get_put_quotes = partial(get_option_quotes, StrategyType.PUT.value)