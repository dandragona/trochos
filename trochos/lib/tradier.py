import requests
from functools import partial
from wheel_enums import StrategyType
from lib.errors import WheelError
from datetime import datetime, timedelta
# Replace with your actual Tradier API token
API_KEY = "zLPdCfGiXbrv6SEbIxIrpXbbEqXH"

HEADERS = {
  'Authorization': f'Bearer {API_KEY}',
  'Accept': 'application/json'
} 

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
    headers=HEADERS
  )
  response.raise_for_status()
  data = response.json()
  if data['quotes'] == None or data['quotes']['quote'] == None:
    return None, WheelError(f"get_stock_quote fetched invalid quote:\n {data}\n")
  return data['quotes']['quote'], None


def get_historical_data(symbol, months):
  end_date = datetime.today().date()
  start_date = end_date - timedelta(days=months * 30)  # approx 2 years of data

  params = {
      'symbol': symbol,
      'interval': 'daily',
      'start': str(start_date),
      'end': str(end_date)
  }

  #------------------------------------------------------------
  # Step 3: Make the request to Tradier API
  #------------------------------------------------------------
  response = requests.get("https://api.tradier.com/v1/markets/history", headers=HEADERS, params=params)
  response.raise_for_status()  # raises an error if the request failed

  data = response.json()
  return data


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
      headers=HEADERS
  )
  response.raise_for_status()

  data = response.json()
  if len(data) == 0 or data['options'] == None:
    return [], None
  if data['options'] == None or data['options']['option'] == None:
    return [], WheelError(f"get_option_quotes returned invalid data:\n {data}\n")
  option_quotes = data['options']['option']
  filtered_quotes = []
  for quote in option_quotes:
    if quote['bid'] == None or quote['ask'] == None:
      continue
    if quote['greeks'] != None:
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
  return filtered_quotes, None

get_call_quotes = partial(get_option_quotes, StrategyType.CALL.value)
get_put_quotes = partial(get_option_quotes, StrategyType.PUT.value)