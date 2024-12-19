import requests
import pandas as pd
from datetime import datetime, timedelta

#------------------------------------------------------------
# Step 1: Set up your Tradier API credentials and endpoint
#------------------------------------------------------------
API_KEY = "zLPdCfGiXbrv6SEbIxIrpXbbEqXH"
BASE_URL = "https://api.tradier.com/v1/markets/history"  # Production endpoint

def main():
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Accept': 'application/json'
    }

    #------------------------------------------------------------
    # Step 2: Define the parameters for data retrieval
    #
    # Choose a symbol, a start date, and end date for your historical query.
    # For example, let's get the last 2 years of daily data for "AAPL".
    #------------------------------------------------------------
    symbol = "AAPL"
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=2*365)  # approx 2 years of data

    params = {
        'symbol': symbol,
        'interval': 'daily',
        'start': str(start_date),
        'end': str(end_date)
    }

    #------------------------------------------------------------
    # Step 3: Make the request to Tradier API
    #------------------------------------------------------------
    response = requests.get(BASE_URL, headers=headers, params=params)
    response.raise_for_status()  # raises an error if the request failed

    data = response.json()

    #------------------------------------------------------------
    # Step 4: Parse the JSON response into a Pandas DataFrame
    #
    # The JSON structure for Tradier historical data looks like:
    # {
    #   "history": {
    #     "day": [
    #       { "date":"2021-01-04","open":"133.52","high":"133.61","low":"126.76","close":"129.41","volume":"143301887" },
    #       ...
    #     ]
    #   }
    # }
    #
    # We need to extract the 'day' array and convert it to a DataFrame
    #------------------------------------------------------------
    if 'history' in data and data['history'] is not None:
        days = data['history'].get('day', [])
    else:
        days = []

    df = pd.DataFrame(days)
    if not df.empty:
        # Convert columns to proper data types
        df['date'] = pd.to_datetime(df['date'])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

        # Set the date as index
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
    else:
        raise ValueError("No data returned for the given symbol and date range.")

    print(df.head())

    #------------------------------------------------------------
    # Step 5: Now you have a DataFrame similar to:
    #
    #              open    high     low   close     volume
    # date
    # 2022-01-03  182.63  182.94  177.71  182.01  104487000
    # 2022-01-04  182.63  182.94  179.12  179.70  116857900
    # ...
    #
    # This is a structure that you can then pass on to the feature engineering
    # and modeling steps shown in the previous code.
    #------------------------------------------------------------


if __name__ == "__main__":
    main()