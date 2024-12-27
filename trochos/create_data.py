import argparse
import lib.stock_loader as stock_loader
import lib.tradier as tradier
import pandas as pd
from lib.errors import WheelError
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class DataCreator:
    def __init__(self, tickers, months, output):
        self.tickers = tickers
        self.months = months
        self.output = output + '.parquet'
        vix_data = tradier.get_historical_data("VIX", self.months)
        self.vix_data = vix_data['history'].get('days', [])

    def get_and_write(self):
        df_symbol_dict = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_item = {executor.submit(self.get_df_for_ticker, ticker): ticker for ticker in self.tickers}
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    (ticker, df), err = future.result()
                    if err:
                        print(f"get_df_for_ticker returned error: {err}")
                    else:
                        print(f"built data for {ticker}")
                        df_symbol_dict[ticker] = df
                except Exception as exc:
                    print(f'{item} generated an exception: {exc}')
                    return False

        combined_df = pd.concat(df_symbol_dict, names=['Symbol', 'Date'])
        combined_df.sort_index(inplace=True)
        # Writing to Parquet
        print(f"writing data to {self.output}!")
        combined_df.to_parquet(self.output, compression='snappy')
            
    def get_df_for_ticker(self, vix, ticker):
            history = tradier.get_historical_data(ticker, self.months)
            df_per_ticker, err = self.get_df_from_history(history)
            if err != None:
                return None, err.wrap(f"Could not construct pandas.DataFrame for {ticker}")
            return (ticker, df_per_ticker), None

    def get_df_from_history(self, history):
        # Parse the JSON data into a Pandas.DataFrame
        if 'history' in history and history['history'] is not None:
            days = history['history'].get('day', [])
        else:
            return None, WheelError("invalid history JSON data")
        
        print("Setting vix data..")
        days['vix'] = self.vix_data
        print("Vix data set..")
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
            return None, WheelError(f"could not construct DataFrame from days: {days}")

        return df, None


if __name__ == "__main__":
    # Handle command line flags
    parser = argparse.ArgumentParser(description="Trochos Tool")
    parser.add_argument("--prod", action='store_true', help="Whether to use a full set of stocks or a test suite of stocks.")
    parser.add_argument("--months", type=int, help="number of months to generate historical data for", required=True)
    parser.add_argument("--output", type=str, help="data output such as ml/example/historical_data_single", required=True)
    args = parser.parse_args()
    
    tickers = stock_loader.parse_stock_tickers(args.prod)
    creator = DataCreator(tickers, args.months, args.output)
    creator.get_and_write()