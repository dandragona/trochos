import argparse
from utils import Util
from options import ParsedCCSpread, ParsedPutOption
from wheel_enums import StrategyType
import tradier
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq

PROD_STOCK_FILE = "stocks_prod.txt"
TEST_STOCK_FILE = "stocks_test.txt"
EXPIRATION_DATE = "2024-12-06"
      

class Wheel:
   def __init__(self, args):
      # Access parsed arguments
      self.spreadLength = args.strike_spread
      self.strategyType = StrategyType.CC_SPREAD if self.spreadLength != None else StrategyType.PUT
      self.processorMap = {
         StrategyType.PUT: self.buildFilteredPutData,
         StrategyType.PC_SPREAD: self.buildFilteredPCSpreadData,
         StrategyType.CC_SPREAD: self.buildFilteredCCSpreadData
      }
      self.targetRatio = args.target_ratio
      self.delta = args.delta
      self.topN = args.top_n
      self.tickers = parse_stock_tickers(PROD_STOCK_FILE if args.prod_env else TEST_STOCK_FILE)
      self.expiration = datetime.strptime(EXPIRATION_DATE, "%Y-%m-%d").date()
      self.utils = Util(self.targetRatio, self.delta, self.expiration)
      self.symbolsToCurrentPrice = {}
      self.runResults = []
      # TODO: validate flag inputs

   def __str__(self):
      return f"Generating best {self.strategyType.value} picks that achieve a {self.targetRatio}% monthly return on capital across the tickers: {self.tickers}"

   def updateSymbolsToCurrentPrice(self, ticker):
      stockQuote = tradier.get_stock_quote(ticker)
      self.symbolsToCurrentPrice[stockQuote['symbol']] = stockQuote['last']
      return stockQuote['symbol']
   
   def buildFilteredPutData(self, ticker):
    symbol = self.updateSymbolsToCurrentPrice(ticker)
    optionData = tradier.get_put_quotes(symbol, self.expiration)
    optionData = self.utils.FilterPutsForTargetRatio(optionData, self.symbolsToCurrentPrice[symbol])
    aggregatePuts = []
    for option in optionData:
        heapq.heappush(aggregatePuts, ParsedPutOption(symbol, ticker, self.symbolsToCurrentPrice[symbol], option))
    return (ticker, [heapq.heappop(aggregatePuts) for _ in range(min(self.topN, len(aggregatePuts)))])
   
   def buildFilteredPCSpreadData(self, ticker):
      print("PCSpread Data Not Implemented")
      return (ticker, [])
    
   def buildFilteredCCSpreadData(self, ticker):
      symbol = self.updateSymbolsToCurrentPrice(ticker)
      optionData = tradier.get_call_quotes(symbol, self.expiration)
      rawSpreads = self.utils.BuildVerticalCallSpreads(optionData, self.spreadLength, symbol, ticker, self.symbolsToCurrentPrice[symbol])
      filteredSpreadData = self.utils.FilterVerticalCallSpreadsForTargetRatio(rawSpreads)
      aggregateSpreads = []
      for spread in filteredSpreadData:
         heapq.heappush(aggregateSpreads, ParsedCCSpread(symbol, ticker, self.symbolsToCurrentPrice[symbol], spread[2], spread[3]))
      return (ticker, [heapq.heappop(aggregateSpreads) for _ in range(min(self.topN, len(aggregateSpreads)))])


   def Run(self):
    resultsMap = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
       future_to_item = {executor.submit(self.processorMap[self.strategyType], ticker): ticker for ticker in self.tickers}
       for future in as_completed(future_to_item):
          item = future_to_item[future]
          try:
             data = future.result()
             resultsMap[data[0]] = data[1]
          except Exception as exc:
             print(f'{item} generated an exception: {exc}')
             return False
    self.resultsMap = resultsMap
    return True
   
   def PrintResults(self):
      if len(self.resultsMap) == 0:
         print("Wheel.Run() has not been called successfully")
         return
      for (ticker, options) in self.resultsMap.items():
         if self.strategyType == StrategyType.PUT:
            self.utils.PrintPuts(options, ticker, self.symbolsToCurrentPrice[ticker], self.topN)
         elif self.strategyType == StrategyType.CC_SPREAD:
            self.utils.PrintSpreads(options, ticker, self.symbolsToCurrentPrice[ticker], self.topN)
    


def parse_stock_tickers(filename="stocks_test.txt"):
  """
  Parses a text file where each line contains a stock ticker.

  Args:
    filename: The name of the text file (default: "stock_tickers.txt").

  Returns:
    A list of stock tickers.
  """
  print("Parsing: ", filename)
  try:
    with open(filename, "r") as file:
      tickers = [line.strip() for line in file]
      return tickers
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found in the current directory.")
    return []
  

def main():
    """
    Parses user flags and launches the application.
    """
    parser = argparse.ArgumentParser(description="My Python Application")
    parser.add_argument("--strike_spread", type=int, help="User capital to play with.")
    parser.add_argument("--target_ratio", type=float, help="Ratio of premium to capital that we want to hit.", required=True)
    parser.add_argument("--top_n", type=int, help="Returns to the top n contracts from the pool of useable contracts.")
    parser.add_argument("--delta", type=float, help="Desired delta to show options for.", required=True)
    parser.add_argument("--prod_env", type=bool, help="Whether to use a full set of stocks or a test suite of stocks.")
    args = parser.parse_args()

    wheeler = Wheel(args)
    print(wheeler)
    if not wheeler.Run():
       print("Wheel.Run() failed.")
       return
    wheeler.PrintResults()            

if __name__ == "__main__":
    main()