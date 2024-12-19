import argparse
from lib.utils import Util
from options import *
from wheel_enums import StrategyType
from lib import tradier
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import heapq
from lib.errors import WheelError

def BuildSpreads(stratType, optionData, spreadLength, symbol, ticker, curPrice):
   if spreadLength <= 0:
            return None, WheelError("invalid spread length: " + spreadLength)

   parsedConstructor = None
   if stratType == StrategyType.CC_SPREAD:
      parsedConstructor = ParsedCCSpread
   else:
      return None, WheelError("Build Put Spreads Not implemented")
   
   spreads = []
   for i in range(len(optionData) - spreadLength):
      spreads.append(parsedConstructor(symbol, ticker, curPrice, optionData[i], optionData[i+spreadLength]))
   return spreads, None


class Wheel:
   def __init__(self, args, tickers):
      # Access parsed arguments
      self.spreadLength = args.strike_spread
      if args.strat == "CALL":
         if self.spreadLength != None:
            self.strategyType = StrategyType.CC_SPREAD
         else:
            self.strategyType = StrategyType.CALL
      else:
         if self.spreadLength != None:
            self.strategyType = StrategyType.PC_SPREAD
         else:
            self.strategyType = StrategyType.PUT
      self.targetRatio = args.target_ratio
      self.delta = args.delta
      self.topN = args.top_n
      self.tickers = tickers
      self.expiration = datetime.strptime(args.expiration, "%Y-%m-%d").date()
      self.utils = Util(self.targetRatio, self.delta, self.expiration, args.capital)
      self.processorMap = {
         StrategyType.PUT: self.buildFilteredPutData,
         StrategyType.CALL: self.buildFilteredCallData,
         StrategyType.PC_SPREAD: self.buildFilteredPCSpreadData,
         StrategyType.CC_SPREAD: self.buildFilteredCCSpreadData
      }
      self.symbolsToCurrentPrice = {}
      self.runResults = []
      # TODO: validate flag inputs

   def __str__(self):
      return f"Generating best {self.strategyType.value} picks expiring on {self.expiration} that achieve a {self.targetRatio}% monthly return on capital across the tickers: {self.tickers}"

   def updateSymbolsToCurrentPrice(self, ticker):
      stockQuote, err = tradier.get_stock_quote(ticker)
      if err:
         return None, None, err
      self.symbolsToCurrentPrice[stockQuote['symbol']] = stockQuote['last']
      return stockQuote['symbol'], stockQuote['last'], None
   
   def buildFilteredPutData(self, ticker):
    symbol, curPrice, err = self.updateSymbolsToCurrentPrice(ticker)
    if err:
       return ticker, err
    optionData, err = tradier.get_put_quotes(symbol, self.expiration)
    if err:
       return ticker, err
    if len(optionData) == 0:
       return ticker, WheelError("get_put_quotes returned nothing")
    
    parsedPuts = []
    for option in optionData:
       parsedPuts.append(ParsedPutOption(symbol, ticker, curPrice, option))
       
    filteredPuts = self.utils.FilterPutsForTargetRatio(parsedPuts, curPrice)
    
    sortedPuts = []
    for put in filteredPuts:
        heapq.heappush(sortedPuts, put)
    return (ticker, [heapq.heappop(sortedPuts) for _ in range(min(self.topN, len(filteredPuts)))]), None
   
   def buildFilteredCallData(self, ticker):
    symbol, curPrice, err = self.updateSymbolsToCurrentPrice(ticker)
    if err:
       return ticker, err
    optionData, err = tradier.get_call_quotes(symbol, self.expiration)
    if err:
       return ticker, err
    if len(optionData) == 0:
       return ticker, WheelError("get_call_quotes returned nothing")
    
    parsedCalls = []
    for option in optionData:
       parsedCalls.append(ParsedCallOption(symbol, ticker, curPrice, option))
    filteredCalls = self.utils.FilterCallsForTargetRatio(parsedCalls, curPrice)
    
    sortedCalls = []
    for put in filteredCalls:
        heapq.heappush(sortedCalls, put)
    return (ticker, [heapq.heappop(sortedCalls) for _ in range(min(self.topN, len(filteredCalls)))]), None
   
   def buildFilteredPCSpreadData(self, ticker):
      return (ticker, []), WheelError("PC Spreads are not implemented.")
    
   def buildFilteredCCSpreadData(self, ticker):
      symbol, curPrice, err = self.updateSymbolsToCurrentPrice(ticker)
      if err:
         return ticker, err
      optionData, err = tradier.get_call_quotes(symbol, self.expiration)
      if err:
         return ticker, err
      
      rawSpreads, err = BuildSpreads(StrategyType.CC_SPREAD, optionData, self.spreadLength, symbol, ticker, curPrice)
      if err:
         return ticker, err

      filteredSpreadData = self.utils.FilterVerticalCallSpreadsForTargetRatio(rawSpreads, curPrice)
      aggregateSpreads = []
      for spread in filteredSpreadData:
         heapq.heappush(aggregateSpreads, spread)
      return (ticker, [heapq.heappop(aggregateSpreads) for _ in range(min(self.topN, len(aggregateSpreads)))]), None


   def Run(self):
    resultsMap = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
       future_to_item = {executor.submit(self.processorMap[self.strategyType], ticker): ticker for ticker in self.tickers}
       for future in as_completed(future_to_item):
          item = future_to_item[future]
          try:
             data, err = future.result()
             if err:
                print(f"{data} returned error: {err}")
             else:
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
         if self.strategyType == StrategyType.PUT or self.strategyType == StrategyType.CALL:
            self.utils.PrintBaseOptions(options, ticker, self.symbolsToCurrentPrice[ticker], self.topN)
         else:
            self.utils.PrintSpreads(options, ticker, self.symbolsToCurrentPrice[ticker], self.topN)