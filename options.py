from wheel_enums import StrategyType
from functools import total_ordering


class ParsedOption:
   def __init__(self, symbol, ticker, currentPrice):
      self.symbol = symbol
      self.ticker = ticker
      self.currentPrice = currentPrice
      self.strike = None
      self.percentMove = None
   
   def __eq__(self, other):
        return self.percentMove == other.percentMove

   def __lt__(self, other):
        return self.percentMove > other.percentMove

@total_ordering
class ParsedPutOption(ParsedOption):
   def __init__(self, symbol, ticker, currentPrice, optionJSONData):
      super().__init__(symbol, ticker, currentPrice)
      self.optionType = StrategyType.PUT
      self.mid = Midpoint(optionJSONData['bid'], optionJSONData['ask'])
      self.strike = optionJSONData['strike']
      self.delta = optionJSONData['greeks']['delta']
      self.percentMove = (self.currentPrice - self.strike) / (1.0 * self.currentPrice)
   
   def __str__(self):
      return f"{self.symbol} Put option @{self.strike} for ${self.mid * 100} with Delta: {self.delta}"

   
@total_ordering
class ParsedCCSpread(ParsedOption):
   def __init__(self, symbol, ticker, currentPrice, optionJSONDataToSell, optionJSONDataToBuy):
      super().__init__(symbol, ticker, currentPrice)
      self.optionType = StrategyType.CC_SPREAD
      self.mid = Midpoint(optionJSONDataToSell['bid'], optionJSONDataToSell['ask']) - Midpoint(optionJSONDataToBuy['bid'], optionJSONDataToBuy['ask'])
      self.strike = optionJSONDataToSell['strike']
      self.width = optionJSONDataToBuy['strike'] - self.strike
      self.percentMove = (self.strike - self.currentPrice) / (1.0 * self.currentPrice)
      self.delta = optionJSONDataToSell['greeks']['delta']
   
   def __str__(self):
      return f"{self.symbol} Call Spread ({self.strike}, {self.strike + self.width}) for ${self.mid * 100} with Delta: {self.delta}"

def Midpoint(x,y):
    return (x + y) / 2.0