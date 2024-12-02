from datetime import date
from functools import partial
from functools import cmp_to_key
from options import ParsedCCSpread, ParsedPutOption

class Util:
    def __init__(self, targetRatio, delta, expiration):
        self.expiration = expiration
        self.desiredDelta= delta
        today = date.today()
        self.mustExceed = targetRatio * ((expiration - today).days)/(30.0)

    def PrintPut(self, option, ticker, curPrice):
        print(f"----{ticker} @ {curPrice}----")
        print(option['description'], option['greeks']['delta'])
        print(option['bid'], option['ask'])

    def PrintPuts(self, options, ticker, curPrice, topN=0):
        if len(options) == 0:
            return
        print(f"----{ticker} @ {curPrice}----")
        if topN == 0:
            topN = len(options)
        for i in range(min(topN, len(options))):
            print(options[i])
    
    def PrintSpreads(self, spreads, ticker, curPrice, topN=0):
        if len(spreads) == 0:
            return
        print(f"----{ticker} @ {curPrice}----")
        if topN == 0:
            topN = len(spreads)
        for i in range(min(topN, len(spreads))):
            print(spreads[len(spreads) - i - 1])

    def FilterPutsForTargetRatio(self, putData, curPrice):
        """
        Filters out the set of puts to show puts whose premium match or exceed the target APY.
        """
        res = []
        for put in putData:
            strike = put['strike']
            if strike < curPrice and abs(put['greeks']['delta']) <= self.desiredDelta:
                # We need to ensure that the premium collected exceeds the cost of having the money locked up.
                if (put['bid'] / (1.0 * strike)) * 100 > self.mustExceed:
                    res.append(put)
        return res
    
    def BuildVerticalCallSpreads(self, calls, spreadLength, symbol, ticker, curPrice):
        if spreadLength <= 0:
            print("invalid spread length: ", spreadLength)
        spreads = []
        for i in range(len(calls) - spreadLength):
            spreads.append(ParsedCCSpread(symbol, ticker, curPrice, calls[i][1], calls[i+spreadLength][1]))
        return spreads
    
    def FilterVerticalCallSpreadsForTargetRatio(self, callSpreads, curPrice):
        filteredITMSpreads = [spread for spread in callSpreads if spread.strike > curPrice]
        meetsTarget = []
        for spread in filteredITMSpreads:
            # Calculate premium at Midpoint.
            capitalNeeded = spread.width
            if spread.delta < self.delta and spread.mid / capitalNeeded * 100 > self.mustExceed:
                meetsTarget.append(spread)
        return meetsTarget
    
    def OptimizeOrdering(aggregateOptionData, symbolsToCurrentPrice):
        comparePutsWithCurrentPriceData = partial(putOptionCompare, symbolsToCurrentPrice)
        sorted_data = sorted(aggregateOptionData, key=cmp_to_key(lambda put1, put2 : comparePutsWithCurrentPriceData(put1, put2)))
        return sorted_data

def Midpoint(x,y):
    return (x + y) / 2.0

# Global Namespace
def putOptionCompare(symbolsToCurrentPrice, put1, put2):
    # print("COMPARING: ", put1, put2)
    symbol1 = put1['symbol']
    symbol2 = put2['symbol']
    symbol1Price = symbolsToCurrentPrice[symbol1]
    symbol2Price = symbolsToCurrentPrice[symbol2]
    percentDiff1 = symbol1Price - put1['strike'] / symbol1Price
    percentDiff2 = symbol2Price - put2['strike'] / symbol2Price
    premiumForChange1 = put1['strike'] / put1['bid'] 
    premiumForChange2 = put2['strike'] / put2['bid']
    return premiumForChange1 - premiumForChange2




