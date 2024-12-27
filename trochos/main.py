

import argparse
import lib.stock_loader as stock_loader
from wheel import Wheel

def main():
    """
    Parses user flags and launches the application.
    """
    parser = argparse.ArgumentParser(description="Trochos Tool")
    parser.add_argument("--strike_spread", type=int, help="User capital to play with.")
    parser.add_argument("--target_ratio", type=float, help="Ratio of premium to capital that we want to hit.", required=True)
    parser.add_argument("--top_n", type=int, help="Returns to the top n contracts from the pool of useable contracts.")
    parser.add_argument("--delta", type=float, help="Desired delta to show options for.", required=True)
    parser.add_argument("--prod", action='store_true', help="Whether to use a full set of stocks or a test suite of stocks.")
    parser.add_argument("--expiration", type=str, help="yyyy-mm-dd formated expiration for options.", required=True)
    parser.add_argument("--capital", type=int, help="How much capital to work with (avoiding margin).")
    parser.add_argument("--strat", type=str, help="strategy to employ: CALL or PUT (--strike_spread turns this into a spread)", required=True)
    args = parser.parse_args()

    tickers = stock_loader.parse_stock_tickers(args.prod)

    wheeler = Wheel(args, tickers)
    print(wheeler)
    if not wheeler.Run():
       print("Wheel.Run() failed.")
       return
    wheeler.PrintResults()            

if __name__ == "__main__":
    main()