PROD_STOCK_FILE = "stocks_prod.txt"
TEST_STOCK_FILE = "stocks_test.txt"

def parse_stock_tickers(prod):
  """
  Parses a text file where each line contains a stock ticker.

  Args:
    filename: The name of the text file (default: "stock_tickers.txt").

  Returns:
    A list of stock tickers.
  """
  filename = TEST_STOCK_FILE
  if prod:
    filename = PROD_STOCK_FILE
  print("Parsing: ", filename)
  try:
    with open(filename, "r") as file:
      tickers = [line.strip() for line in file]
      return tickers
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found in the current directory.")
    return []