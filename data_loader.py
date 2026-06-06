import yfinance as yf
import pandas as pd

def load_historical_data(ticker:str, start_date:str, end_date:str, interval:str="1d") -> pd.DataFrame:
    print(f"Loading historical data of {ticker} from {start_date} to {end_date} with interval '{interval}'")

    historical_data = yf.download(
        tickers = ticker,
        start = start_date,
        end = end_date,
        interval = interval,
        auto_adjust = True,
        progress = True
    )

    if historical_data.empty:
        raise ValueError("Unable to load data")
    # fix new yfinance behaviour: multi-index columns
    if isinstance(historical_data.columns, pd.MultiIndex):
        # flatten the columns: remove the ticker level, just give me level 0
        historical_data.columns = historical_data.columns.get_level_values(0)
    
    #sort the data
    historical_data = historical_data[["Open", "High", "Low", "Close", "Volume"]]
    historical_data = historical_data.copy() #make a copy first
    historical_data.columns = ["open", "high", "low", "close", "volume"]
    historical_data = historical_data.dropna().sort_index()

    print(f"Successfully loaded {len(historical_data)} bars for {ticker}")
    return historical_data


#test the loader
if __name__ == "__main__":
    data_loaded = load_historical_data("TSLA", "2010-01-01", "2026-04-01")
    print(data_loaded)

