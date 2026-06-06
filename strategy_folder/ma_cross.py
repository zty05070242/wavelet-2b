from strategy_folder._strategy_base_class import Strategy
from data_loader import load_historical_data
import pandas as pd

class MovingAverageCrossover(Strategy):
    def __init__(self, fast_period:int=50, slow_period:int=200):
        # Safety check
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("fast_ma or low_ma must be positive.")
        if fast_period >= slow_period:
            raise ValueError("fast_ma must be shorter than slow_ma.")
        
        # Call the parent __init__ setup
        super().__init__(name=f"MA Crossover Strategy {fast_period}/{slow_period}")

        # Store them after safety checking
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self) -> pd.DataFrame:
        # Safety check
        if self.data is None:
            raise ValueError("No data loaded, call set_data() first.")
        
        df = self.data.copy()       # Make a copy of the fetched data for generating signals
        
        # Create columns of 'fast_ma', 'slow_ma' and 'signal'
        df['fast_ma'] = df['close'].rolling(window=self.fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_period).mean()
        df['signal'] = 0.0

        # BUY signal is 1, SELL signal is -1, defaul is 0.
        df.loc[(df['fast_ma'] > df['slow_ma']) & (df['fast_ma'].shift(1) <= df['slow_ma'].shift(1)), 'signal'] = 1
        df.loc[(df['fast_ma'] < df['slow_ma']) & (df['fast_ma'].shift(1) >= df['slow_ma'].shift(1)), 'signal'] = -1

        # Clean up
        df = df.dropna()
        self.data = df
        self._signals_generated = True

        return df

if __name__ == "__main__":
    strategy = MovingAverageCrossover(fast_period=10, slow_period=20)
    print(f"Testing strategy {strategy.name}")
    df = load_historical_data("NVDA", "2010-01-01", "2026-04-12")
    strategy.set_data(df)
    signals = strategy.generate_signals()
    print(f"Bars:{len(signals)}")
    print(signals[['close', 'fast_ma', 'slow_ma', 'signal']].tail(100))
    print(f"BUY signals: {(signals['signal']==1).sum()}")
    print(f"SELL signals: {(signals['signal']==-1).sum()}")

# ----------------------------------------------------------Strategy Logic Flow--------------------------------------------------------
# 1. MA cross strategy is called with fast_period and slow_period inputs. So this strategy runs.
# 2. load_historical_data() in data_loader.py is called. So it returns the historical data.
# 3. set_data() in the Strategy base class is called using the returned historical data. 
#    It stores the data in self.data (this happens in the child class too, because it inherits its parent class's properties).
# 4. When generate_signals() runs, it reads self.data, and uses self.data to generate signals. It returns the signals generated as df (the other df).
# 5. The signals returned are printed out.