from strategy_folder._strategy_base_class import Strategy
import pandas as pd

class TwoB(Strategy):

    # ============ STEP 1: THE __init__ METHOD ============
    def __init__(self, lookback:int=20, confirmation_days:int=3):

        if lookback < 3:                                                        # Safety check
            raise ValueError("lookback is too short for meaningful outcome.")
        if confirmation_days < 1 or confirmation_days > 5:                      # Safety check
            raise ValueError("confirmation_days must be between 1 and 5.")
        
        # Calls the parent class. sets self.name, self.data=None, and self._signals_generated=False
        super().__init__(name=f"2B Rule (lookback={lookback}, confirm={confirmation_days})")
        
        # Store everything.
        self.lookback = lookback
        self.confirmation_days = confirmation_days

    # ============ STEP 2: OPENING OF generate_signals() FUNCTION ============
    def generate_signals(self) -> pd.DataFrame:

        if self.data is None:                                                   # Safety check
            raise ValueError("No data loaded, call set_data() first.")
        
        df = self.data.copy()       # Give a clean working copy to modify freely.
        df['signal'] = 0.0          # Add a new column to the DataFrame

        # ============ STEP 3: SWING HIGH AND SWING LOW ============
        df['swing_high'] = df['high'].rolling(window=self.lookback).max()
        df['swing_low'] = df['low'].rolling(window=self.lookback).min()

        # ============ STEP 4: CONVERT COLUMNS TO NUMPY ARRAYS (to reduce pandas workload) ============
        highs   = df['high'].values
        lows    = df['low'].values
        closes  = df['close'].values
        signals = df['signal'].values.copy()        # .values gives a view, not a copy. We don't want to change the DataFrame.
        n       = len(df)                           # Store a variable for len(df) for easier future use.

        # ============ STEP 5: THE MAIN LOOP ============
        for i in range(self.lookback, n):           # Start at self.lookback (=20): we need the whole window of prior bars behind us to define a swing high/low.
            '''
            Start at self.lookback (=20): we need a whole window of prior bars behind us to define a swing high/low.
            --------------------------------------------------------------------------------------------------------
                    Look back 20 bars (self.lookback)     i     look forward 3 bars (self.confirmation_days)
                                         ←──────────────  │  ──────────────→
                           ...bar17  bar18  bar19  bar20 [i] bar21  bar22  bar23 ...
            '''
            # ====== STEP 5b: BEARISH 2B SIGNAL ======
            prior_swing_high = df['high'].iloc[i - self.lookback:i].max()   # Finds the highest high in the 20 bars before i. self.lookback:i intentionally excludes i.
            if highs[i] > prior_swing_high:                                 # If bar i is higher than the prior 20-bar high, breakout confirmed.
                end = min(i + self.confirmation_days, n-1)                  # An index for the loop below, tell the loop to scan forward up to confirmation_days bars, but don't go past the end of the data n.
                for j in range(i, end+1):                                   # If i=21, then end=24, and it scans forward from bar 21 to bar 24. end+1 so it includes bar 24.
                    if closes[j] < prior_swing_high:                        # If bar j closes below swing high, bearish 2B confirmed.
                        signals[j] = -1.0                                   # Sell signal.
                        break                                               # The first bar we can confirm the 2B signal, we stop scanning the rest bars.

            # ====== STEP 5c: BULLISH 2B SIGNAL ======
            prior_swing_low = df['low'].iloc[i - self.lookback:i].min()     # Finds the lowest low in the 20 bars before i. self.lookback:i intentionally excludes i.                
            if lows[i] < prior_swing_low:                                   # If bar i is lower than the prior 20-bar low, breakout confirmed.
                end = min(i + self.confirmation_days, n-1)                  # end is usually equal to i+3, unless it's the last 3 bars of data.
                for j in range(i, end+1):                                   # Scans from bar i to i+3 inclusive.
                    if closes[j] > prior_swing_low:                         # If bar j closes above the swing low, bullish 2B confrimed.
                        if signals[j] == 0.0:                               # Safe Guard: only shoot long signal if nothing else has claimed this bar already.
                            signals[j] = 1.0                                # Buy signal.
                        break                                               # The first bar we can confirm the 2B signal, we stop scanning the rest bars.

        # ============ STEP 6: WRITE BACK AND FINALISE ============
        df['signal'] = signals              # Writes numpy array back into the DataFrame.
        df = df.dropna()                   
        self.data = df                      # Stores finished DataFrame back into the strategy object so get_signals() can return it.
        self._signals_generated = True    
        return df
                        


if __name__ == "__main__":
    from data_loader import load_historical_data
    s = TwoB(lookback=20, confirmation_days=3)
    df = load_historical_data("NVDA", "2010-01-01", "2026-01-01")
    s.set_data(df)
    out = s.generate_signals()
    print(out[['close', 'high', 'low', 'swing_high', 'swing_low', 'signal']].tail(30))
    print("longs :", (out['signal'] ==  1).sum())
    print("shorts:", (out['signal'] == -1).sum())