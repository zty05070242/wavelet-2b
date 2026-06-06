import pandas as pd
from abc import ABC, abstractmethod

class Strategy(ABC):
    """
    Every strategy must be able to:
    1. Have a name
    2. Load data via set_data()
    3. Generate their own signals via generate_signals()
    4. Return the signals via get_signals()
    """
    def __init__(self, name: str = "default_name"):
        self.name = name
        self.data = None
        self._signals_generated = False

    def set_data(self, data: pd.DataFrame):
        """Receives a pre-loaded OHLCV DataFrame and stores it in the strategy."""
        self.data = data.copy()
        self._signals_generated = False

    @abstractmethod     # Every strategy must have its own generate_signals() function. This is a shell.
    def generate_signals(self) -> pd.DataFrame:
        pass

    def get_signals(self) -> pd.DataFrame:
        if self.data is None:
            raise ValueError("No data loaded. Call set_data() first.")
        if not self._signals_generated:
            raise ValueError("Signals not yet generated. Call generate_signals() first.")
        return self.data