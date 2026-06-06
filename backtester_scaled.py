import pandas as pd
import numpy as np
from typing import Dict, List
from position_sizer import calculate_position_size


class BacktesterScaled:
    """
    Backtesting engine with scaling in/out.

    Each signal opens a new tranche (1/3 of full risk) while simultaneously
    closing the oldest opposite tranche. Up to max_tranches per side.
    Each tranche has its own independent entry price and stop loss.
    """

    def __init__(self, initial_balance: float = 6000.0, risk_pct: float = 0.02,
                 slippage_pct: float = 0.0, max_tranches: int = 3):
        self.initial_balance = initial_balance
        self.risk_pct = risk_pct
        self.slippage_pct = slippage_pct
        self.max_tranches = max_tranches
        self.tranche_risk_pct = risk_pct / max_tranches  # each tranche risks 1/N of total
        self._reset()

    def _reset(self):
        self.current_balance = self.initial_balance
        self.long_tranches: List[Dict] = []   # each: {entry_price, stop_loss, size, entry_date}
        self.short_tranches: List[Dict] = []
        self.trades = []
        self.equity_curve = []

    def _close_tranche(self, tranche: Dict, exit_price: float, exit_date,
                       direction: int, reason: str, verbose: bool):
        """Close a single tranche and record the trade."""
        pnl = (exit_price - tranche['entry_price']) * tranche['size'] * direction
        pnl_pct = (pnl / self.current_balance) * 100 if self.current_balance > 0 else 0
        self.current_balance += pnl

        self.trades.append({
            'entry_date': tranche['entry_date'],
            'exit_date': exit_date,
            'direction': "long" if direction == 1 else "short",
            'entry_price': tranche['entry_price'],
            'exit_price': round(exit_price, 2),
            'position_size': tranche['size'],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        })

        if verbose:
            label = "LONG" if direction == 1 else "SHORT"
            n_long = len(self.long_tranches)
            n_short = len(self.short_tranches)
            print(f"  Balance: £{self.current_balance:,.2f} | CLOSED {label} tranche on {exit_date.date()} | "
                  f"{tranche['size']:.4f} units @ £{exit_price:.2f} | "
                  f"PnL: £{pnl:.2f} ({pnl_pct:.2f}%) | {reason} | "
                  f"Open: {n_long}L / {n_short}S")

    def run(self, data: pd.DataFrame, strategy, verbose: bool = True) -> Dict:
        self._reset()

        strategy.set_data(data)
        df = strategy.generate_signals()

        if verbose:
            print(f"Strategy       : {strategy.name}")
            print(f"Initial balance: £{self.initial_balance:,.2f}")
            print(f"Risk per trade : {self.risk_pct * 100:.1f}% "
                  f"({self.tranche_risk_pct * 100:.2f}% per tranche x {self.max_tranches})")
            print(f"Slippage       : {self.slippage_pct * 100:.3f}%\n")

        pending_signal = 0
        pending_sl = 0.0
        has_custom_sl = 'stop_loss' in df.columns

        for date, row in df.iterrows():

            # === CHECK STOP LOSSES on all open tranches ===
            # Long tranches: SL hit if bar's low <= stop_loss
            closed_longs = []
            for i, t in enumerate(self.long_tranches):
                if row['low'] <= t['stop_loss']:
                    exit_price = t['stop_loss'] * (1 - self.slippage_pct)
                    closed_longs.append(i)
                    self._close_tranche(t, exit_price, date, 1, "SL HIT", verbose)
            for i in reversed(closed_longs):
                self.long_tranches.pop(i)

            # Short tranches: SL hit if bar's high >= stop_loss
            closed_shorts = []
            for i, t in enumerate(self.short_tranches):
                if row['high'] >= t['stop_loss']:
                    exit_price = t['stop_loss'] * (1 + self.slippage_pct)
                    closed_shorts.append(i)
                    self._close_tranche(t, exit_price, date, -1, "SL HIT", verbose)
            for i in reversed(closed_shorts):
                self.short_tranches.pop(i)

            # === EXECUTE PENDING SIGNAL from previous bar ===
            if pending_signal != 0 and self.current_balance > 0:

                if pending_signal == 1:
                    # BUY: close oldest short tranche + open new long tranche

                    # 1) Scale out of shorts (FIFO — close the oldest)
                    if self.short_tranches:
                        t = self.short_tranches.pop(0)
                        exit_price = row['open'] * (1 + self.slippage_pct)  # covering: slips higher
                        self._close_tranche(t, exit_price, date, -1, "SIGNAL", verbose)

                    # 2) Scale into longs
                    if len(self.long_tranches) < self.max_tranches:
                        entry_price = row['open'] * (1 + self.slippage_pct)  # buying: slips higher
                        stop_loss = pending_sl

                        if stop_loss != entry_price and self.current_balance > 0:
                            try:
                                sizing = calculate_position_size(
                                    account_balance=self.current_balance,
                                    risk_pct=self.tranche_risk_pct,
                                    entry_price=entry_price,
                                    stop_loss_price=stop_loss
                                )
                                max_units = (self.current_balance * 20) / entry_price
                                units = min(sizing['units_to_trade'], max_units)

                                tranche = {
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'size': units,
                                    'entry_date': date
                                }
                                self.long_tranches.append(tranche)

                                if verbose:
                                    print(f"  Balance: £{self.current_balance:,.2f} | OPENED LONG tranche on {date.date()} | "
                                          f"{units:.4f} units @ £{entry_price:.2f} | SL: £{stop_loss:.2f} | "
                                          f"Open: {len(self.long_tranches)}L / {len(self.short_tranches)}S")
                            except ValueError:
                                pass  # position too small, skip

                elif pending_signal == -1:
                    # SELL: close oldest long tranche + open new short tranche

                    # 1) Scale out of longs (FIFO)
                    if self.long_tranches:
                        t = self.long_tranches.pop(0)
                        exit_price = row['open'] * (1 - self.slippage_pct)  # selling: slips lower
                        self._close_tranche(t, exit_price, date, 1, "SIGNAL", verbose)

                    # 2) Scale into shorts
                    if len(self.short_tranches) < self.max_tranches:
                        entry_price = row['open'] * (1 - self.slippage_pct)  # shorting: slips lower
                        stop_loss = pending_sl

                        if stop_loss != entry_price and self.current_balance > 0:
                            try:
                                sizing = calculate_position_size(
                                    account_balance=self.current_balance,
                                    risk_pct=self.tranche_risk_pct,
                                    entry_price=entry_price,
                                    stop_loss_price=stop_loss
                                )
                                max_units = (self.current_balance * 20) / entry_price
                                units = min(sizing['units_to_trade'], max_units)

                                tranche = {
                                    'entry_price': entry_price,
                                    'stop_loss': stop_loss,
                                    'size': units,
                                    'entry_date': date
                                }
                                self.short_tranches.append(tranche)

                                if verbose:
                                    print(f"  Balance: £{self.current_balance:,.2f} | OPENED SHORT tranche on {date.date()} | "
                                          f"{units:.4f} units @ £{entry_price:.2f} | SL: £{stop_loss:.2f} | "
                                          f"Open: {len(self.long_tranches)}L / {len(self.short_tranches)}S")
                            except ValueError:
                                pass

            # Capture signal for next bar execution
            pending_signal = int(row['signal'])
            if pending_signal == 1:
                pending_sl = (row['stop_loss']
                              if has_custom_sl and pd.notna(row['stop_loss'])
                              else row['low'])
            elif pending_signal == -1:
                pending_sl = (row['stop_loss']
                              if has_custom_sl and pd.notna(row['stop_loss'])
                              else row['high'])

            # Record equity
            self.equity_curve.append({
                'date': date,
                'balance': self.current_balance
            })

        # === Force-close all open tranches at end of data ===
        final_price = df.iloc[-1]['close']
        had_open_tranches = bool(self.long_tranches or self.short_tranches)
        for t in self.long_tranches:
            pnl = (final_price - t['entry_price']) * t['size']
            self.current_balance += pnl
        for t in self.short_tranches:
            pnl = (t['entry_price'] - final_price) * t['size']
            self.current_balance += pnl
        if had_open_tranches:
            self.equity_curve.append({'date': df.index[-1], 'balance': self.current_balance})
            if verbose:
                print(f"\nForce-closed {len(self.long_tranches)} long + {len(self.short_tranches)} short tranches at end of data")
        self.long_tranches.clear()
        self.short_tranches.clear()

        # === Metrics ===
        metrics = self._calculate_metrics()

        if verbose:
            print(f"\n{'='*40}")
            print(f"Final balance  : £{metrics['final_balance']:,.2f}")
            print(f"Total return   : {metrics['total_return_pct']:.2f}%")
            print(f"Total trades   : {metrics['num_trades']}")
            print(f"Win rate       : {metrics['win_rate_pct']:.1f}%")
            print(f"Sharpe ratio   : {metrics['sharpe_ratio']:.2f}")
            print(f"Max drawdown   : {metrics['max_drawdown_pct']:.2f}%")
            print(f"Profit factor  : {metrics['profit_factor']}")
            print(f"Expectancy     : £{metrics['expectancy']:,.2f} per trade")
            print(f"Avg win        : £{metrics['avg_win']:,.2f}")
            print(f"Avg loss       : £{metrics['avg_loss']:,.2f}")
            print(f"Largest win    : £{metrics['largest_win']:,.2f}")
            print(f"Largest loss   : £{metrics['largest_loss']:,.2f}")

        return metrics

    def _calculate_metrics(self) -> Dict:
        total_return_pct = (
            (self.current_balance - self.initial_balance) / self.initial_balance
        ) * 100

        num_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0

        equity_df = pd.DataFrame(self.equity_curve)
        daily_returns = equity_df['balance'].pct_change().dropna()

        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        rolling_max = equity_df['balance'].cummax()
        drawdown = (equity_df['balance'] - rolling_max) / rolling_max
        max_drawdown_pct = drawdown.min() * 100

        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else float('inf')

        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        avg_win = round(gross_profit / len(winning_trades), 2) if winning_trades else 0
        avg_loss = round(gross_loss / len(losing_trades), 2) if losing_trades else 0

        expectancy = round(sum(t['pnl'] for t in self.trades) / num_trades, 2) if num_trades > 0 else 0

        largest_win = round(max((t['pnl'] for t in self.trades), default=0), 2)
        largest_loss = round(min((t['pnl'] for t in self.trades), default=0), 2)

        return {
            'final_balance': self.current_balance,
            'total_return_pct': total_return_pct,
            'num_trades': num_trades,
            'win_rate_pct': round(win_rate * 100, 1),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }


if __name__ == "__main__":
    from data_loader import load_historical_data
    from strategy_folder.two_b import TwoB

    df = load_historical_data("ZW=F", "2010-01-01", "2026-04-16")
    strategy = TwoB(lookback=20, confirmation_days=3)

    print("=" * 70)
    print("ORIGINAL BACKTESTER (all-in / all-out)")
    print("=" * 70)
    from backtester import Backtester
    bt_orig = Backtester(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001)
    r1 = bt_orig.run(df, strategy, verbose=False)
    for k in ['final_balance','total_return_pct','num_trades','win_rate_pct',
              'sharpe_ratio','max_drawdown_pct','profit_factor','expectancy']:
        print(f"  {k:20s}: {r1[k]}")

    print()
    print("=" * 70)
    print("SCALED BACKTESTER (1/3 tranches, independent SLs)")
    print("=" * 70)
    strategy2 = TwoB(lookback=20, confirmation_days=3)
    bt_scaled = BacktesterScaled(initial_balance=10000, risk_pct=0.02, slippage_pct=0.0001, max_tranches=3)
    r2 = bt_scaled.run(df, strategy2, verbose=False)
    for k in ['final_balance','total_return_pct','num_trades','win_rate_pct',
              'sharpe_ratio','max_drawdown_pct','profit_factor','expectancy']:
        print(f"  {k:20s}: {r2[k]}")
