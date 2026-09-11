from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def calculate_metrics(
    initial_balance: float,
    final_balance: float,
    trades: list[dict],
    equity_curve: list[dict],
) -> Dict:
    """Calculate account-level metrics from a daily mark-to-market equity curve."""
    equity = pd.DataFrame(equity_curve)
    if equity.empty:
        equity = pd.DataFrame([{"date": pd.NaT, "balance": initial_balance, "exposed": False}])

    equity = equity.drop_duplicates("date", keep="last").sort_values("date")
    daily_returns = equity["balance"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    volatility = daily_returns.std()
    sharpe = (
        daily_returns.mean() / volatility * np.sqrt(252)
        if pd.notna(volatility) and volatility > 0
        else 0.0
    )
    annualized_volatility = volatility * np.sqrt(252) * 100 if pd.notna(volatility) else 0.0

    rolling_max = equity["balance"].cummax()
    drawdown = (equity["balance"] - rolling_max) / rolling_max
    max_drawdown_pct = float(drawdown.min() * 100)

    first_date = pd.Timestamp(equity["date"].iloc[0])
    last_date = pd.Timestamp(equity["date"].iloc[-1])
    years = max((last_date - first_date).days / 365.25, 0.0) if pd.notna(first_date) else 0.0
    if years > 0 and initial_balance > 0 and final_balance > 0:
        cagr_pct = ((final_balance / initial_balance) ** (1 / years) - 1) * 100
    else:
        cagr_pct = 0.0
    calmar = cagr_pct / abs(max_drawdown_pct) if max_drawdown_pct < 0 else 0.0

    wins = [trade for trade in trades if trade["pnl"] > 0]
    losses = [trade for trade in trades if trade["pnl"] < 0]
    gross_profit = sum(trade["pnl"] for trade in wins)
    gross_loss = abs(sum(trade["pnl"] for trade in losses))
    num_trades = len(trades)

    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss else float("inf")
    exposure_pct = (
        float(equity["exposed"].astype(bool).mean() * 100)
        if "exposed" in equity.columns
        else 0.0
    )

    return {
        "final_balance": round(float(final_balance), 2),
        "total_return_pct": round((final_balance / initial_balance - 1) * 100, 2),
        "cagr_pct": round(float(cagr_pct), 2),
        "annualized_volatility_pct": round(float(annualized_volatility), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "calmar_ratio": round(float(calmar), 2),
        "exposure_pct": round(exposure_pct, 1),
        "num_trades": num_trades,
        "win_rate_pct": round(len(wins) / num_trades * 100, 1) if num_trades else 0.0,
        "profit_factor": profit_factor,
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(gross_loss / len(losses), 2) if losses else 0.0,
        "expectancy": round(sum(t["pnl"] for t in trades) / num_trades, 2) if num_trades else 0.0,
        "largest_win": round(max((t["pnl"] for t in trades), default=0.0), 2),
        "largest_loss": round(min((t["pnl"] for t in trades), default=0.0), 2),
        "total_costs": round(sum(t.get("costs", 0.0) for t in trades), 2),
        "trades": trades,
        "equity_curve": equity.to_dict("records"),
    }
