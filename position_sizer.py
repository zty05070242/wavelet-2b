import math


def calculate_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss_price: float,
    risk_pct: float,
    min_position: float = 0.1,
    contract_multiplier: float = 1.0,
    integer_positions: bool = False,
) -> dict:
    """Size a position so a stop loss risks no more than ``risk_pct`` of equity."""
    if account_balance <= 0:
        raise ValueError("Account balance must be above 0")
    if risk_pct <= 0 or risk_pct > 0.1:
        raise ValueError("Risk percentage must be between 0 and 0.1")
    if not math.isfinite(entry_price) or not math.isfinite(stop_loss_price):
        raise ValueError("Entry and stop loss must be finite")
    if entry_price == stop_loss_price:
        raise ValueError("Entry should not equal to stop loss.")
    if contract_multiplier <= 0:
        raise ValueError("Contract multiplier must be positive")

    max_risk = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_loss_price) * contract_multiplier
    units_to_trade = max_risk / risk_per_unit
    if integer_positions:
        units_to_trade = math.floor(units_to_trade)
    position_size = units_to_trade * abs(entry_price) * contract_multiplier

    if 0 < position_size < min_position:
        raise ValueError(
            f"Calculated position size {position_size} is below the minimum {min_position}"
        )

    direction = "long" if entry_price > stop_loss_price else "short"

    return {
        "account_balance": account_balance,
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "direction": direction,
        "risk_pct": risk_pct,
        "max_risk_allowed": round(max_risk, 2),
        "risk_per_unit": risk_per_unit,
        "units_to_trade": units_to_trade,
        "position_size": round(position_size, 2),
        "contract_multiplier": contract_multiplier,
    }

if __name__ == "__main__":
    position = calculate_position_size(10000, 100, 99, 0.02)
    for key, value in position.items():
        print(f"{key:15} : {value}")
