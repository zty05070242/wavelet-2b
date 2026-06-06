
def calculate_position_size(
        account_balance:float, 
        entry_price:float, 
        stop_loss_price:float, 
        risk_pct:float, 
        min_position:float=0.1
        ) -> dict:
   
    # Safety: validate any input before any calculation
    if account_balance <= 0:
        raise ValueError("Account balance must be above 0")
    if risk_pct <= 0 or risk_pct > 0.1:
        raise ValueError("Risk percentage must be between 0 and 0.1")
    if entry_price <= 0 or stop_loss_price <= 0:
        raise ValueError("Entry and stop loss must be positive")
    if entry_price == stop_loss_price:
        raise ValueError("Entry should not equal to stop loss.")
 
    max_risk = account_balance * risk_pct
    risk_per_unit = abs(entry_price - stop_loss_price)
    units_to_trade = max_risk / risk_per_unit
    position_size = units_to_trade * entry_price

    # Safety: check values after calculation as well.
    if position_size < min_position:
        raise ValueError(f"Calculated position size {position_size} is smaller than the minimum: {min_position}. Adjust your levels.")

    direction = "long" if entry_price > stop_loss_price else "short"

    return {
        # canonical keys for engine use
        "account_balance": account_balance,
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "direction": direction,
        "risk_pct": risk_pct,
        "max_risk_allowed": round(max_risk, 2),
        "risk_per_unit": risk_per_unit,
        "units_to_trade": units_to_trade,
        "position_size": round(position_size, 2),
    }

# test the code using main guard
if __name__ == "__main__":
    position = calculate_position_size(10000, 100, 99, 0.02)
    for key, value in position.items():
        print(f"{key:15} : {value}")

