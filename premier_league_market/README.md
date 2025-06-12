# Premier League Market Simulation

This is a Python-based market simulator where each Premier League team is modeled as a separate trading environment. Odds evolve randomly, and a GARCH(1,1) model is calibrated on-the-fly to simulate volatility.

## Features
- Individual market per team
- Liquidity-adjusted pricing
- GARCH(1,1) volatility estimation and dynamic calibration
- CVaR risk management
- Zero-sum PnL enforcement
- Simulated participant trading
- CSV logging of trades

## Installation
```bash
pip install -r requirements.txt
```

## Run the simulation
```bash
python premier_league_market.py
```

## Dependencies
- `arch` for GARCH volatility modeling

## Output
Trades are logged into `trades.csv` by default.
