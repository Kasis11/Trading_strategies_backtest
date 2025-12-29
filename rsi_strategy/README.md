# RSI Trend Strategy Backtest

**youtube link** - https://www.youtube.com/watch?v=zNvFmSSj_OE

This is a **momentum + trend trading strategy** using the RSI indicator and the 200-period SMA. The strategy is designed for **1-hour time frame futures data**.

---

## Strategy Overview

1. **Indicators**:
   - **RSI (Relative Strength Index)**: 14-period RSI smoothed with a 3-period SMA (`RSI_Signal`).
   - **200-period SMA**: Used as a long-term trend filter.

2. **Entry Rules**:
   - **Long Entry**:
     - Close > SMA200 (bullish trend)
     - RSI_Signal > `rsi_upper` (default 55)
     - Candle size < `max_candle_size` (default 4%)
   - **Short Entry**:
     - Close < SMA200 (bearish trend)
     - RSI_Signal < `rsi_lower` (default 35)
     - Candle size < `max_candle_size`

3. **Exit Rules**:
   - Stop-loss based on ATR: `stop_loss = entry ± ATR * atr_mult`
   - Take-profit: `2:1` risk-reward ratio
   - Maximum holding period: default 7 days
   - Minimum holding period: default 2 days

4. **Position Management**:
   - Only one open position at a time
   - Trades taken only when all conditions align

---