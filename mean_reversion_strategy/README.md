
# Mean Reversion Strategy Backtest

**youtube link** - https://www.youtube.com/watch?v=c9-SIpy3dEw



This implements a **mean reversion trading strategy** using **Bollinger Bands (BB), RSI, and ADX**. The strategy is designed for **1-hour time frame futures data**, with trend confirmation from the **4-hour RSI**.

---

## Strategy Overview

1. **Indicators**:

   * **Bollinger Bands (BB)**: 20-period SMA ± 2 standard deviations (customizable).
     Used to identify overbought/oversold conditions for mean reversion.
   * **RSI (Relative Strength Index)**: 4-hour RSI used as a trend filter.

     * Above 55 → only consider long trades
     * Below 45 → only consider short trades
   * **ADX (Average Directional Index)**: Measures trend strength (0–100).

     * 1H_ADX > 20 and 4H_ADX > 25 → confirm trend strength before taking trades

2. **Entry Rules**:

   * **Long Entry**:

     * 4H_RSI > 55 (bullish trend)
     * 1H_ADX > 20 and 4H_ADX > 25 on 
     * Close < Lower BB on 1H  (oversold condition)
     * Enter next candle
   * **Short Entry**:

     * 4H_RSI < 45 (bearish trend)
     * 1H_ADX > 20 and 4H_ADX > 25 on
     * Close > Upper BB on 1H (overbought condition)
     * Enter next candle

3. **Exit Rules**:

   * Stop-loss: signal candle high/low ± ATR × 4.5
   * Take-profit: exit when price reverts to the opposite BB
   * Automatic exit at end of data (EOD)

4. **Position Management**:

   * Only one open position at a time
   * Trades taken only when all conditions align
   * Pending entry signal used to enter at next candle

