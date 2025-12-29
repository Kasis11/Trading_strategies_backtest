# 8-13-21 EMA & MACD Strategy 

**youtube link** - https://www.youtube.com/watch?v=abvxUhbjJak 

This strategy combines 8, 13, and 21 EMAs with MACD to generate trade signals.  


## Long (Buy) Entry
**Conditions (all must be true on the signal candle):**
1. EMA 8 crosses **above** EMA 21 → short-term bullish momentum.  
2. EMA 13 crosses **above** EMA 21 → confirms stronger uptrend.  
3. EMA 8 > EMA 13 > EMA 21 → clear bullish EMA structure.  
4. MACD line > MACD signal line → confirms bullish momentum.

**Action:**  
- Enter **long trade on the next candle** after the signal candle.  

**Stop-Loss:**  
- Slightly below the lowest low of the last 3 candles (swing low).  

**Take-Profit:**  
- 2.5× the stop-loss distance (risk-reward ratio 2.5:1).  

---

## Short (Sell) Entry
**Conditions (all must be true on the signal candle):**
1. EMA 8 crosses **below** EMA 21 → short-term bearish momentum.  
2. EMA 13 crosses **below** EMA 21 → confirms stronger downtrend.  
3. EMA 8 < EMA 13 < EMA 21 → clear bearish EMA structure.  
4. MACD line < MACD signal line → confirms selling pressure.

**Action:**  
- Enter **short trade on the next candle** after the signal candle.  

**Stop-Loss:**  
- Slightly above the highest high of the last 3 candles (swing high).  

**Take-Profit:**  
- 2.5× the stop-loss distance (risk-reward ratio 2.5:1).  

---

## Trade Exit
1. Exit when **stop-loss or take-profit is hit**.  
2. If still open at the last candle, exit at **close of the last candle** (End-of-Day exit).  

---

**Summary:**  
- **Signal candle:** The candle where EMA & MACD conditions align.  
- **Entry candle:** The **next candle** after the signal candle.  
- **Exit:** On stop-loss, take-profit, or end-of-data.  
