# Scalping Strategy — 3 EMA + Stochastic RSI + ATR

**youtube link** - https://www.youtube.com/watch?v=RQpu6bJBUaQ


This strategy implements a scalping system using EMA trend alignment, Stochastic RSI momentum signals, and ATR-based risk management.
Trades execute on the next candle open after a valid signal.

## 📌 Indicators Used
1. Exponential Moving Averages (EMA)

EMA 7 → Fast trend
EMA 13 → Mid-term trend
EMA 400 → Long-term trend filter

2. Stochastic RSI (15-period on HLC3)

RSI calculated on HLC3 = (High + Low + Close) / 3
Stochastic (%K and %D) computed on RSI values
Used for momentum crossovers

3. ATR (Average True Range — 16 period)

Used for volatility-based Stop Loss (SL) and Take Profit (TP):
TP = 1.9 × ATR(16)
SL = ATR × 1.9 × 1.57

## 📈 Strategy Rules
⏱️ Timeframe - 15 Min candles

**🟩 LONG TRADE (BUY)**
Enter a long position only when all conditions are true:

1. EMA Alignment (Bullish Trend): EMA 400 < EMA 13 < EMA 7
2. Stochastic RSI Bullish Cross: %K crosses above %D
(prev %K < prev %D) and (current %K > current %D)

**Entry:** Next candle after conditions met.  


**🟩 Long Exit Conditions**
Exit a long trade when any of the following occur:

1. Take Profit Hit
TP = Entry Price + (1.9 × ATR)

2. Stop Loss Hit
SL = Entry Price – (ATR × 1.9 × 1.57)

## 🟥 SHORT TRADE (SELL)
Enter a short position only when all conditions are true:

**✅ Entry Conditions**

1. EMA Alignment (Bearish Trend): EMA 7 < EMA 13 < EMA 400
2. Stochastic RSI Bearish Cross: %K crosses below %D
(prev %K > prev %D) and (current %K < current %D)

**Entry:** Next candle after conditions met.  


**🟥 Short Exit Conditions**
Exit a short trade when any of the following occur:

1. Take Profit Hit
TP = Entry Price – (1.9 × ATR)

2. Stop Loss Hit
SL = Entry Price + (ATR × 1.9 × 1.57)
