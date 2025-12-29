# Ichimoku Cloud Trading Strategy

**youtube link** - https://www.youtube.com/watch?v=EumlRRIx0WA

This implements a **4-hour Ichimoku Cloud trading strategy** in Python. The strategy uses Ichimoku components to identify trends, generate trading signals, and manage risk with ATR-based exits.  

## Key Ichimoku Components

1. **Tenkan-sen (Conversion Line)**  
   - Average of highest high and lowest low over 9 periods (short-term trend).  
   - Sloping upward → bullish momentum  
   - Sloping downward → bearish momentum  

2. **Kijun-sen (Base Line)**  
   - Average of highest high and lowest low over 26 periods (medium-term trend).  
   - Price above Kijun-sen → uptrend  
   - Price below Kijun-sen → downtrend  

3. **Chikou Span (Lagging Span)**  
   - Current closing price plotted 26 periods back.  
   - Above past price → confirms uptrend  
   - Below past price → confirms downtrend  

4. **Senkou Span A (Leading Span A)**  
   - Average of Tenkan-sen and Kijun-sen plotted 26 periods into the future.  
   - Forms one boundary of the Ichimoku cloud.  

5. **Senkou Span B (Leading Span B)**  
   - Average of highest high and lowest low over 52 periods, plotted 26 periods forward.  
   - Slower than Senkou Span A, forms the other boundary of the cloud.  

- **Cloud** = space between Senkou Span A & B, used to identify trend direction and support/resistance.

---

## Strategy Rules

### Timeframe: 4-Hour Candles

---

### 1️⃣ Long Trade (Buy) Conditions

Enter a **long position** when **all** the following are true:

1. Tenkan-sen (Conversion Line) **crosses above** Kijun-sen (Base Line) → bullish momentum.  
2. Price is **above the Ichimoku Cloud**.  
3. Tenkan-sen and Kijun-sen are **sloping upward or flat** (trend confirmation).  

**Entry:** Next candle after conditions met.  

**Exit Condition:**  
- Exit when **price drops below Kijun-sen by 2 ATR** (to avoid premature stop-outs).

---

### 2️⃣ Short Trade (Sell) Conditions

Enter a **short position** when **all** the following are true:

1. Tenkan-sen **crosses below** Kijun-sen → bearish momentum.  
2. Price is **below the Ichimoku Cloud**.  
3. Tenkan-sen and Kijun-sen are **sloping downward or flat**.  

**Entry:** Next candle after conditions met.  

**Exit Condition:**  
- Exit when **price moves above Kijun-sen by 2 ATR**.

---

## ATR & Risk Management

- **ATR (Average True Range)** is used to calculate volatility-based stop-loss.  
- Exit trades only when price moves ±2 ATR from the Kijun-sen, allowing trades to ride strong trends.  
