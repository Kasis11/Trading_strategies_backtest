# Multi-Entry Scalping Strategy — KRI + Squeeze Momentum 

**youtube link** - https://www.youtube.com/watch?v=UydcLDCCuUY


This strategy implements a **high-confluence scalping system** using **dual momentum confirmation** (Kairi Relative Index and Squeeze Momentum) combined with **strict trend, momentum, volatility, and risk filters**.

A trade is executed **only when both momentum signals agree within a defined tolerance window and all filters are satisfied**.  

## Indicators Used

### 1️. Kairi Relative Index (KRI)

- Measures percentage deviation of price from its SMA
- Used for **momentum direction**
- Zero-line crossover generates signals

**Parameters**
- Length: `21`

### 2️. Squeeze Momentum (Aroon Oscillator Proxy)

- Uses Aroon Oscillator as a proxy for squeeze momentum
- Identifies expansion and contraction phases
- Zero-line crossover confirms momentum direction

**Parameters**
- Length: `21`

### 3️. 200 Simple Moving Average (Trend Filter)

- Identifies long-term trend direction
- Trades are allowed **only in the direction of the 200 SMA**

**Parameters**
- Length: `200`

### 4️. Chandelier Momentum Oscillator (CMO) + EMA

- Used as a momentum confirmation filter
- CMO must be aligned with its EMA

**Parameters**
- CMO Length: `50`
- CMO EMA Length: `50`

### 5️. Average Directional Index (ADX)

- Measures trend strength
- Ensures trades occur only in **strong trending conditions**

**Parameters**
- ADX Length: `5`
- Minimum ADX Value: `20`


### 6️. Average True Range (ATR)

- Used for **initial stop loss**
- Also used by SuperTrend for trailing stop

**Parameters**
- ATR Length: `8`
- Multiplier: `2.5`

### 7️. SuperTrend (Trailing Stop)

- Acts as a **dynamic trailing stop**
- Takes priority over initial stop loss

**Parameters**
- Length: `8`
- Multiplier: `2.5`

## Strategy Rules

**Timeframe**  
- **1 Hour (1H) candles**

## Entry Signal Agreement (Mandatory)

A trade is considered **only when BOTH momentum signals agree within a 12-candle window**.

### Kairi Relative Index (KRI)
- **BUY:** KRI crosses **above 0**
- **SELL:** KRI crosses **below 0**

### Squeeze Momentum
- **BUY:** Squeeze line crosses **above 0**
- **SELL:** Squeeze line crosses **below 0**

### Signal Timing Rules
- Both signals must occur **within 12 candles of each other**
- Signals outside this window are **discarded**
- Older signals are automatically removed from consideration

## Entry Confirmation Filters (ALL Must Pass)

After signal agreement, **all seven filters must be satisfied** for a trade to be executed.

### 1️. Trend Filter — 200 SMA
- **BUY:** Close price is **above** the 200 SMA
- **SELL:** Close price is **below** the 200 SMA

### 2️. Momentum Filter — CMO
- **BUY:** CMO is **above** its 50-period EMA
- **SELL:** CMO is **below** its 50-period EMA

### 3️. Candle Height Filter — Minimum Range
- Candle High-Low range must be **greater than 0.1%** of the Low price
- Avoids very small or stagnant candles

### 4️. Candle Height Filter — Maximum Range
- Candle High-Low range must be **less than 1.3%** of the Low price
- Avoids excessively volatile candles

### 5️. Trend Strength Filter — ADX
- ADX value must be **greater than 20**
- Ensures trades occur only during strong trends

### 6️. Abnormal Movement Filter
- The **maximum candle height** of the **previous 5 candles** must be **less than 0.8%**
- Prevents entries after sudden spikes or abnormal moves

### 7️⃣ Day-of-Week Filter
- Trades are **not allowed on Saturdays**
- Avoids low-liquidity and irregular sessions

## LONG TRADE (BUY)

### Entry Conditions
- KRI crosses above 0
- Squeeze Momentum crosses above 0
- Both signals occur within 12 candles
- **All 7 filters pass**

**Entry Price:**  
- Close price of the signal candle

## SHORT TRADE (SELL)

### Entry Conditions
- KRI crosses below 0
- Squeeze Momentum crosses below 0
- Both signals occur within 12 candles
- **All 7 filters pass**

**Entry Price:**  
- Close price of the signal candle

## ❌ Exit Conditions

A trade exits when **any one** of the following is triggered:

### 1️. Initial Stop Loss (Global SL)

for **BUY** - Entry − (2.5 × ATR(8))
for **SELL** - Entry + (2.5 × ATR(8)) 


### 2️⃣ Trailing Stop (SuperTrend)

for **BUY** - Low ≤ SuperTrend Line 
for **SELL** - High ≥ SuperTrend Line

**Important Rule:**  
The **SuperTrend trailing stop takes precedence** over the initial stop loss, allowing tighter exits and reduced drawdowns.
