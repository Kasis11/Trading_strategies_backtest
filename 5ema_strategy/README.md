# Modified 5 EMA Strategy — Power of Stocks (Subasish)

**YouTube Reference** - https://www.youtube.com/watch?v=fBTMspiqjS4

This strategy is a **price-action–driven EMA pullback system** designed to capture **large reward-to-risk trades (1:3)** while accepting **frequent small stop losses**.

## Timeframes Used
Execution - **5 Minutes (5M)** 
Trend Context - **Daily (1D)** 


## Indicator Used

### 1️⃣ 5 EMA (Execution EMA)

- Used for **pullback and reversal entries**
- Candle must **not touch** the EMA

**Parameters**
- Length: `5`

---

### 2️⃣ Daily 20 EMA (Trend Filter – “Boss”)

- Acts as **higher-timeframe trend confirmation**
- Trades are allowed **only in the direction of the daily trend**

**Parameters**
- Length: `20`

## Strategy Rules
### Candle Qualification Rule (MANDATORY)

A candle is considered **valid** only if:

- **SELL setup:** Candle closes **fully above** 5 EMA  
- **BUY setup:** Candle closes **fully below** 5 EMA  
- Candle **must not touch EMA** (no wick or body contact)

If this rule fails → **No trade**


## Entry Logic

### 🔴 SELL Trade (5 EMA Pullback)

**Conditions**
- Daily trend is **bearish** (Close < Daily 20 EMA)
- Previous 5M candle closes **fully above 5 EMA**
- Current candle **breaks below the previous candle’s Low**

**Entry Price**
- Open of the breakdown candle


### Exit Logic

#### ❌ Stop Loss
- Placed at the **High of the signal candle**

#### Target =  Entry − (Risk × R:R)
- Fixed Risk-Reward based target  
- Default: **1:3**


### 🔵 BUY Trade (5 EMA Pullback)

**Conditions**
- Daily trend is **bullish** (Close > Daily 20 EMA)
- Previous 5M candle closes **fully below 5 EMA**
- Current candle **breaks above the previous candle’s High**

**Entry Price**
- Open of the breakout candle

# Exit Logic

#### ❌ Stop Loss
- Placed at the **Low of the signal candle**

#### 🎯 Target = Entry + (Risk × R:R)
- Fixed Risk-Reward based target  
- Default: **1:3**

## Position Sizing Rules (MOST IMPORTANT)

Trades are **never skipped**.  
Only **quantity is adjusted** based on probability.


## High-Probability Conditions (Increase Quantity)

### 1️⃣ Gap Areas (Daily Chart)

- Gaps act as **strong support or resistance**
- If price approaches a gap and:
  - 5 EMA setup appears
- Trade with **aggressive quantity = 3** 


### 2️⃣ “Range Shift 

**Bearish Range Shift**
- Market was making higher highs
- Starts making lower lows
- First 5 EMA pullback → aggressive SELL

**Bullish Range Shift**
- Market breaks resistance
- Previous resistance becomes support
- First pullback → aggressive BUY
