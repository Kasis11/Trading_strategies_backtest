import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, List, Dict
import os
import quantstats as qs
import matplotlib.pyplot as plt

class FiveEMAStrategy:
  """
  Original 5 EMA Strategy (Simplified) Implementation.
  - Uses 1D 20 EMA as the TREND FILTER (the Boss).
  - Uses 5M 5 EMA for BOTH Buy and Sell Pullback Entries.
  - Multi-timeframe merging of 15m, 1h, 4h is removed from the core logic 
   to stick to the original two-timeframe (1D + 5m) principle.
  """

  def __init__(
    self,
    rr_ratio: float = 3.0,
    base_qty: int = 1,
    aggressive_qty: int = 3,
    gap_threshold_pct: float = 0.005
  ):
    self.rr_ratio = rr_ratio
    self.base_qty = base_qty
    self.aggressive_qty = aggressive_qty
    self.gap_threshold_pct = gap_threshold_pct

    self.position: Optional[str] = None
    self.pending_signal: Optional[str] = None
    self.signal_candle: Optional[pd.Series] = None
    self.entry_price = None
    self.sl = None
    self.tp = None
    self.qty = None

    self.trades: List[Dict] = []
    
  def _reset(self):
    """Helper function to reset position tracking variables."""
    self.position = None
    self.entry_price = None
    self.sl = None
    self.tp = None
    self.qty = None
    self.signal_candle = None

  @staticmethod
  def pick_contract(dt: pd.Timestamp) -> str:
    # Standard logic for picking future contract month
    year = dt.year
    month = dt.month
    for m in (3, 6, 9, 12):
      if month <= m:
        return f"{year}-{m:02d}"
    return f"{year + 1}-03"

  def prepare_data(
    self, 
    df_5m: pd.DataFrame, 
    df_daily: pd.DataFrame
  ) -> pd.DataFrame:
    """
    Prepares data for the simple 5 EMA strategy:
    1. Calculates 5 EMA on the execution chart (5m).
    2. Calculates 1D 20 EMA as the trend filter proxy and merges it into the 5m data.
    """
    df_5m = df_5m.copy()
    df_daily = df_daily.copy()
  
    # 1. Intraday EMA (Execution)
    df_5m["ema5"] = ta.ema(df_5m["Close"], length=5)
    
    # 2. Daily Trend Filter Proxy
    # Using 20 EMA on 1D as an objective proxy for visual HH/HL trend confirmation.
    df_daily["ema20_1d"] = ta.ema(df_daily["Close"], length=20) 
    df_1d_cols = ["Datetime", "ema20_1d"]
    
    # Merge 1D data into 5M data
    df_5m = pd.merge_asof(
      df_5m.sort_values("Datetime"),
      df_daily[df_1d_cols].sort_values("Datetime"),
      on="Datetime",
      direction="backward",
      suffixes=("_5m", "_1d")
    ).rename(columns={'ema20_1d': 'ema20_1d_merged'})
  
    # Retaining Range Shift and Gap Logic (though optional for core theory, useful for backtest)
    lookback = 20
    df_5m["rolling_high"] = df_5m["High"].rolling(lookback).max()
    df_5m["rolling_low"] = df_5m["Low"].rolling(lookback).min()
  
    return df_5m
  
  def extract_gap_levels(self, df_daily: pd.DataFrame) -> np.ndarray:
    # 1D is used for context (Gap Levels) - remains for risk management
    df = df_daily.copy().sort_values("Datetime")
    df["prev_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
    gaps = df.loc[df["gap_pct"].abs() >= self.gap_threshold_pct]
    return gaps["Open"].dropna().values

  @staticmethod
  def near_level(price: float, levels: np.ndarray, tolerance: float = 0.003) -> bool:
    if len(levels) == 0:
      return False
    return np.any(np.abs(levels - price) / price <= tolerance)

  @staticmethod
  def detect_range_shift(df: pd.DataFrame, i: int, lookback: int = 20) -> str:
    if i < lookback:
      return "none"
    highs = df["High"].iloc[i - lookback:i]
    lows = df["Low"].iloc[i - lookback:i]
    if lows.iloc[-1] < lows[:-1].min():
      return "bearish"
    if highs.iloc[-1] > highs[:-1].max():
      return "bullish"
    return "none"

  @staticmethod
  def fully_above_ema(row, ema_col):
    # Checks if the entire candle body is above the EMA (High > EMA AND Low > EMA)
    return pd.notna(row[ema_col]) and row["Low"] > row[ema_col]

  @staticmethod
  def fully_below_ema(row, ema_col):
    # Checks if the entire candle body is below the EMA (High < EMA AND Low < EMA)
    return pd.notna(row[ema_col]) and row["High"] < row[ema_col]
  
  def run_backtest(
    self,
    df_5m: pd.DataFrame,
    df_daily: pd.DataFrame,
  ) -> pd.DataFrame:

    # Only pass 5m and 1D data to prepare_data now
    df_5m = self.prepare_data(df_5m, df_daily) 
    gap_levels = self.extract_gap_levels(df_daily)

    if 'ema20_1d_merged' not in df_5m.columns:
      print("CRITICAL ERROR: 'ema20_1d_merged' column is missing after prepare_data. Check merge logic.")
      return pd.DataFrame(self.trades)

    for i in range(2, len(df_5m)):
      row = df_5m.iloc[i]
      prev = df_5m.iloc[i - 1]

      # Skip iteration if the daily trend filter value is NaN
      if pd.isna(row["ema20_1d_merged"]):
        continue
      
      #  EXIT LOGIC (Remains the same R:R exit) 
      
      if self.position:
        trade = self.trades[-1]
        # SELL Exit Logic
        if self.position == "SELL":
          if row["High"] >= self.sl: # Stop Loss Hit
            trade.update({"Exit_Time": row["Datetime"], "Exit": self.sl, "PnL": (trade["Entry"] - self.sl) * self.qty})
            self._reset()
            continue
          elif row["Low"] <= self.tp: # Target Hit
            trade.update({"Exit_Time": row["Datetime"], "Exit": self.tp, "PnL": (trade["Entry"] - self.tp) * self.qty})
            self._reset()
            continue
        # BUY Exit Logic
        if self.position == "BUY":
          if row["Low"] <= self.sl: # Stop Loss Hit
            trade.update({"Exit_Time": row["Datetime"], "Exit": self.sl, "PnL": (self.sl - trade["Entry"]) * self.qty})
            self._reset()
            continue
          elif row["High"] >= self.tp: # Target Hit
            trade.update({"Exit_Time": row["Datetime"], "Exit": self.tp, "PnL": (self.tp - trade["Entry"]) * self.qty})
            self._reset()
            continue

      
      #  EXECUTE PENDING ENTRY (Remains the same SL/TP calculation) 
      
      if self.pending_signal and not self.position:
        entry = row["Open"]
        # SL is the High/Low of the signal candle (prev)
        risk = max(abs(entry - self.signal_candle["High" if self.pending_signal == "SELL" else "Low"]),1e-5)

        self.sl = (
          self.signal_candle["High"] if self.pending_signal == "SELL"
          else self.signal_candle["Low"]
        )
        self.tp = (
          entry - self.rr_ratio * risk if self.pending_signal == "SELL"
          else entry + self.rr_ratio * risk
        )

        # Use 1D Gaps and 5m Range Shift for dynamic quantity
        near_gap = self.near_level(entry, gap_levels)
        range_shift = self.detect_range_shift(df_5m, i)

        self.qty = (
          self.aggressive_qty
          if near_gap or
          (range_shift == "bearish" and self.pending_signal == "SELL") or
          (range_shift == "bullish" and self.pending_signal == "BUY")
          else self.base_qty
        )

        self.trades.append({
          "Entry_Time": row["Datetime"], "Side": self.pending_signal, "Entry": entry, "SL": self.sl,
          "TP": self.tp, "Qty": self.qty, "Exit_Time": None, "Exit": None, "PnL": None,
          "Contract": self.pick_contract(row["Datetime"]), "Near_Gap": near_gap, "Range_Shift": range_shift
        })

        self.position = self.pending_signal
        self.pending_signal = None
        self.signal_candle = None
        continue

      
      #  SIGNAL GENERATION: PURE ORIGINAL 5 EMA LOGIC 
      
      if not self.position and not self.pending_signal:
        
        # 1. TREND FILTER: Daily Trend Proxy (Original Theory's "Boss")
        daily_trend_bullish = row["Close"] > row["ema20_1d_merged"]
        daily_trend_bearish = row["Close"] < row["ema20_1d_merged"]
        
        # SELL → 5 EMA Pullback (Requires Daily Bearish Trend)
        if (
          daily_trend_bearish 
          and self.fully_above_ema(prev, "ema5") # Candle fully above the 5 EMA
          and row["Low"] < prev["Low"]     # Low of current candle breaks below low of previous candle
        ):
          self.pending_signal = "SELL"
          self.signal_candle = prev

        # BUY → 5 EMA Pullback (Requires Daily Bullish Trend)
        elif (
          daily_trend_bullish 
          and self.fully_below_ema(prev, "ema5") # Candle fully below the 5 EMA
          and row["High"] > prev["High"]     # High of current candle breaks above high of previous candle
        ):
          self.pending_signal = "BUY"
          self.signal_candle = prev


    return pd.DataFrame(self.trades)

  
  #  METRICS, EQUITY CURVE, PLOTTING (No changes needed) 
  
  @staticmethod
  def evaluate_metrics(trades_df: pd.DataFrame):
    trades = trades_df.copy()
    trades = trades.dropna(subset=["PnL"])
    total_trades = len(trades)
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    avg_pnl = trades["PnL"].mean() if total_trades > 0 else 0
    net_pnl = trades["PnL"].sum()
    
    cumulative_pnl = trades["PnL"].cumsum()
    max_drawdown = (cumulative_pnl.cummax() - cumulative_pnl).max() if len(cumulative_pnl) > 0 else 0
    
    return {
      "Total Trades": total_trades,
      "Winning Trades": len(wins),
      "Losing Trades": len(losses),
      "Win Rate (%)": round(win_rate, 2),
      "Average PnL": round(avg_pnl, 2),
      "Net PnL": round(net_pnl, 2),
      "Max Drawdown (PnL-based)": round(max_drawdown, 2)
    }

  @staticmethod
  def build_equity_curve(trades: pd.DataFrame, initial_capital=100000):
    trades = trades.dropna(subset=["Exit_Time", "PnL"]).copy()
    trades = trades.sort_values("Exit_Time")

    equity = initial_capital
    records = []

    for _, row in trades.iterrows():
      equity += row["PnL"]
      records.append({
        "Date": row["Exit_Time"].floor("D"),
        "Equity": equity
      })

    eq_df = pd.DataFrame(records).groupby("Date").last()
    eq_df = eq_df.asfreq("D")
    eq_df["Equity"] = eq_df["Equity"].ffill().fillna(initial_capital)

    return eq_df

  @staticmethod
  def run_quantstats(tradebook, output_file):
    tb = tradebook.dropna(subset=["Exit_Time", "PnL"]).copy()

    eq = FiveEMAStrategy.build_equity_curve(tb, initial_capital=100000)

    returns = eq["Equity"].pct_change().fillna(0.0)

    qs.reports.html(
      returns,
      output=output_file,
      title="5 EMA Strategy (Equity-Based, Realistic)",
      benchmark=None
    )

    print(f"QuantStats report generated → {output_file}")

  @staticmethod
  def plot_5ema_signals(trades: pd.DataFrame, df_5m: pd.DataFrame, start_year=2010, end_year=2016):

    trades_period = trades[(trades['Exit_Time'].dt.year >= start_year) & 
               (trades['Exit_Time'].dt.year <= end_year)]

    plt.figure(figsize=(16,6))

    plt.plot(df_5m['Datetime'], df_5m['Close'], label='Market Close', color='blue', alpha=0.5)

    buy_trades = trades_period[trades_period['Side'] == 'BUY']
    plt.scatter(buy_trades['Entry_Time'], buy_trades['Entry'], marker='^', color='green', s=50, label='Buy Entry')

    sell_trades = trades_period[trades_period['Side'] == 'SELL']
    plt.scatter(sell_trades['Entry_Time'], sell_trades['Entry'], marker='v', color='red', s=50, label='Sell Entry')

    plt.scatter(trades_period['Exit_Time'], trades_period['Exit'], marker='o', color='black', s=30, label='Exit')

    plt.title(f'5 EMA Strategy Signals {start_year}–{end_year}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


  
if __name__ == "__main__":
  BASE_DIR = os.path.dirname(os.path.abspath(__file__))

  # NOTE: The code now only needs 5Min and 1D data.
  DATA_PATH_5M = os.path.join(BASE_DIR, "..", "data", "backtest_data_5Min.csv")
  DATA_PATH_1D = os.path.join(BASE_DIR, "..", "data", "backtest_data_1D.csv")  
  
  # # We still define the other paths, but they are no longer loaded for the core logic
  # DATA_PATH_15M = os.path.join(BASE_DIR, "..", "data", "backtest_data_15Min.csv") 
  # DATA_PATH_1H = os.path.join(BASE_DIR, "..", "data", "backtest_data_1H.csv") 
  # DATA_PATH_4H = os.path.join(BASE_DIR, "..", "data", "backtest_data_4H.csv") 
  

  def load_csv(path: str, tf_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
      raise FileNotFoundError(f"{tf_name} data file not found:\n{path}")

    df = pd.read_csv(path)

    if "Datetime" not in df.columns:
      possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
      if not possible:
        raise ValueError(f"{tf_name} file has no Datetime column")
      df.rename(columns={possible[0]: "Datetime"}, inplace=True)

    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(df.columns):
      raise ValueError(
        f"{tf_name} missing columns: {required - set(df.columns)}"
      )

    return df

  df_5m = load_csv(DATA_PATH_5M, "5-Minute")
  df_1d = load_csv(DATA_PATH_1D, "Daily")

  print("\nRunning Original 5 EMA Strategy backtest (1D Trend + 5M Entry)...")

  strategy = FiveEMAStrategy(
    rr_ratio=3.0,
    base_qty=1,
    aggressive_qty=3,
    gap_threshold_pct=0.005
  )

  trades = strategy.run_backtest(df_5m, df_1d) 
  trades = trades.dropna(subset=["Exit"]).reset_index(drop=True)

  # ------------------ OUTPUT PATHS ------------------
  BACKTEST_DIR = os.path.join(BASE_DIR, "..", "backtest")
  QS_DIR = os.path.join(BACKTEST_DIR, "output_quant_s")

  os.makedirs(BACKTEST_DIR, exist_ok=True)
  os.makedirs(QS_DIR, exist_ok=True)

  # Save tradebook
  tradebook_path = os.path.join(BACKTEST_DIR, "5ema_tradebook.csv")
  trades.to_csv(tradebook_path, index=False)

  metrics = FiveEMAStrategy.evaluate_metrics(trades)
  print("\n ORIGINAL 5 EMA STRATEGY METRICS")
  for k, v in metrics.items():
    print(f"{k:20s}: {v}")


  # QuantStats output
  qs_output = os.path.join(QS_DIR, "5ema_quantstats.html")
  FiveEMAStrategy.run_quantstats(trades, output_file=qs_output)
  FiveEMAStrategy.plot_5ema_signals(trades, df_5m, start_year=2010, end_year=2025)