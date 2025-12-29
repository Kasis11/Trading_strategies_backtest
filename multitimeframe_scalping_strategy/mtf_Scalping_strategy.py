import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from typing import Optional, List, Dict, Tuple
import datetime
import quantstats as qs 

KRI_LENGTH = 21
SQUEEZE_MOM_LENGTH = 21
SIGNAL_TOLERANCE = 6
ATR_LENGTH = 10
ATR_MULTIPLIER = 3.5
# INITIAL_SL_MULTIPLIER = 2.0   
SMA_LENGTH = 200
CMO_LENGTH = 50
ADX_LENGTH = 5
ADX_SMOOTHING = 5
ADX_LOW_LIMIT = 30
CANDLE_HEIGHT_LOW_LIMIT = 0.001
CANDLE_HEIGHT_HIGH_LIMIT = 0.007
ABNORMAL_MOVEMENT_LIMIT = 0.008

class MultiEntryScalpingStrategy:

    def __init__(
        self,
        initial_capital: float = 100000.0,
        signal_tolerance: int = SIGNAL_TOLERANCE,
         ):
        self.signal_tolerance = signal_tolerance
        self.capital = initial_capital 
        self._reset()
        self.trades: List[Dict] = []
        self.kri_signals: List[Dict] = []
        self.squeeze_signals: List[Dict] = []
        
    def _reset(self):
        self.position: Optional[str] = None
        self.entry_time = None
        self.entry_price = None
        self.initial_sl = None 
        self.qty = None
        
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        df["kri_sma"] = ta.sma(df["Close"], length=KRI_LENGTH)
        df["kri"] = ((df["Close"] - df["kri_sma"]) / df["kri_sma"]) * 100
        
        df["squeeze"] = ta.aroon(df["High"], df["Low"], length=SQUEEZE_MOM_LENGTH)[f"AROONOSC_{SQUEEZE_MOM_LENGTH}"]
        
        df["sma200"] = ta.sma(df["Close"], length=SMA_LENGTH)
        
        df["cmo"] = ta.cmo(df["Close"], length=CMO_LENGTH)
        df["cmo_ema"] = ta.ema(df["cmo"], length=CMO_LENGTH)
        
        df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=ATR_LENGTH)
        
        df["sl_buy"] = df["Close"] - df["atr"] * ATR_MULTIPLIER
        df["sl_sell"] = df["Close"] + df["atr"] * ATR_MULTIPLIER
        
        st = ta.supertrend(df["High"], df["Low"], df["Close"], length=ATR_LENGTH, multiplier=ATR_MULTIPLIER)
        df["st_line"] = st[f"SUPERT_{ATR_LENGTH}_{ATR_MULTIPLIER}"]
        
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=ADX_LENGTH, mamode="ema") # mamode="ema" mimics the smoothing 
        df["adx"] = adx[f"ADX_{ADX_LENGTH}"]
        
        df["candle_height_pct"] = (df["High"] - df["Low"]) / df["Low"]
        df["prev_5_max_height"] = df["candle_height_pct"].shift(1).rolling(5).max()
        
        return df

    def _check_kri_signal(self, i: int, df: pd.DataFrame):
        if i < 1: return None
        prev = df.iloc[i-1]; curr = df.iloc[i]
        if prev["kri"] <= 0 and curr["kri"] > 0: return "BUY"
        elif prev["kri"] >= 0 and curr["kri"] < 0: return "SELL"
        return None
    
    # Squeeze Momentum signal: Crossover zero line
    def _check_squeeze_signal(self, i: int, df: pd.DataFrame):
        if i < 1: return None
        prev = df.iloc[i-1]; curr = df.iloc[i]
        if prev["squeeze"] <= 0 and curr["squeeze"] > 0: return "BUY"
        elif prev["squeeze"] >= 0 and curr["squeeze"] < 0: return "SELL"
        return None
    
    def _check_all_filters(self, i: int, df: pd.DataFrame, side: str, debug: bool = False) -> Tuple[bool, Optional[str]]:
        if i < max(SMA_LENGTH, CMO_LENGTH, ADX_SMOOTHING, 6): return (False, "INSUFFICIENT_LOOKBACK") if debug else (False, None)
        row = df.iloc[i]
        
        sma_ok = (side == "BUY" and row["Close"] > row["sma200"]) or \
            (side == "SELL" and row["Close"] < row["sma200"])
        if not sma_ok: return (False, "FILTER_SMA") if debug else (False, None)
        
        cmo_ok = (side == "BUY" and row["cmo"] > row["cmo_ema"]) or \
            (side == "SELL" and row["cmo"] < row["cmo_ema"])
        if not cmo_ok: return (False, "FILTER_CMO") if debug else (False, None)
        
        height = row["candle_height_pct"]
        height_ok = (height > CANDLE_HEIGHT_LOW_LIMIT) and \
              (height < CANDLE_HEIGHT_HIGH_LIMIT)
        if not height_ok: return (False, "FILTER_CANDLE_HEIGHT") if debug else (False, None)
        
        adx_ok = row["adx"] > ADX_LOW_LIMIT
        if not adx_ok: return (False, "FILTER_ADX") if debug else (False, None)
        
        abnormal_ok = row["prev_5_max_height"] < ABNORMAL_MOVEMENT_LIMIT
        if not abnormal_ok: return (False, "FILTER_ABNORMAL_MOVEMENT") if debug else (False, None)
        
        dow_ok = row["Datetime"].dayofweek != 5 
        if not dow_ok: return (False, "FILTER_DOW") if debug else (False, None)
        
        return (True, None) if debug else (True, None)

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.prepare_data(df)
        df = df.dropna().reset_index(drop=True)
        
        for i in range(1, len(df) - 1): 
            row = df.iloc[i]
            next_row = df.iloc[i+1]
            
            active_st_line = df.iloc[i-1]["st_line"] 
            
            # EXIT LOGIC  
            if self.position:
                trade = self.trades[-1]
    
                st_hit = (self.position == "BUY" and row["Low"] <= active_st_line)
                st_hit = st_hit or (self.position == "SELL" and row["High"] >= active_st_line)
    
                sl_hit = (self.position == "BUY" and row["Low"] <= self.initial_sl)
                sl_hit = sl_hit or (self.position == "SELL" and row["High"] >= self.initial_sl)
    
                if st_hit or sl_hit:
                    
                    exit_price = None
                    exit_time = None
                    
                    if sl_hit:
                        exit_price = self.initial_sl 
                        exit_time = row["Datetime"]
                    
                    elif st_hit:
                        exit_price = next_row["Open"]
                        exit_time = next_row["Datetime"]
                    
                    pnl = (trade["Entry"] - exit_price) * self.qty if self.position == "SELL" else \
                       (exit_price - trade["Entry"]) * self.qty 
    
                    trade.update({"Exit_Time": exit_time, "Exit": exit_price, "PnL": pnl})
                    self.capital += pnl
                    self._reset()
                    continue
        

            kri_sig = self._check_kri_signal(i, df)
            squeeze_sig = self._check_squeeze_signal(i, df)
            
            if kri_sig: self.kri_signals.append({"time": row["Datetime"], "side": kri_sig})
            if squeeze_sig: self.squeeze_signals.append({"time": row["Datetime"], "side": squeeze_sig})

            timeframe_in_minutes = 60 
            tolerance_start = row["Datetime"] - pd.Timedelta(minutes=timeframe_in_minutes * self.signal_tolerance)
            self.kri_signals = [s for s in self.kri_signals if s["time"] >= tolerance_start]
            self.squeeze_signals = [s for s in self.squeeze_signals if s["time"] >= tolerance_start]
            
            # ENTRY LOGIC
            if not self.position:
                
                buy_agreement = any(s1["time"] for s1 in self.kri_signals if s1["side"] == "BUY") and \
                        any(s2["time"] for s2 in self.squeeze_signals if s2["side"] == "BUY")
                        
                sell_agreement = any(s1["time"] for s1 in self.kri_signals if s1["side"] == "SELL") and \
                        any(s2["time"] for s2 in self.squeeze_signals if s2["side"] == "SELL")
                        
                signal_side = None
                if buy_agreement: signal_side = "BUY"
                if sell_agreement and not buy_agreement: signal_side = "SELL" 
                
                if signal_side:
                    entry_price = row["Close"]
                
                    if self._check_all_filters(i, df, signal_side):
                        self.position = signal_side
                        self.entry_time = row["Datetime"]
                        self.entry_price = entry_price
                        self.initial_sl = row["sl_buy"] if signal_side == "BUY" else row["sl_sell"]
                        self.qty = 1.0

                        self.trades.append({
                            "Entry_Time": self.entry_time, "Side": self.position, "Entry": self.entry_price, 
                            "Initial_SL": self.initial_sl, "Qty": self.qty, "Capital_Start": self.capital,
                            "Exit_Time": None, "Exit": None, "PnL": None,
                        })
                        
                        self.kri_signals = []
                        self.squeeze_signals = []
                        
        return pd.DataFrame(self.trades)
    
def load_csv(path: str, tf_name: str) -> pd.DataFrame:
    """Loads and preprocesses the OHLCV data."""
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
    df.columns = [c.capitalize() if c.lower() in [r.lower() for r in required] else c for c in df.columns]
    if not required.issubset(df.columns):
        raise ValueError(
            f"{tf_name} missing columns: {required - set(df.columns)}"
        )
    df = df[["Datetime", "Open", "High", "Low", "Close"]].copy() 
    return df


def _build_equity_curve(tradebook: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """
    Calculates the equity curve from the trade PnL for use in QuantStats.
    It resamples the data to daily frequency as QuantStats usually expects daily returns.
    """
    if tradebook.empty or "PnL" not in tradebook.columns:
        return pd.DataFrame({"Equity": [initial_capital]}, index=[datetime.datetime.now().date()])

    equity_df = tradebook.set_index("Exit_Time")["PnL"].cumsum() + initial_capital
    equity_df = equity_df.resample('D').ffill()
    
    first_exit_date = equity_df.index.min()
    if first_exit_date is not pd.NaT:
        start_date = equity_df.index.min() - pd.Timedelta(days=1)
        if start_date < equity_df.index.min():
            equity_df.loc[start_date] = initial_capital
            equity_df = equity_df.sort_index()

    return pd.DataFrame({"Equity": equity_df})

def run_quantstats(tradebook: pd.DataFrame, output_file: str, initial_capital: float = 100000.0):

    tb = tradebook.dropna(subset=["Exit_Time", "PnL"]).copy()
    
    if tb.empty:
        print("⚠️ No closed trades found to run QuantStats report.")
        return

    eq = _build_equity_curve(tb, initial_capital=initial_capital)

    returns = eq["Equity"].pct_change().fillna(0.0)

    returns.index.name = 'Date'
    
    qs.reports.html(
        returns,
        output=output_file,
        title="Multi-Entry Scalping Strategy",
        benchmark=None
    )

    print(f"✅ QuantStats report generated to: {output_file}")


def evaluate_metrics(trades_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict:
    """Calculates key trading performance metrics."""
    trades = trades_df.copy().dropna(subset=["PnL"]) 
    
    total_trades = len(trades)
    if total_trades == 0:
        return {"Total Trades": 0, "Net PnL": 0, "Profit Factor": 0, "Max Drawdown (%)": 0}

    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    
    total_gross_profit = wins["PnL"].sum()
    total_gross_loss = abs(losses["PnL"].sum())
    net_pnl = total_gross_profit - total_gross_loss
    
    win_rate = len(wins) / total_trades * 100
    profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else np.inf
    avg_win = wins["PnL"].mean() if len(wins) > 0 else 0
    avg_loss = losses["PnL"].mean() if len(losses) > 0 else 0
    
    trades["Equity"] = trades["PnL"].cumsum() + initial_capital
    trades["Peak"] = trades["Equity"].cummax()
    trades["Drawdown"] = trades["Peak"] - trades["Equity"]
    max_drawdown_amount = trades["Drawdown"].max()
    max_drawdown_percent = (max_drawdown_amount / trades["Peak"].max()) * 100 if trades["Peak"].max() > 0 else 0
    
    return {
        "Total Trades": total_trades,
        "Net PnL": round(net_pnl, 2),
        "Final Equity": round(trades["Equity"].iloc[-1], 2),
        "Winning Trades": len(wins),
        "Losing Trades": len(losses),
        "Win Rate (%)": round(win_rate, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "Inf", 
        "Max Drawdown ": round(max_drawdown_amount, 2),
        "Max Drawdown (%)": round(max_drawdown_percent, 2), 
        "Average Win": round(avg_win, 4),
        "Average Loss": round(avg_loss, 4),
    }
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ------------------ DIRECTORIES ------------------
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    BACKTEST_DIR = os.path.join(BASE_DIR, "..", "backtest")
    # TRADEBOOK_DIR = os.path.join(BACKTEST_DIR, "backtest_data")
    QS_DIR = os.path.join(BACKTEST_DIR, "output_quant_s")

    # Create directories safely
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(BACKTEST_DIR, exist_ok=True)
    os.makedirs(QS_DIR, exist_ok=True)

    # ------------------ PATHS ------------------
    DATA_PATH_1H = os.path.join(DATA_DIR, "backtest_data_30Min.csv")
    TRADEBOOK_PATH = os.path.join(BACKTEST_DIR, "mtf_ss_tradebook.csv")
    REPORT_PATH = os.path.join(QS_DIR, "QuantStats_Report_mtf_ss.html")

    try:
        df_1h = load_csv(DATA_PATH_1H, "30-Minute")

        print("\nMulti-Entry Scalping Strategy backtest")

        INITIAL_CAPITAL = 100000.0
        strategy = MultiEntryScalpingStrategy(initial_capital=INITIAL_CAPITAL)
        
        trades_df_all = strategy.run_backtest(df_1h)

        trades_df = trades_df_all.dropna(subset=["Exit"]).reset_index(drop=True)
        
        open_trades = trades_df_all[trades_df_all["Exit_Time"].isna()]
        if not open_trades.empty:
            print("\n--- Unclosed Position Found! ---")
            print(open_trades[["Entry_Time", "Side", "Entry", "Initial_SL"]].to_string(index=False))
            print("\n(This open trade is excluded from metrics/QuantStats reports.)")
            
        if not trades_df.empty:
            
            metrics = evaluate_metrics(trades_df, initial_capital=INITIAL_CAPITAL)
            
            # Save tradebook
            trades_df.to_csv(TRADEBOOK_PATH, index=False)
            print(f"\nClosed Tradebook saved to: {TRADEBOOK_PATH}")

            # QuantStats report
            run_quantstats(trades_df, REPORT_PATH, initial_capital=INITIAL_CAPITAL)

            print("\n--- Strategy Metrics (Closed Trades Only) ---")
            for k, v in metrics.items():
                print(f"{k:35s}: {v}")
            
        else:
            print("\n⚠️ Backtest completed but no trades were executed or closed.")

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        if "quantstats" in str(e).lower() and "no module named" in str(e).lower():
            print("💡 Install QuantStats using: pip install quantstats")
        print(f"Check data path: {DATA_PATH_1H}")
