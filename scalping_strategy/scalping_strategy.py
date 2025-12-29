# 3EMA (7, 13, 400 of close price), 
# stochastic RSI (15 period and High, Low, Close Devide by 3),
# 3 ATR values (16 period, multiplied by 1.9 and 1.57)

### Rules:

## Long Entry:
# 1. EMA order should be 400 < 13 < 7
# 2. Entry on stochastic RSI crossing up

## Long Exit:
# 1. Take profit: 1.9 * ATR(16)
# 2. Stop loss: 3 ATR(16)

## Short Entry:
# 1. EMA order should be 7 < 13 < 400
# 2. Entry on stochastic RSI crossing down

## Short Exit:
# 1. Take profit: 1.9 * ATR(16)
# 2. Stop loss: 3 ATR(16)

# scalping_strategy.py
# Implements the scalping rules from the transcript:
# 3 EMA (7,13,400), Stochastic RSI (15,15 on HLC3), ATR(16)
# TP = 1.9 * ATR, SL = 3.0 * ATR
# Entry on next candle open after StochRSI cross when EMAs aligned.


import os
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
import quantstats as qs

class ScalpingStrategy:
    def __init__(
        self,
        ema_fast: int = 7,
        ema_mid: int = 13,
        ema_trend: int = 400,
        rsi_length: int = 15,
        stoch_length: int = 15,
        atr_length: int = 16,
        tp_mult: float = 1.9,            # TP multiplier (1.9 * ATR)
        atr_mult1: float = 1.9,         # part of SL multiplier (we will multiply atr_mult1 * atr_mult2)
        atr_mult2: float = 1.57,        # part of SL multiplier
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_trend = ema_trend
        self.rsi_length = rsi_length
        self.stoch_length = stoch_length
        self.atr_length = atr_length

        # Keep the "B" option behavior: SL = atr * atr_mult1 * atr_mult2
        self.tp_mult = tp_mult
        self.atr_mult1 = atr_mult1
        self.atr_mult2 = atr_mult2

        # Ichimoku-style flow variables
        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None
        self.entry_index: Optional[int] = None
        self.contract: Optional[str] = None
        self.pending_entry_signal: Optional[str] = None
        self.signal_candle_data: Optional[pd.Series] = None

        # stoch parameters
        self.k = 3
        self.d = 3

    @staticmethod
    def pick_contract(dt: pd.Timestamp) -> str:
        year = dt.year
        month = dt.month
        roll_months = [3, 6, 9, 12]
        for rm in roll_months:
            if month < rm:
                return f"{year}-{rm:02d}"
        return f"{year+1}-03"

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds hlc3, EMAs, RSI(HLC3), Stoch of RSI, ATR.
        Stoch is computed on the RSI series (stochastic of RSI) so cross signals are meaningful.
        """
        df = df.copy().reset_index(drop=True)

        required = ["Open", "High", "Low", "Close", "Datetime"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Input DataFrame must contain column '{c}'")

        # HLC3
        df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3.0

        # EMAs on Close
        df["ema_fast"] = ta.ema(df["Close"], length=self.ema_fast)
        df["ema_mid"] = ta.ema(df["Close"], length=self.ema_mid)
        df["ema_trend"] = ta.ema(df["Close"], length=self.ema_trend)

        # RSI on HLC3
        df["rsi_hlc3"] = ta.rsi(df["hlc3"], length=self.rsi_length)

        stoch_rsi = ta.stoch(high=df["rsi_hlc3"], low=df["rsi_hlc3"], close=df["rsi_hlc3"],
                             k=self.k, d=self.d, length=self.stoch_length)

        if stoch_rsi is None or stoch_rsi.empty:
            df["stoch_k"] = np.nan
            df["stoch_d"] = np.nan
        else:
            # pandas_ta names can vary; find columns containing 'k' and 'd'
            k_cols = [c for c in stoch_rsi.columns if "k" in c.lower()]
            d_cols = [c for c in stoch_rsi.columns if "d" in c.lower()]
            df["stoch_k"] = stoch_rsi[k_cols[0]] if k_cols else np.nan
            df["stoch_d"] = stoch_rsi[d_cols[0]] if d_cols else np.nan

        # ATR for sizing TP/SL
        df["atr"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=self.atr_length)

        return df

    def is_bull_trend(self, row: pd.Series) -> bool:
        """EMA order: 400 < 13 < 7 (trend EMA < mid EMA < fast EMA)"""
        try:
            return (pd.notna(row["ema_trend"]) and pd.notna(row["ema_mid"]) and pd.notna(row["ema_fast"]) and
                    (row["ema_trend"] < row["ema_mid"] < row["ema_fast"]))
        except Exception:
            return False

    def is_bear_trend(self, row: pd.Series) -> bool:
        """EMA order: 7 < 13 < 400"""
        try:
            return (pd.notna(row["ema_trend"]) and pd.notna(row["ema_mid"]) and pd.notna(row["ema_fast"]) and
                    (row["ema_fast"] < row["ema_mid"] < row["ema_trend"]))
        except Exception:
            return False

    def stoch_cross_up(self, prev: pd.Series, curr: pd.Series) -> bool:
        if pd.isna(prev.get("stoch_k")) or pd.isna(prev.get("stoch_d")) or pd.isna(curr.get("stoch_k")) or pd.isna(curr.get("stoch_d")):
            return False
        return (prev["stoch_k"] < prev["stoch_d"]) and (curr["stoch_k"] > curr["stoch_d"])

    def stoch_cross_down(self, prev: pd.Series, curr: pd.Series) -> bool:
        if pd.isna(prev.get("stoch_k")) or pd.isna(prev.get("stoch_d")) or pd.isna(curr.get("stoch_k")) or pd.isna(curr.get("stoch_d")):
            return False
        return (prev["stoch_k"] > prev["stoch_d"]) and (curr["stoch_k"] < curr["stoch_d"])

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ichimoku-style flow:
        - On signal candle set pending_entry_signal and record signal_candle_data
        - Execute on next candle open (entry at next candle Open)
        - Manage position exits using TP/SL (bar high/low)
        """
        df = self.compute_indicators(df).reset_index(drop=False)  # keep integer index, Datetime column must exist
        trades: List[Dict[str, Any]] = []

        # we will iterate until len(df)-2 so entry at i+1 is valid
        last_index_to_check = max(1, len(df) - 2)

        for i in range(1, last_index_to_check + 1):
            prev = df.iloc[i - 1]
            row = df.iloc[i]

            # Close EXECUTE PENDING ENTRY Close
            if self.pending_entry_signal and self.position is None:
                # ensure next candle exists
                if i + 1 >= len(df):
                    # no next candle to execute on; skip
                    self.pending_entry_signal = None
                    self.signal_candle_data = None
                else:
                    entry_side = self.pending_entry_signal
                    entry_row = df.iloc[i + 1]
                    self.entry_price = float(entry_row["Open"])
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(pd.to_datetime(row["Datetime"]))

                    atr_val = float(row["atr"]) if not pd.isna(row["atr"]) else np.nan
                    if pd.isna(atr_val):
                        self.pending_entry_signal = None
                        self.signal_candle_data = None
                        continue

                    if entry_side == "long":
                        self.tp = self.entry_price + (atr_val * self.tp_mult)
                        self.sl = self.entry_price - (atr_val * self.atr_mult1 * self.atr_mult2)
                    else:  # short
                        self.tp = self.entry_price - (atr_val * self.tp_mult)
                        self.sl = self.entry_price + (atr_val * self.atr_mult1 * self.atr_mult2)

                    trades.append({
                        "Entry_Time": pd.to_datetime(entry_row["Datetime"]),
                        "Exit_Time": pd.NaT,
                        "Side": entry_side,
                        "Entry": self.entry_price,
                        "Exit": np.nan,
                        "Tp": self.tp,
                        "Sl": self.sl,
                        "PnL": np.nan,
                        "Contract": self.contract
                    })

                    self.position = entry_side
                    self.entry_index = i + 1
                    self.pending_entry_signal = None
                    continue

            # Close EXIT LOGIC Close
            if self.position is not None:
                if not trades:
                    self.position = None
                    self.entry_price = None
                    self.tp = None
                    self.sl = None
                    self.entry_index = None
                    continue

                last_trade = trades[-1]
                bar_low = float(row["Low"])
                bar_high = float(row["High"])
                bar_dt = pd.to_datetime(row["Datetime"])

                if self.position == "long":
                    if bar_low <= self.sl:
                        last_trade["Exit_Time"] = bar_dt
                        last_trade["Exit"] = float(self.sl)
                        last_trade["PnL"] = float(self.sl) - float(last_trade["Entry"])
                        # reset position
                        self.position = None
                        self.entry_price = None
                        self.tp = None
                        self.sl = None
                        self.entry_index = None
                        continue
                    elif bar_high >= self.tp:
                        last_trade["Exit_Time"] = bar_dt
                        last_trade["Exit"] = float(self.tp)
                        last_trade["PnL"] = float(self.tp) - float(last_trade["Entry"])
                        self.position = None
                        self.entry_price = None
                        self.tp = None
                        self.sl = None
                        self.entry_index = None
                        continue

                else:  # short
                    if bar_high >= self.sl:
                        last_trade["Exit_Time"] = bar_dt
                        last_trade["Exit"] = float(self.sl)
                        last_trade["PnL"] = float(last_trade["Entry"]) - float(self.sl)
                        self.position = None
                        self.entry_price = None
                        self.tp = None
                        self.sl = None
                        self.entry_index = None
                        continue
                    elif bar_low <= self.tp:
                        last_trade["Exit_Time"] = bar_dt
                        last_trade["Exit"] = float(self.tp)
                        last_trade["PnL"] = float(last_trade["Entry"]) - float(self.tp)
                        self.position = None
                        self.entry_price = None
                        self.tp = None
                        self.sl = None
                        self.entry_index = None
                        continue

            # Close GENERATE PENDING ENTRY Close
            if self.position is None and self.pending_entry_signal is None:
                if self.is_bull_trend(row) and self.stoch_cross_up(prev, row):
                    self.pending_entry_signal = "long"
                    self.signal_candle_data = row
                    continue
                elif self.is_bear_trend(row) and self.stoch_cross_down(prev, row):
                    self.pending_entry_signal = "short"
                    self.signal_candle_data = row
                    continue

        # Auto exit at final candle if a position still open
        if self.position is not None and trades:
            last_bar = df.iloc[-1]
            bar_dt = pd.to_datetime(last_bar["Datetime"])
            final_price = float(last_bar["Close"])
            last_trade = trades[-1]

            if self.position == "long":
                last_trade["Exit_Time"] = bar_dt
                last_trade["Exit"] = final_price
                last_trade["PnL"] = final_price - float(last_trade["Entry"])
            else:
                last_trade["Exit_Time"] = bar_dt
                last_trade["Exit"] = final_price
                last_trade["PnL"] = float(last_trade["Entry"]) - final_price

            # reset
            self.position = None
            self.entry_price = None
            self.tp = None
            self.sl = None
            self.entry_index = None

        trades_df = pd.DataFrame(trades)
        # ensure types
        if not trades_df.empty:
            trades_df["Entry_Time"] = pd.to_datetime(trades_df["Entry_Time"])
            trades_df["Exit_Time"] = pd.to_datetime(trades_df["Exit_Time"])
            # Fill missing exits with NaN (already)
        return trades_df


def evaluate_metrics(trades: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    """
    Evaluates tradebook (expects columns Entry_Time, Exit_Time, Side, Entry, Exit, PnL, Contract).
    Returns (results_df, summary_dict) — summary_dict is None if no trades.
    """
    if trades is None or trades.empty:
        print("No trades executed.")
        return pd.DataFrame(), None

    df = trades.copy()

    if "PnL" not in df.columns or df["PnL"].isna().any():
        def compute_row_pnl(r):
            try:
                if pd.isna(r["Exit"]) or pd.isna(r["Entry"]):
                    return np.nan
                return (r["Exit"] - r["Entry"]) if r["Side"] == "long" else (r["Entry"] - r["Exit"])
            except Exception:
                return np.nan
        df["PnL"] = df.apply(compute_row_pnl, axis=1)

    df = df.dropna(subset=["Entry", "Exit", "Side"]).reset_index(drop=True)

    if df.empty:
        print("No completed trades to evaluate.")
        return pd.DataFrame(), None

    wins = (df["PnL"] > 0).sum()
    losses = (df["PnL"] <= 0).sum()
    total = len(df)
    win_rate = (wins / total * 100) if total > 0 else 0.0
    net = df["PnL"].sum()
    df["Equity"] = df["PnL"].cumsum()
    drawdown = df["Equity"] - df["Equity"].cummax()
    max_dd = drawdown.min()

    summary = {
        "Total Trades": int(total),
        "Winning Trades": int(wins),
        "Losing Trades": int(losses),
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit": round(float(net), 6),
        "Max Drawdown": round(float(max_dd), 6) if not np.isnan(max_dd) else 0.0
    }

    return df, summary


def plot_equity_curve(tradebook: pd.DataFrame):
    import matplotlib.pyplot as plt

    if tradebook is None or tradebook.empty:
        print("Nothing to plot (no trades).")
        return

    if "Equity" not in tradebook.columns:
        tradebook["Equity"] = tradebook["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(tradebook["Exit_Time"], tradebook["Equity"], linewidth=2)
    plt.title("Scalping Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.show()


def plot_executed_trades(df: pd.DataFrame, tradebook: pd.DataFrame):
    import matplotlib.pyplot as plt

    if tradebook is None or tradebook.empty:
        print("No trades to plot.")
        return

    plt.figure(figsize=(20, 10))
    plt.plot(pd.to_datetime(df["Datetime"]), df["Close"], label="Close Price", alpha=0.7)

    for _, trade in tradebook.iterrows():
        entry_time = trade["Entry_Time"]
        exit_time = trade["Exit_Time"]
        entry = trade["Entry"]
        exitp = trade["Exit"]
        side = trade["Side"]

        color = "green" if side == "long" else "red"
        marker = "^" if side == "long" else "v"

        plt.scatter(entry_time, entry, s=120, marker=marker, color=color, edgecolor='black')
        plt.scatter(exit_time, exitp, s=120, marker="X", color="black")
        line_color = "green" if (side == "long" and exitp > entry) or (side == "short" and exitp < entry) else "red"
        plt.plot([entry_time, exit_time], [entry, exitp], color=line_color, linewidth=2)

    plt.title("Executed Trades Over Price - Scalping Strategy")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_monthly_heatmap(tradebook: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if tradebook is None or tradebook.empty:
        print("No trades to plot heatmap.")
        return

    df = tradebook.copy()
    df["Month"] = df["Entry_Time"].dt.month
    df["Year"] = df["Entry_Time"].dt.year

    pivot = df.pivot_table(values="PnL", index="Year", columns="Month", aggfunc="sum").fillna(0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Monthly Profit Heatmap - Scalping Strategy")
    plt.show()
def run_quantstats(tradebook: pd.DataFrame, output_file: str):
    if tradebook is None or tradebook.empty:
        print("No trades for QuantStats report.")
        return

    tradebook = tradebook.copy()

    tradebook["Exit_Time"] = pd.to_datetime(tradebook["Exit_Time"]).dt.tz_localize(None)
    tradebook["Entry"] = tradebook["Entry"].astype(float)
    tradebook["PnL"] = tradebook["PnL"].astype(float)

    # ---- TRADE RETURNS → DAILY RETURNS ----
    returns_pct = tradebook["PnL"] / tradebook["Entry"]
    returns_pct.index = tradebook["Exit_Time"].dt.floor("D")

    returns_pct = returns_pct.groupby(returns_pct.index).sum()
    returns_pct = returns_pct.asfreq("D", fill_value=0)
    returns_pct = returns_pct.replace([np.inf, -np.inf], np.nan).dropna()

    # ---- Ensure output folder exists ----
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # ---- Generate QuantStats report ----
    qs.reports.html(
        returns_pct,
        output=output_file,
        title="Scalping Strategy Report",
        benchmark=None
    )

    print(f"QuantStats report generated → {output_file}")


if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        print("Please provide CSV with columns: Datetime, Open, High, Low, Close")
        raise SystemExit(1)

    df = pd.read_csv(DATA_PATH)
    if "Datetime" not in df.columns:
        possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    strat = ScalpingStrategy()
    trades = strat.generate_signals(df)

    out_folder = os.path.join(os.path.dirname(__file__), "..", "backtest")
    qs_dir = os.path.join(out_folder, "output_quant_s")
    os.makedirs(qs_dir, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "Scalping_tradebook_1H.csv")
    trades.to_csv(out_file, index=False)
    print(f"\nTradebook saved → {out_file}\n")

    tradebook, stats = evaluate_metrics(trades)
    if tradebook is not None and not tradebook.empty:
        print(pd.DataFrame([stats]).T)
        plot_equity_curve(tradebook)
        plot_executed_trades(df, tradebook)
        plot_monthly_heatmap(tradebook)
    
        qs_file = os.path.join(qs_dir, "scalping_strategy_quantstats.html")
        run_quantstats(trades, output_file=qs_file)

    else:
        print("No completed trades to show statistics/plots.")
