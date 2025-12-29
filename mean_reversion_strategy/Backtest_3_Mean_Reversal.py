
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import pandas_ta as ta
import numpy as np
import os
class MeanReversionStrategy:
    def __init__(
        self,
        bb_period=20,
        bb_stdev=2.0,
        adx_len=14,
        rsi_4h_bull_thresh=55,
        rsi_4h_bear_thresh=45,
        adx_1h_thresh=20,
        adx_4h_thresh=25,
        atr_mult=4.5,
        # min_hold = 2 * 24 # Minimum hold for 2 days on 1hr time
    ):
        self.bb_period = bb_period
        self.bb_stdev = bb_stdev
        self.adx_len = adx_len
        self.rsi_4h_bull_thresh = rsi_4h_bull_thresh
        self.rsi_4h_bear_thresh = rsi_4h_bear_thresh
        self.adx_1h_thresh = adx_1h_thresh
        self.adx_4h_thresh = adx_4h_thresh
        self.atr_mult = atr_mult
        
        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.entry_index = None
        self.contract = None

        self.pending_entry_signal = None 
        self.signal_candle_data = None    

    @staticmethod
    def pick_contract(dt: pd.Timestamp) -> str:
        year = dt.year
        month = dt.month
        roll_months = [3, 6, 9, 12]
        for rm in roll_months:
            if month < rm:
                return f"{year}-{rm:02d}"
        return f"{year+1}-03" 

    def prepare_indicators(self, df: pd.DataFrame):
        bb = ta.bbands(df["Close"], length=self.bb_period, std=self.bb_stdev)
        df["BB_LOWER"] = bb[f"BBL_{self.bb_period}_{self.bb_stdev}"]
        df["BB_UPPER"] = bb[f"BBU_{self.bb_period}_{self.bb_stdev}"]
        df["BB_MIDDLE"] = bb[f"BBM_{self.bb_period}_{self.bb_stdev}"]
        
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.adx_len)

        adx_1h = ta.adx(df["High"], df["Low"], df["Close"], length=self.adx_len)
        df["ADX_1H"] = adx_1h[f"ADX_{self.adx_len}"]

        df_4h = df.resample('4H').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
        }).dropna()

        df_4h["RSI_4H"] = ta.rsi(df_4h["Close"], length=14)
        
        adx_4h = ta.adx(df_4h["High"], df_4h["Low"], df_4h["Close"], length=self.adx_len)
        df_4h["ADX_4H"] = adx_4h[f"ADX_{self.adx_len}"]
        
        df = df.merge(df_4h[["RSI_4H", "ADX_4H"]], left_index=True, right_index=True, how='left')
        df["RSI_4H"] = df["RSI_4H"].fillna(method='ffill')
        df["ADX_4H"] = df["ADX_4H"].fillna(method='ffill')
        
        return df

    def generate_signals(self, df: pd.DataFrame):
        trades = []

        if 'Open' not in df.columns:
            raise ValueError("DataFrame must contain an 'Open' column for accurate next-candle entry.")

        for i in range(len(df)):
            row = df.iloc[i]

            if self.position is not None:
                
                if self.position == "long" and row["Low"] <= self.stop_loss:
                    trades.append((df.index[i], "EXIT SL", self.stop_loss, self.position, self.contract))
                    self.position = None
                    self.signal_candle_data = None
                    continue

                elif self.position == "short" and row["High"] >= self.stop_loss:
                    trades.append((df.index[i], "EXIT SL", self.stop_loss, self.position, self.contract))
                    self.position = None
                    self.signal_candle_data = None
                    continue

                if self.position == "long" and row["Close"] > row["BB_UPPER"]:
                    trades.append((df.index[i], "EXIT TP (BB)", row["Close"], self.position, self.contract))
                    self.position = None
                    self.signal_candle_data = None
                    continue

                elif self.position == "short" and row["Close"] < row["BB_LOWER"]:
                    trades.append((df.index[i], "EXIT TP (BB)", row["Close"], self.position, self.contract))
                    self.position = None
                    self.signal_candle_data = None
                    continue

            if self.pending_entry_signal is not None and self.position is None and i > 0:
                entry_side = self.pending_entry_signal
                
                self.entry_price = row["Open"] 
                
                signal_data = self.signal_candle_data
                
                if entry_side == "long":
                    self.stop_loss = signal_data["Low"] - (signal_data["ATR"] * self.atr_mult)
                    trades.append((df.index[i], "BUY", self.entry_price, "long", self.contract))
                
                elif entry_side == "short":
                    self.stop_loss = signal_data["High"] + (signal_data["ATR"] * self.atr_mult)
                    trades.append((df.index[i], "SELL", self.entry_price, "short", self.contract))

                self.position = entry_side
                self.entry_index = i
                

                self.pending_entry_signal = None
                self.signal_candle_data = None
                continue

            if self.position is None and self.pending_entry_signal is None:
                
                if pd.isna(row["BB_LOWER"]) or pd.isna(row["RSI_4H"]) or pd.isna(row["ADX_1H"]):
                    continue

                long_signal = (
                    row["RSI_4H"] > self.rsi_4h_bull_thresh
                    and row["ADX_1H"] > self.adx_1h_thresh
                    and row["ADX_4H"] > self.adx_4h_thresh
                    and row["Close"] < row["BB_LOWER"]
                )

                if long_signal:
                    self.pending_entry_signal = "long"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

                short_signal = (
                    row["RSI_4H"] < self.rsi_4h_bear_thresh
                    and row["ADX_1H"] > self.adx_1h_thresh
                    and row["ADX_4H"] > self.adx_4h_thresh
                    and row["Close"] > row["BB_UPPER"]
                )

                if short_signal:
                    self.pending_entry_signal = "short"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

            # AUTO EXIT ON LAST BAR 
        if self.position is not None:
            last_time = df.index[-1]
            last_price = df["Close"].iloc[-1]
            trades.append(
                (last_time, "EXIT EOD", last_price, self.position, self.contract)
            )
            # reset state
            self.position = None
            self.pending_entry_signal = None
            self.signal_candle_data = None


        return pd.DataFrame(trades, columns=["Time", "Action", "Price", "Position", "Contract"])
    
def evaluate_metrics(trades: pd.DataFrame):
    """Convert entry/exit pairs into profit metrics."""
    trade_log = []
    open_trade = None

    for idx, row in trades.iterrows():
        if row["Action"] in ["BUY", "SELL"]:
            open_trade = {
                "Entry_Time": row["Time"],
                "Entry_Price": row["Price"],
                "Side": row["Position"],
                "Contract": row["Contract"],
            }
        else:  # exit
            if open_trade:
                exit_price = row["Price"]
                entry = open_trade["Entry_Price"]

                if open_trade["Side"] == "long":
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price

                trade_log.append({
                    "Entry_Time": open_trade["Entry_Time"],
                    "Exit_Time": row["Time"],
                    "Side": open_trade["Side"],
                    "Entry": entry,
                    "Exit": exit_price,
                    "Contract": open_trade["Contract"],
                    "PnL": pnl
                })
                open_trade = None

    results = pd.DataFrame(trade_log)

    if len(results) == 0:
        print("No completed trades.")
        return results, None

    win_trades = (results["PnL"] > 0).sum()
    loss_trades = (results["PnL"] <= 0).sum()
    win_rate = win_trades / len(results) * 100
    results["Equity"] = results["PnL"].cumsum()

    drawdown = (results["Equity"] - results["Equity"].cummax())
    max_dd = drawdown.min()

    summary = {
        "Total Trades": len(results),
        "Winning Trades": win_trades,
        "Losing Trades": loss_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit": round(results["PnL"].sum(), 2),
        "Max Drawdown": round(max_dd, 2),
    }

    return results, summary

def plot_equity_curve(tradebook):
    plt.figure(figsize=(12,6))
    plt.plot(tradebook["Exit_Time"], tradebook["Equity"])
    plt.title("Equity Curve (Based on Exit Times)")
    plt.xlabel("Time")
    plt.ylabel("Equity (PnL)")
    plt.grid(True)
    plt.show()


def plot_executed_trades(df, tradebook):
    plt.figure(figsize=(18, 8))
    price_line, = plt.plot(df.index, df["Close"], label="Close Price", alpha=0.5)

    legend_items = {
        "Long Entry (Green ▲)": None,
        "Short Entry (Red ▼)": None,
        "Exit Point (Black X)": None,
        "Winning Trade Line (Green)": None,
        "Losing Trade Line (Red)": None
    }

    for i in range(len(tradebook)):
        entry_time = tradebook.loc[i, "Entry_Time"]
        exit_time = tradebook.loc[i, "Exit_Time"]
        entry_price = tradebook.loc[i, "Entry"]
        exit_price = tradebook.loc[i, "Exit"]
        side = tradebook.loc[i, "Side"]

        if side == "long":
            entry_color = "green"
            entry_marker = "^"
            legend_key = "Long Entry (Green ▲)"
        else:
            entry_color = "red"
            entry_marker = "v"
            legend_key = "Short Entry (Red ▼)"

        point = plt.scatter(entry_time, entry_price, marker=entry_marker, color=entry_color, s=140, edgecolor="black")
        if legend_items[legend_key] is None:
            legend_items[legend_key] = point

        exit_point = plt.scatter(exit_time, exit_price, marker="x", color="black", s=120)
        if legend_items["Exit Point (Black X)"] is None:
            legend_items["Exit Point (Black X)"] = exit_point

        if (side == "long" and exit_price > entry_price) or (side == "short" and exit_price < entry_price):
            trade_color = "green"
            legend_line_key = "Winning Trade Line (Green)"
        else:
            trade_color = "red"
            legend_line_key = "Losing Trade Line (Red)"

        trade_line, = plt.plot([entry_time, exit_time], [entry_price, exit_price], color=trade_color, linewidth=2)

        if legend_items[legend_line_key] is None:
            legend_items[legend_line_key] = trade_line

    plt.legend([price_line] + list(legend_items.values()), 
               ["Close Price"] + list(legend_items.keys()), 
               fontsize=10, loc="upper left")

    plt.title("Executed Trades — Entries, Exits & PnL Direction", fontsize=14)
    plt.xlabel("Date / Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

def plot_monthly_heatmap(tradebook):
    df = tradebook.copy()
    df["Month"] = df["Entry_Time"].dt.month
    df["Year"] = df["Entry_Time"].dt.year
    pivot = df.pivot_table(values="PnL", index="Year", columns="Month", aggfunc="sum")

    plt.figure(figsize=(12,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Monthly Profit Heatmap")
    plt.show()

def plot_close_vs_bb(df):
    """Plot Close Price vs Bollinger Bands for visualization."""
    plt.figure(figsize=(16,6))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
    plt.plot(df.index, df["BB_UPPER"], label="Upper BB", linewidth=1.5, color='orange', linestyle='--')
    plt.plot(df.index, df["BB_MIDDLE"], label="Middle BB (SMA)", linewidth=2, color='blue')
    plt.plot(df.index, df["BB_LOWER"], label="Lower BB", linewidth=1.5, color='orange', linestyle='--')

    plt.title("Close Price vs Bollinger Bands")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try:

        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")
        df = pd.read_csv(data_path)

        # df = pd.read_csv("backtest_data_1H.csv")
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
    except FileNotFoundError:
        print("ERROR: 'backtest_data_1H.csv' not found. Please provide your data file.")
        exit()
        
    strategy = MeanReversionStrategy()
    df_with_indicators = strategy.prepare_indicators(df.copy())

    trades = strategy.generate_signals(df_with_indicators)

    tradebook, stats = evaluate_metrics(trades)

    if tradebook is not None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "Backtest_MeanReversion_tradebook.csv")
        tradebook.to_csv(save_path, index=False)
        print("\nTradebook saved → Backtest_MeanReversion_tradebook.csv\n")
        print(pd.DataFrame([stats]).T)

        # PLOTS
        plot_close_vs_bb(df_with_indicators)
        plot_executed_trades(df_with_indicators, tradebook)
        plot_equity_curve(tradebook)
        plot_monthly_heatmap(tradebook)