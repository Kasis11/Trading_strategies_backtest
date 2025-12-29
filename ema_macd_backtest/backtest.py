import pandas as pd
import pandas_ta as ta
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class EMAMACDStrategy:
    def __init__(
        self,
        ema_fast=8,
        ema_mid=13,
        ema_slow=21,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        swing_lookback=3,
        risk_reward=2.5,
        stop_buffer=0.002  # 0.2% default
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_slow = ema_slow
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.swing_lookback = swing_lookback
        self.risk_reward = risk_reward
        self.stop_buffer = stop_buffer

        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_index = None

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
        df["EMA_FAST"] = ta.ema(df["Close"], length=self.ema_fast)
        df["EMA_MID"] = ta.ema(df["Close"], length=self.ema_mid)
        df["EMA_SLOW"] = ta.ema(df["Close"], length=self.ema_slow)

        macd = ta.macd(df["Close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]

        return df

    def generate_signals(self, df: pd.DataFrame):
        trades = []

        for i in range(len(df)):
            row = df.iloc[i]

            if self.position:
                # Stop-loss / Take-profit check
                if self.position == "long" and (row["Low"] <= self.stop_loss or row["High"] >= self.take_profit):
                    trades.append((df.index[i], "EXIT", row["Close"], self.position, self.contract))
                    self.position = None
                    self.pending_entry_signal = None
                    self.signal_candle_data = None
                    continue
                if self.position == "short" and (row["High"] >= self.stop_loss or row["Low"] <= self.take_profit):
                    trades.append((df.index[i], "EXIT", row["Close"], self.position, self.contract))
                    self.position = None
                    self.pending_entry_signal = None
                    self.signal_candle_data = None
                    continue

            if self.pending_entry_signal and self.position is None and i > 0:
                # Enter trade next candle
                entry_side = self.pending_entry_signal
                self.entry_price = row["Open"]
                signal_data = self.signal_candle_data

                if entry_side == "long":
                    swing_low = df["Low"].iloc[max(0, i - self.swing_lookback):i].min()
                    self.stop_loss = swing_low * (1 - self.stop_buffer)
                    self.take_profit = self.entry_price + (self.entry_price - self.stop_loss) * self.risk_reward
                    trades.append((df.index[i], "BUY", self.entry_price, "long", self.contract))
                else:
                    swing_high = df["High"].iloc[max(0, i - self.swing_lookback):i].max()
                    self.stop_loss = swing_high * (1 + self.stop_buffer)
                    self.take_profit = self.entry_price - (self.stop_loss - self.entry_price) * self.risk_reward
                    trades.append((df.index[i], "SELL", self.entry_price, "short", self.contract))

                self.position = entry_side
                self.entry_index = i
                self.pending_entry_signal = None
                self.signal_candle_data = None
                continue

            if self.position is None and self.pending_entry_signal is None:
                # Check long conditions
                long_signal = (
                    row["EMA_FAST"] > row["EMA_MID"] > row["EMA_SLOW"] and
                    df["MACD"].iloc[i] > df["MACD_SIGNAL"].iloc[i]
                )
                if long_signal:
                    self.pending_entry_signal = "long"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

                # Check short conditions
                short_signal = (
                    row["EMA_FAST"] < row["EMA_MID"] < row["EMA_SLOW"] and
                    df["MACD"].iloc[i] < df["MACD_SIGNAL"].iloc[i]
                )
                if short_signal:
                    self.pending_entry_signal = "short"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

        # Auto-exit last bar
        if self.position:
            last_time = df.index[-1]
            last_price = df["Close"].iloc[-1]
            trades.append((last_time, "EXIT EOD", last_price, self.position, self.contract))
            self.position = None

        return pd.DataFrame(trades, columns=["Time", "Action", "Price", "Position", "Contract"])


def evaluate_metrics(trades: pd.DataFrame):
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
        else:
            if open_trade:
                exit_price = row["Price"]
                entry = open_trade["Entry_Price"]
                pnl = (exit_price - entry) if open_trade["Side"] == "long" else (entry - exit_price)
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
    if results.empty:
        print("No trades executed.")
        return results, None

    win_trades = (results["PnL"] > 0).sum()
    loss_trades = (results["PnL"] <= 0).sum()
    win_rate = win_trades / len(results) * 100
    results["Equity"] = results["PnL"].cumsum()
    drawdown = results["Equity"] - results["Equity"].cummax()
    max_dd = drawdown.min()

    summary = {
        "Total Trades": len(results),
        "Winning Trades": win_trades,
        "Losing Trades": loss_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Net Profit": round(results["PnL"].sum(), 2),
        "Max Drawdown": round(max_dd, 2)
    }

    return results, summary


# -------------------- PLOTTING -------------------- #

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


def plot_close_vs_ema(df):
    """Plot Close Price vs EMAs for visualization."""
    plt.figure(figsize=(16,6))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
    plt.plot(df.index, df["EMA_FAST"], label="EMA 8", color="green", linewidth=1.5)
    plt.plot(df.index, df["EMA_MID"], label="EMA 13", color="orange", linewidth=1.5)
    plt.plot(df.index, df["EMA_SLOW"], label="EMA 21", color="blue", linewidth=1.5)

    plt.title("Close Price vs EMAs")
    plt.xlabel("Date / Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    # Load CSV
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")
    df = pd.read_csv(data_path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    strategy = EMAMACDStrategy()
    df_ind = strategy.prepare_indicators(df.copy())
    trades = strategy.generate_signals(df_ind)
    tradebook, stats = evaluate_metrics(trades)

    if tradebook is not None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "Backtest_EMAMACD_tradebook.csv")
        tradebook.to_csv(save_path, index=False)
        print("\nTradebook saved → Backtest_EMAMACD_tradebook.csv\n")
        print(pd.DataFrame([stats]).T)
        # PLOTS
        plot_close_vs_ema(df_ind)
        plot_executed_trades(df_ind, tradebook)
        plot_equity_curve(tradebook)
        plot_monthly_heatmap(tradebook)
