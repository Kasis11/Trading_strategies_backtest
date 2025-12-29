import pandas as pd
import pandas_ta as ta
import numpy as np
import os

class RSIStrategy:
    def __init__(
        self,
        rsi_len=14,
        rsi_upper=55,
        rsi_lower=35,
        max_candle_size=0.04,
        atr_mult=2.0,
        rr_ratio=2.0,
        min_hold_days=2 * 24,     # min hold 2 days
        holding_period=7 * 24     # Max hold 7 days
    ):

        self.rsi_len = rsi_len
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        self.max_candle = max_candle_size
        self.atr_mult = atr_mult
        self.rr_ratio = rr_ratio
        self.min_hold_days = min_hold_days
        self.holding_period = holding_period

        self.position = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        self.entry_index = None
        self.contract = None

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
        df["SMA200"] = ta.sma(df["Close"], length=200)
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

        df["RSI"] = ta.rsi(df["Close"], length=self.rsi_len)
        df["RSI_Signal"] = df["RSI"].rolling(3).mean()

        df["Candle_Size"] = (df["High"] - df["Low"]) / df["Close"]

        return df


    def generate_signals(self, df: pd.DataFrame):
        trades = []

        for i in range(len(df)):
            row = df.iloc[i]

            # EXIT RULE
            if self.position is not None:
                bars_open = i - self.entry_index

                if bars_open >= self.min_hold_days:

                    # SL exit
                    if self.position == "long" and row["Low"] <= self.stop_loss:
                        trades.append((df.index[i], "EXIT SL", self.stop_loss, self.position, self.contract))
                        self.position = None
                        continue

                    elif self.position == "short" and row["High"] >= self.stop_loss:
                        trades.append((df.index[i], "EXIT SL", self.stop_loss, self.position, self.contract))
                        self.position = None
                        continue

                    # TP exit
                    if self.position == "long" and row["High"] >= self.take_profit:
                        trades.append((df.index[i], "EXIT TP", self.take_profit, self.position, self.contract))
                        self.position = None
                        continue

                    elif self.position == "short" and row["Low"] <= self.take_profit:
                        trades.append((df.index[i], "EXIT TP", self.take_profit, self.position, self.contract))
                        self.position = None
                        continue

                    # Max period exit
                    if bars_open >= self.holding_period:
                        trades.append((df.index[i], "EXIT MAX HOLD", row["Close"], self.position, self.contract))
                        self.position = None
                        continue

            # ENTRY RULE
            if self.position is None:
                if pd.isna(row["SMA200"]) or pd.isna(row["RSI_Signal"]):
                    continue

                if row["Candle_Size"] > self.max_candle:
                    continue

                # LONG ENTRY
                if row["Close"] > row["SMA200"] and row["RSI_Signal"] > self.rsi_upper:
                    self.position = "long"
                    self.entry_price = row["Close"]
                    self.stop_loss = self.entry_price - row["ATR"] * self.atr_mult
                    sl_dist = abs(self.entry_price - self.stop_loss)
                    self.take_profit = self.entry_price + sl_dist * self.rr_ratio
                    self.entry_index = i
                    self.contract = self.pick_contract(df.index[i])
                    trades.append((df.index[i], "BUY", self.entry_price, "long", self.contract))

                # SHORT ENTRY
                elif row["Close"] < row["SMA200"] and row["RSI_Signal"] < self.rsi_lower:
                    self.position = "short"
                    self.entry_price = row["Close"]
                    self.stop_loss = self.entry_price + row["ATR"] * self.atr_mult
                    sl_dist = abs(self.entry_price - self.stop_loss)
                    self.take_profit = self.entry_price - sl_dist * self.rr_ratio
                    self.entry_index = i
                    self.contract = self.pick_contract(df.index[i])
                    trades.append((df.index[i], "SELL", self.entry_price, "short", self.contract))

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

    # Calculate performance metrics
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

import matplotlib.pyplot as plt
import seaborn as sns

def plot_equity_curve(tradebook):
    plt.figure(figsize=(12,6))
    plt.plot(tradebook["Entry_Time"], tradebook["Equity"])
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity (PnL)")
    plt.grid(True)
    plt.show()
def plot_close_vs_sma(df):
    plt.figure(figsize=(16,6))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.7)
    plt.plot(df.index, df["SMA200"], label="SMA200", linewidth=2)

    plt.title("Close Price vs SMA200")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()
import matplotlib.pyplot as plt

def plot_executed_trades(df, tradebook):
    plt.figure(figsize=(18, 8))

    # Plot the base chart (Close price)
    price_line, = plt.plot(df.index, df["Close"], label="Close Price", alpha=0.5)

    # Legend placeholders
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

        # Determine entry marker
        if side == "long":
            entry_color = "green"
            entry_marker = "^"
            legend_key = "Long Entry (Green ▲)"
        else:
            entry_color = "red"
            entry_marker = "v"
            legend_key = "Short Entry (Red ▼)"

        # Plot entry marker
        point = plt.scatter(entry_time, entry_price, marker=entry_marker, color=entry_color, s=140, edgecolor="black")
        if legend_items[legend_key] is None:
            legend_items[legend_key] = point

        # Plot exit marker
        exit_point = plt.scatter(exit_time, exit_price, marker="x", color="black", s=120)
        if legend_items["Exit Point (Black X)"] is None:
            legend_items["Exit Point (Black X)"] = exit_point

        # Determine trade result color
        if (side == "long" and exit_price > entry_price) or (side == "short" and exit_price < entry_price):
            trade_color = "green"
            legend_line_key = "Winning Trade Line (Green)"
        else:
            trade_color = "red"
            legend_line_key = "Losing Trade Line (Red)"

        # Connecting trade line
        trade_line, = plt.plot([entry_time, exit_time], [entry_price, exit_price], color=trade_color, linewidth=2)

        if legend_items[legend_line_key] is None:
            legend_items[legend_line_key] = trade_line

    # Build Legend
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


if __name__ == "__main__":

    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")
    df = pd.read_csv(data_path)


    # Convert index to datetime
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)

    strategy = RSIStrategy()
    df = strategy.prepare_indicators(df)

    trades = strategy.generate_signals(df)

    tradebook, stats = evaluate_metrics(trades)

    if tradebook is not None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "Backtest2_rsi_tradebook.csv")
        tradebook.to_csv(save_path, index=False)
        print("\nTradebook saved → Backtest2_rsi_tradebook.csv\n")
        print(pd.DataFrame([stats]).T)

     #  PLOTS 
        plot_close_vs_sma(df)
        plot_executed_trades(df, tradebook)
        plot_equity_curve(tradebook)
        plot_monthly_heatmap(tradebook)
