import pandas as pd
import pandas_ta as ta
import numpy as np
import os

class Ichimoku_CloudStrategy:
    def __init__(
        self,
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        atr_period=14,
        atr_mult=3.5,
        slope_lookback=3,  # number of candles to check slope
        cloud_margin=0.002  # 0.2% above/below cloud for signal strength
    ):
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.slope_lookback = slope_lookback
        self.cloud_margin = cloud_margin

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
        df = df.copy().reset_index()

        # Ichimoku components
        df["tenkan_sen"] = (df["High"].rolling(self.tenkan_period).max() +
                            df["Low"].rolling(self.tenkan_period).min()) / 2
        df["kijun_sen"] = (df["High"].rolling(self.kijun_period).max() +
                           df["Low"].rolling(self.kijun_period).min()) / 2
        df["senkou_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(self.kijun_period)
        df["senkou_b"] = ((df["High"].rolling(self.senkou_b_period).max() +
                           df["Low"].rolling(self.senkou_b_period).min()) / 2).shift(self.kijun_period)
        df["chikou_span"] = df["Close"].shift(-self.kijun_period)

        # ATR for exit
        df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=self.atr_period)

        df.set_index("Datetime", inplace=True)
        return df

    def slope_upward(self, series):
        return all(series.iloc[-self.slope_lookback + i] >= series.iloc[-self.slope_lookback + i - 1] 
                   for i in range(1, self.slope_lookback))

    def slope_downward(self, series):
        return all(series.iloc[-self.slope_lookback + i] <= series.iloc[-self.slope_lookback + i - 1] 
                   for i in range(1, self.slope_lookback))

    def generate_signals(self, df: pd.DataFrame):
        trades = []
        if 'Open' not in df.columns:
            raise ValueError("DataFrame must contain 'Open' column for next-candle entries.")

        for i in range(self.slope_lookback, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # senkou_b HANDLE EXIT senkou_b
            if self.position == "long" and row["Close"] < (row["kijun_sen"] - self.atr_mult * row["ATR"]):
                trades.append((df.index[i], "EXIT LONG", row["Close"], "long", self.contract))
                self.position = None
                self.pending_entry_signal = None
                self.signal_candle_data = None
                continue
            elif self.position == "short" and row["Close"] > (row["kijun_sen"] + self.atr_mult * row["ATR"]):
                trades.append((df.index[i], "EXIT SHORT", row["Close"], "short", self.contract))
                self.position = None
                self.pending_entry_signal = None
                self.signal_candle_data = None
                continue

            # senkou_b EXECUTE PENDING ENTRY senkou_b
            if self.pending_entry_signal and self.position is None:
                entry_side = self.pending_entry_signal
                self.entry_price = row["Open"]
                signal_data = self.signal_candle_data

                if entry_side == "long":
                    self.stop_loss = signal_data["kijun_sen"] - self.atr_mult * signal_data["ATR"]
                    trades.append((df.index[i], "BUY", self.entry_price, "long", self.contract))
                elif entry_side == "short":
                    self.stop_loss = signal_data["kijun_sen"] + self.atr_mult * signal_data["ATR"]
                    trades.append((df.index[i], "SELL", self.entry_price, "short", self.contract))

                self.position = entry_side
                self.entry_index = i
                self.pending_entry_signal = None
                self.signal_candle_data = None
                continue

            if self.position is None and self.pending_entry_signal is None:
                # Long conditions
                tenkan_slope_up = self.slope_upward(df["tenkan_sen"].iloc[i - self.slope_lookback:i+1])
                kijun_slope_up = self.slope_upward(df["kijun_sen"].iloc[i - self.slope_lookback:i+1])
                cloud_up = row["senkou_a"] > row["senkou_b"]
                strength_long = row["Close"] > max(row["senkou_a"], row["senkou_b"]) * (1 + self.cloud_margin)
                
                min_sep = 0.01 

                long_cond = (
                    prev["tenkan_sen"] <= prev["kijun_sen"] and row["tenkan_sen"] > row["kijun_sen"] and
                    tenkan_slope_up and kijun_slope_up and cloud_up and strength_long and
                    (row["tenkan_sen"] - row["kijun_sen"]) >= min_sep
                )

                if long_cond:
                    self.pending_entry_signal = "long"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

                # Short conditions
                tenkan_slope_down = self.slope_downward(df["tenkan_sen"].iloc[i - self.slope_lookback:i+1])
                kijun_slope_down = self.slope_downward(df["kijun_sen"].iloc[i - self.slope_lookback:i+1])
                cloud_down = row["senkou_a"] < row["senkou_b"]
                strength_short = row["Close"] < min(row["senkou_a"], row["senkou_b"]) * (1 - self.cloud_margin)
                


                short_cond = (
                    prev["tenkan_sen"] >= prev["kijun_sen"] and row["tenkan_sen"] < row["kijun_sen"] and
                    tenkan_slope_down and kijun_slope_down and cloud_down and strength_short and
                    (row["kijun_sen"] - row["tenkan_sen"]) >= min_sep
                )


                if short_cond:
                    self.pending_entry_signal = "short"
                    self.signal_candle_data = row
                    self.contract = self.pick_contract(df.index[i])
                    continue

        # senkou_b AUTO EXIT ON LAST BAR senkou_b
        if self.position is not None:
            last_time = df.index[-1]
            last_price = df["Close"].iloc[-1]
            trades.append(
                (last_time, "EXIT EOD", last_price, self.position, self.contract)
            )
            self.position = None
            self.pending_entry_signal = None
            self.signal_candle_data = None

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
                "Contract": row["Contract"]
            }
        else:
            if open_trade:
                exit_price = row["Price"]
                entry = open_trade["Entry_Price"]
                pnl = exit_price - entry if open_trade["Side"] == "long" else entry - exit_price
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
    max_dd = (results["Equity"] - results["Equity"].cummax()).min()

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
    if "Equity" not in tradebook.columns:
        tradebook["Equity"] = tradebook["PnL"].cumsum()

    plt.figure(figsize=(12, 6))
    plt.plot(tradebook["Exit_Time"], tradebook["Equity"], linewidth=2)
    plt.title("Ichimoku Strategy Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.show()


def plot_ichimoku(df):
    """Plot Close Price with Ichimoku Cloud"""
    plt.figure(figsize=(18, 8))
    
    plt.plot(df.index, df["Close"], label="Close Price", linewidth=1.2, color="black")
    plt.plot(df.index, df["tenkan_sen"], label="Tenkan Sen", linewidth=1.5, color="blue")
    plt.plot(df.index, df["kijun_sen"], label="Kijun Sen", linewidth=1.5, color="red")
    plt.plot(df.index, df["chikou_span"], label="Chikou Span", linewidth=1, color="green", alpha=0.6)

    # Cloud Plot
    plt.plot(df.index, df["senkou_a"], label="Senkou Span A", linewidth=1.5, color="orange")
    plt.plot(df.index, df["senkou_b"], label="Senkou Span B", linewidth=1.5, color="purple")

    plt.fill_between(df.index, df["senkou_a"], df["senkou_b"],
                     where=df["senkou_a"] >= df["senkou_b"],
                     color="lightgreen", alpha=0.3)

    plt.fill_between(df.index, df["senkou_a"], df["senkou_b"],
                     where=df["senkou_a"] < df["senkou_b"],
                     color="lightcoral", alpha=0.3)

    plt.title("Ichimoku Cloud Overview")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_executed_trades(df, tradebook):
    """Plot trades over Ichimoku chart for clarity"""
    plt.figure(figsize=(20, 10))

    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.5)

    for i, trade in tradebook.iterrows():
        entry_time, exit_time = trade["Entry_Time"], trade["Exit_Time"]
        entry, exitp = trade["Entry"], trade["Exit"]
        side = trade["Side"]

        if side == "long":
            color, marker = "green", "^"
        else:
            color, marker = "red", "v"

        plt.scatter(entry_time, entry, s=120, marker=marker, color=color, edgecolor='black')
        plt.scatter(exit_time, exitp, s=120, marker="X", color="black")

        line_color = "green" if (side == "long" and exitp > entry) or (side == "short" and exitp < entry) else "red"
        plt.plot([entry_time, exit_time], [entry, exitp], color=line_color, linewidth=2)

    plt.title("Executed Trades Over Price (With Ichimoku)")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()


def plot_monthly_heatmap(tradebook):
    df = tradebook.copy()
    df["Month"] = df["Entry_Time"].dt.month
    df["Year"] = df["Entry_Time"].dt.year

    pivot = df.pivot_table(values="PnL", index="Year", columns="Month", aggfunc="sum")

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Monthly Profit Heatmap")
    plt.show()
    
if __name__ == "__main__":
    import os

    try:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")
        df = pd.read_csv(data_path)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        print(f"📥 Loaded data from → {data_path}")
    except FileNotFoundError:
        print("❌ ERROR: backtest_data_4H.csv NOT found. Place the file in the 'data' folder.")
        exit()

    strategy = Ichimoku_CloudStrategy(
        tenkan_period=9,
        kijun_period=26,
        senkou_b_period=52,
        atr_period=14,
        atr_mult=3.5,
        slope_lookback=3,
        cloud_margin=0.002
    )

    df_with_indicators = strategy.prepare_indicators(df.copy())
    trades = strategy.generate_signals(df_with_indicators)

    if trades.empty:
        print("⚠️ No trades triggered.")
        exit()

    tradebook, stats = evaluate_metrics(trades)

    if tradebook is not None:
        save_path = os.path.join(os.path.dirname(__file__), "..", "backtest", "Ichimoku_Backtest.csv")
        tradebook.to_csv(save_path, index=False)

        print(f"\n💾 Tradebook saved → {save_path}\n")
        print(pd.DataFrame([stats]).T)

        plot_ichimoku(df_with_indicators)
        plot_executed_trades(df_with_indicators, tradebook)
        plot_equity_curve(tradebook)
        plot_monthly_heatmap(tradebook)
