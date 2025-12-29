# iter1_quantstats_integration.py
# Variant B: TP reduced to 1.5 * ATR, SL = 2.25 * ATR,
# - TP = 1.5 * ATR
# - SL parts product = 2.25 * ATR (1.5 * 1.5)
# - StochRSI %K between 10 and 90
# - EMA distance filter and ATR min ratio filter
# QuantStats integration added: builds equity curve, returns series, and generates a QuantStats report.

import os
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import pandas_ta as ta
import quantstats as qs

# --- CONSTANT FOR REALISM ---
TRANSACTION_COST_PCT = 0.0001  # unused by default, kept if you want to apply costs


class ScalpingStrategyIterB:
    def __init__(
        self,
        ema_fast: int = 7,
        ema_mid: int = 13,
        ema_trend: int = 400,
        rsi_length: int = 15,
        stoch_length: int = 15,
        atr_length: int = 16,
        tp_mult: float = 1.5,
        atr_mult1: float = 1.5,
        atr_mult2: float = 1.5,
        stoch_k_min: float = 10.0,
        stoch_k_max: float = 90.0,
        ema_distance_thresh: float = 0.001,
        atr_min_ratio: float = 0.0005,
    ):
        self.ema_fast = ema_fast
        self.ema_mid = ema_mid
        self.ema_trend = ema_trend
        self.rsi_length = rsi_length
        self.stoch_length = stoch_length
        self.atr_length = atr_length

        self.tp_mult = tp_mult
        self.atr_mult1 = atr_mult1
        self.atr_mult2 = atr_mult2

        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.tp: Optional[float] = None
        self.sl: Optional[float] = None
        self.entry_index: Optional[int] = None
        self.contract: Optional[str] = None
        self.pending_entry_signal: Optional[str] = None
        self.signal_candle_data: Optional[pd.Series] = None

        self.k = 3
        self.d = 3

        self.stoch_k_min = stoch_k_min
        self.stoch_k_max = stoch_k_max
        self.ema_distance_thresh = ema_distance_thresh
        self.atr_min_ratio = atr_min_ratio

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
        df = df.copy().reset_index(drop=True)
        required = ["Open", "High", "Low", "Close", "Datetime"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Input DataFrame must contain column '{c}'")
        df["hlc3"] = (df["High"] + df["Low"] + df["Close"]) / 3.0
        df["ema_fast"] = ta.ema(df["Close"], length=self.ema_fast)
        df["ema_mid"] = ta.ema(df["Close"], length=self.ema_mid)
        df["ema_trend"] = ta.ema(df["Close"], length=self.ema_trend)
        df["rsi_hlc3"] = ta.rsi(df["hlc3"], length=self.rsi_length)
        stoch_rsi = ta.stoch(high=df["rsi_hlc3"], low=df["rsi_hlc3"], close=df["rsi_hlc3"],
                             k=self.k, d=self.d, length=self.stoch_length)
        if stoch_rsi is None or stoch_rsi.empty:
            df["stoch_k"] = np.nan
            df["stoch_d"] = np.nan
        else:
            k_cols = [c for c in stoch_rsi.columns if "k" in c.lower()]
            d_cols = [c for c in stoch_rsi.columns if "d" in c.lower()]
            df["stoch_k"] = stoch_rsi[k_cols[0]] if k_cols else np.nan
            df["stoch_d"] = stoch_rsi[d_cols[0]] if d_cols else np.nan
        df["atr"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=self.atr_length)
        return df

    def is_bull_trend(self, row: pd.Series) -> bool:
        try:
            return (pd.notna(row["ema_trend"]) and pd.notna(row["ema_mid"]) and pd.notna(row["ema_fast"]) and
                    (row["ema_trend"] < row["ema_mid"] < row["ema_fast"]))
        except Exception:
            return False

    def is_bear_trend(self, row: pd.Series) -> bool:
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
        df = self.compute_indicators(df).reset_index(drop=False)
        trades: List[Dict[str, Any]] = []
        last_index_to_check = max(1, len(df) - 2)
        for i in range(1, last_index_to_check + 1):
            prev = df.iloc[i - 1]
            row = df.iloc[i]

            # Execute pending entry
            if self.pending_entry_signal and self.position is None:
                if i + 1 >= len(df):
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
                    else:
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

            # Exit logic
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
                else:
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

            # Generate pending entry with filters
            if self.position is None and self.pending_entry_signal is None:
                # EMA distance filter (relative to price)
                if pd.notna(row["ema_trend"]) and pd.notna(row["ema_fast"]) and pd.notna(row["Close"]):
                    ema_dist = abs(row["ema_fast"] - row["ema_trend"]) / max(abs(row["Close"]), 1e-9)
                else:
                    ema_dist = 0.0
                # ATR ratio
                atr_ratio = (float(row["atr"]) / float(row["Close"])) if (not pd.isna(row.get("atr")) and row["Close"] != 0) else 0.0
                curr_k = row.get("stoch_k", np.nan)
                if self.is_bull_trend(row) and self.stoch_cross_up(prev, row):
                    if not pd.isna(curr_k) and (self.stoch_k_min <= curr_k <= self.stoch_k_max) and (ema_dist > self.ema_distance_thresh) and (atr_ratio > self.atr_min_ratio):
                        self.pending_entry_signal = "long"
                        self.signal_candle_data = row
                        continue
                elif self.is_bear_trend(row) and self.stoch_cross_down(prev, row):
                    if not pd.isna(curr_k) and (self.stoch_k_min <= curr_k <= self.stoch_k_max) and (ema_dist > self.ema_distance_thresh) and (atr_ratio > self.atr_min_ratio):
                        self.pending_entry_signal = "short"
                        self.signal_candle_data = row
                        continue

        # Auto exit last open position
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
            self.position = None
            self.entry_price = None
            self.tp = None
            self.sl = None
            self.entry_index = None

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["Entry_Time"] = pd.to_datetime(trades_df["Entry_Time"])
            trades_df["Exit_Time"] = pd.to_datetime(trades_df["Exit_Time"])
        return trades_df


def evaluate_metrics(trades: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
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


def build_mark_to_market_equity(candles: pd.DataFrame,
                                trades: pd.DataFrame,
                                initial_capital: float = 100000.0,
                                position_size: float = 1.0,
                                commission_per_trade: float = 0.0) -> pd.DataFrame:
    """
    Build candle-by-candle equity with mark-to-market (unrealized PnL while trade is open)
    and realized PnL applied on exit (minus commission).
    - candles: OHLC DataFrame with 'Datetime' and 'Close'.
    - trades: tradebook with 'Entry_Time','Exit_Time','Entry','Exit','Side','PnL' (PnL optional).
    Returns equity_df with columns ['Datetime','Equity'] (hourly / original timeframe).
    """
    if candles is None or candles.empty:
        raise ValueError("candles must be provided (your original OHLC DataFrame).")

    c = candles.copy().reset_index(drop=True)
    c['Datetime'] = pd.to_datetime(c['Datetime'])
    c = c.sort_values('Datetime').reset_index(drop=True)
    closes = c['Close'].values
    idx = pd.DatetimeIndex(c['Datetime'])

    trade_list = []
    if trades is not None and not trades.empty:
        tdf = trades.copy()
        for col in ['Entry_Time', 'Exit_Time']:
            if col in tdf.columns:
                tdf[col] = pd.to_datetime(tdf[col], errors='coerce')
        tdf = tdf.dropna(subset=['Entry_Time', 'Exit_Time']).reset_index(drop=True)

        for _, tr in tdf.iterrows():
            entry_time = tr['Entry_Time']
            exit_time = tr['Exit_Time']
            try:
                entry_pos = idx.get_indexer([entry_time], method=None)[0]
            except Exception:
                entry_pos = idx.get_indexer([entry_time], method='pad')[0]
            try:
                exit_pos = idx.get_indexer([exit_time], method=None)[0]
            except Exception:
                exit_pos = idx.get_indexer([exit_time], method='pad')[0]

            if entry_pos < 0 or exit_pos < 0:
                continue
            side = 1 if str(tr.get('Side', '')).lower().startswith('l') else -1
            entry_price = float(tr.get('Entry', np.nan))
            exit_price = float(tr.get('Exit', np.nan))
            if not pd.isna(tr.get('PnL')):
                pnl = float(tr.get('PnL'))
            else:
                if not (np.isnan(exit_price) or np.isnan(entry_price)):
                    pnl = (exit_price - entry_price) * side
                else:
                    pnl = np.nan
            trade_list.append({
                'entry_idx': int(entry_pos),
                'exit_idx': int(exit_pos),
                'entry_price': entry_price,
                'exit_price': exit_price,
                'side': side,
                'pnl': pnl
            })

    n = len(c)
    realized_by_candle = np.zeros(n, dtype=float) 
    for tr in trade_list:
        ex = tr['exit_idx']
        pnl_effect = tr['pnl'] * position_size if not np.isnan(tr['pnl']) else 0.0
        pnl_effect = pnl_effect - commission_per_trade
        realized_by_candle[ex] += pnl_effect

    equity = np.zeros(n, dtype=float)
    realized_cum = 0.0

    entry_map = {tr['entry_idx']: tr for tr in trade_list}
    exit_map = {tr['exit_idx']: tr for tr in trade_list}

    open_trade = None
    for i in range(n):
        if i in entry_map:
            open_trade = entry_map[i].copy()
        if i in exit_map:
            realized_cum += realized_by_candle[i]
            if open_trade is not None and open_trade['entry_idx'] == exit_map[i]['entry_idx']:
                open_trade = None

        unrealized = 0.0
        if open_trade is not None:
            if i <= open_trade['exit_idx']:
                curr_close = float(closes[i])
                unrealized = (curr_close - float(open_trade['entry_price'])) * float(open_trade['side']) * position_size

        equity[i] = initial_capital + realized_cum + unrealized

    equity_df = pd.DataFrame({
        'Datetime': c['Datetime'],
        'Equity': equity
    })
    return equity_df

def compute_hourly_and_daily_returns(equity_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """
    From hourly equity_df (Datetime, Equity), compute:
      - hourly_returns Series (indexed by Datetime)
      - daily_returns Series (resampled EOD, indexed by date)
      - daily_equity_df (Datetime (date), Equity_end_of_day)
    Returns (hourly_returns, daily_returns, daily_equity_df)
    """
    eq = equity_df.copy()
    eq['Datetime'] = pd.to_datetime(eq['Datetime'])
    eq = eq.sort_values('Datetime').set_index('Datetime')
    # Hourly returns (intraday) —
    hourly_returns = eq['Equity'].pct_change().fillna(0).astype(float)

    # Daily end-of-day equity
    daily_equity = eq['Equity'].resample('D').last().ffill()
    # If there're days with no data at beginning, drop leading NaNs
    daily_equity = daily_equity.dropna()
    # Daily returns
    daily_returns = daily_equity.pct_change().dropna().astype(float)

    # Prepare daily_equity_df for saving
    daily_equity_df = daily_equity.reset_index()
    daily_equity_df.columns = ['Datetime', 'Equity']

    return hourly_returns, daily_returns, daily_equity_df


def generate_quantstats_report(returns: pd.Series,
                               output_html: str,
                               benchmark: Optional[pd.Series] = None,
                               freq: str = 'H') -> Dict[str, Any]:
    """
    Generate a QuantStats HTML report and return key metrics dict.
    """
    if not isinstance(returns, pd.Series):
        raise ValueError("returns must be a pandas Series")
    returns.index = pd.to_datetime(returns.index)
    returns.name = 'strategy_returns'

    returns = returns.fillna(0).astype(float)

    try:
        qs.reports.html(returns, benchmark=benchmark, output=output_html)
    except Exception as e:
        print("quantstats full HTML report failed:", str(e))
    metrics = {}
    try:
        metrics['sharpe'] = qs.stats.sharpe(returns)
        metrics['cagr'] = qs.stats.cagr(returns)
        metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
        metrics['annualized_vol'] = qs.stats.volatility(returns)
    except Exception as e:
        print("quantstats metric extraction failed:", str(e))
    return metrics



if __name__ == "__main__":
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "backtest_data_1H.csv")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: data file not found at {DATA_PATH}")
        raise SystemExit(1)
    df = pd.read_csv(DATA_PATH)
    if "Datetime" not in df.columns:
        possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible:
            df.rename(columns={possible[0]: "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # run strategy
    strat = ScalpingStrategyIterB()
    trades = strat.generate_signals(df)

    out_folder = os.path.join(os.path.dirname(__file__), "..", "backtest")
    output_quant_s = os.path.join(out_folder, "output_quant_s")
    os.makedirs(output_quant_s, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    out_file = os.path.join(out_folder, "Scalping_tradebook_iterB_1H.csv")
    trades.to_csv(out_file, index=False)
    print(f"Tradebook saved → {out_file}")

    # evaluate quick trade metrics (your old summary)
    tb, stats = evaluate_metrics(trades)
    if tb is not None and not tb.empty:
        print(pd.DataFrame([stats]).T)
    else:
        print("No completed trades to show statistics.")

    # ---------------- QuantStats pipeline ----------------
    # default parameters — change these as required
    INITIAL_CAPITAL = 100000.0
    POSITION_SIZE = 1.0
    COMMISSION_PER_TRADE = 0.0  # absolute commission applied at trade exit
    QS_REPORT_PATH = os.path.join(output_quant_s, "Scalping_quantstats_report_iterB_1H.html")
    # EQUITY_CSV = os.path.join(out_folder, "Scalping_equity_iterB_1H.csv")
    # RETURNS_CSV = os.path.join(out_folder, "Scalping_returns_iterB_1H.csv")

        # Build mark-to-market equity
    equity_df = build_mark_to_market_equity(df, trades,
                                            initial_capital=INITIAL_CAPITAL,
                                            position_size=POSITION_SIZE,
                                            commission_per_trade=COMMISSION_PER_TRADE)
    # equity_df.to_csv(EQUITY_CSV, index=False)
    # print(f"Equity (hourly MTM) saved → {EQUITY_CSV}")

    # Compute hourly + daily returns
    hourly_returns, daily_returns, daily_equity_df = compute_hourly_and_daily_returns(equity_df)
    # Save hourly returns
    hourly_df = hourly_returns.reset_index()
    hourly_df.columns = ['Datetime', 'Return']
    # hourly_df.to_csv(RETURNS_CSV, index=False)
    # print(f"Hourly returns saved → {RETURNS_CSV}")

    # Save daily equity and daily returns
    # DAILY_EQUITY_CSV = os.path.join(out_folder, "Scalping_daily_equity_iterB_1H.csv")
    # DAILY_RETURNS_CSV = os.path.join(out_folder, "Scalping_daily_returns_iterB_1H.csv")
    # daily_equity_df.to_csv(DAILY_EQUITY_CSV, index=False)
    # daily_returns.reset_index().rename(columns={'index': 'Datetime', 0: 'Return'}).to_csv(DAILY_RETURNS_CSV, index=False)
    # print(f"Daily equity saved → {DAILY_EQUITY_CSV}")
    # print(f"Daily returns saved → {DAILY_RETURNS_CSV}")

    # Use daily_returns for QuantStats (keeps annualization stable)
    quant_metrics = generate_quantstats_report(daily_returns, output_html=QS_REPORT_PATH, benchmark=None, freq='D')
    print("QuantStats quick metrics:")
    print(quant_metrics)
    print(f"QuantStats HTML report attempted → {QS_REPORT_PATH}")