import pandas as pd
from pathlib import Path

input_csv = Path(r"d:\Backtest_str-8_13_21\data\backtest_data.csv")  # change if your 1min file has a different name
output_csv = input_csv.with_name(input_csv.stem + "_5Min.csv")

df = pd.read_csv(input_csv)
# parse datetime (assume ISO with Z) and coerce numeric OHLC
df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# drop rows with missing OHLC (removes rows like ,,,,,0.0)
df = df.dropna(subset=['Open', 'High', 'Low', 'Close'], how='any')

# resample to 1H
df = df.set_index('Datetime')
res = df.groupby(pd.Grouper(freq='5Min')).agg({
    "Symbol": "first",
    "Contract_month": "first",
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum"
}).dropna(how='any').reset_index()

# keep requested column order and ISO-like Datetime formatting with Z
res = res[["Datetime", "Symbol", "Contract_month", "Open", "High", "Low", "Close", "Volume"]]
res['Datetime'] = res['Datetime'].dt.strftime('%Y-%m-%dT%H:%M:%S.000000Z')

res.to_csv(output_csv, index=False)
print(f"Wrote {output_csv} rows={len(res)}")