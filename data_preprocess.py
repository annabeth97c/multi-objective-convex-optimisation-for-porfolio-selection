import numpy as np
import pandas as pd
from base_dataset import BaseDataset


dataset_summary_csv_file_path = '/Users/anuradhachopra/Downloads/stock_dataset/dataset_summary.csv'
stock_prices_csv_file_path   = '/Users/anuradhachopra/Downloads/stock_dataset/stocks_latest/stock_prices_latest.csv'

output_filtered_csv_file_path = 'filtered_stock_prices_with_returns.csv'

df = pd.read_csv(stock_prices_csv_file_path, parse_dates=['date'])

df = (
    df
      .sort_values(['symbol', 'date'])
      .assign(
          daily_return=lambda df_: df_
            .groupby('symbol')['close_adjusted']
            .pct_change()
      )
)

# Replace infinities with NaN, then drop them
df['daily_return'].replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['daily_return'], inplace=True)

# cap any extreme returns to ±50%
df['daily_return'] = df['daily_return'].clip(lower=-0.5, upper=0.5)

df.to_csv(output_filtered_csv_file_path, index=False)
print(f"Saved filtered prices+returns to {output_filtered_csv_file_path}")


### EXAMPLE LOADING ###
df = pd.read_csv(output_filtered_csv_file_path, parse_dates=['date'])

pivot = (
    df
      .pivot(index='date', columns='symbol', values='daily_return')
      .dropna()
)

symbols = pivot.columns.tolist()
price_changes_array = pivot.values  # already shape (T, N)

print(f"P stats: min={price_changes_array.min():.4f}, "
      f"max={price_changes_array.max():.4f}, "
      f"mean={price_changes_array.mean():.6f}")

