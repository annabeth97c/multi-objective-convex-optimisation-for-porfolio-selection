import pandas as pd
import numpy as np
from base_dataset import BaseDataset


class FilteredDataset(BaseDataset):
	def __init__(self, prices_path: str):
		self.prices_path = prices_path
		self.symbols = []
		self.price_changes_array = None
		self.time_steps_count = 0
		self.stocks_count = 0

	def load_dataset(self) -> np.ndarray:
		df = pd.read_csv(self.prices_path, parse_dates=['date'])

		lo, hi = df['daily_return'].quantile([0.01, 0.99])
		df['daily_return'] = df['daily_return'].clip(lo, hi)

		ranges = (
				df
					.groupby('symbol')['date']
					.agg(min_date='min', max_date='max')
					.reset_index()
		)
		ranges['range_key'] = (
				ranges['min_date'].dt.strftime('%Y-%m-%d')
				+ '_'
				+ ranges['max_date'].dt.strftime('%Y-%m-%d')
		)
		best_key = ranges['range_key'].value_counts().idxmax()
		symbols = ranges.loc[ranges['range_key'] == best_key, 'symbol'].tolist()

		df_sub = df[df['symbol'].isin(symbols)]
		pivot = (
				df_sub
					.pivot(index='date', columns='symbol', values='daily_return')
					.dropna()
		)

		self.symbols = pivot.columns.tolist()
		self.price_changes_array = pivot.values
		self.time_steps_count, self.stocks_count = self.price_changes_array.shape
		return self.price_changes_array

	def get_stocks_count(self) -> int:
		return self.stocks_count

	def get_time_steps_count(self) -> int:
		return self.time_steps_count

	def get_symbols(self) -> list:
		return self.symbols