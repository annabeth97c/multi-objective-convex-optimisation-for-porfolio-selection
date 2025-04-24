from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseDataset(ABC):
    @abstractmethod
    def load_dataset(self) -> np.array:
        """Load and return the data as np array"""
        pass

    def get_stocks_count(self) -> int:
      """Return number of stocks in dataset"""
      pass

    def get_time_steps_count(self) -> int:
      """Return number of time steps in dataset"""
      pass

    def get_symbols(self) -> list:
      """Return list of symbols in dataset"""
      pass