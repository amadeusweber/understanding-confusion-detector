import os
import pandas as pd
from dataloader.base import DataLoaderBase

class DataLoaderState(DataLoaderBase):
    def __init__(self, source:str) -> None:
        super().__init__()
        self._source = source
    
    def load(self, token:str, condition:str, state:str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(
            self._source,
            condition,
            state,
            token + '.csv'
        ))