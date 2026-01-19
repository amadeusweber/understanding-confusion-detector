import glob
import os
import pandas as pd
from dataloader.base import DataLoaderBase

class DataLoaderRecording(DataLoaderBase):
    def __init__(self, source:str, file:str) -> None:
        super().__init__()
        self._source = source 
        self._file =  file
    
    def load(self, token:str, condition:str) -> pd.DataFrame:
        pattern = os.path.join(
            self._source,
            token,
            self._file % condition
        )
        files = glob.glob(pattern)

        assert len(files) == 1, f"Cannot find {pattern}"
        return pd.read_csv(files[0], on_bad_lines='skip')