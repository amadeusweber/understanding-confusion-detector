import pandas as pd
from dataloader.base import DataLoaderBase
from dataloader.recording import DataLoaderRecording
from dataloader.state import DataLoaderState

class DataLoaderFull(DataLoaderBase):
    def __init__(self, tokens:set, conditions:set, states:set, 
                 source_state:str, source_record:str, file_record:str,
                 col_join:str='frame') -> None:
        super().__init__()
        self._tokens = set(tokens)
        self._conditions = set(conditions)
        self._states = set(states)
        self._col_join = col_join
        self._dl_state = DataLoaderState(
            source=source_state
        )
        self._dl_record = DataLoaderRecording(
            source = source_record,
            file = file_record
        )

    def load(self, token:str, condition:str) -> pd.DataFrame:
        data = self._dl_record.load(token, condition)
        # write state data
        for state in self._states:
            data[f"state_{state}"] = data[self._col_join].isin(
                self._dl_state.load(token, condition, state).get(self._col_join, [])
            )
        
        return data.assign(token=token, condition=condition)

    def load_full(self) -> pd.DataFrame:
        return pd.concat(
            pd.concat(
                self.load(token, condition)
                for condition in self._conditions
            )
            for token in self._tokens
        )