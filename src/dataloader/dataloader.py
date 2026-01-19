import pandas as pd
import os
from functools import cache

class LazyCachedDataLoader:
    def __init__(self, source:str, conditions:set, states:set, tokens:set):
        self._source = source
        self._conditions = conditions
        self._states = states
        self._tokens = tokens

        for condition in self._conditions:
            for state in self._states:
                for token in self._tokens:
                    f = self._filename(token, condition, state)
                    if not os.path.exists(f):
                        raise FileNotFoundError(f)

    @classmethod
    def _segment(cls, df:pd.DataFrame, threshold=1):
        df = df.sort_values('frame').reset_index(drop=True)
        segment_idcs = (df.frame.diff().fillna(1) > threshold).cumsum()
        _, segments = zip(*df.groupby(segment_idcs))
        return segments

    @property
    @cache
    def columns(self):
        return self.get(
            next(iter(self._tokens)),
            next(iter(self._conditions)),
            next(iter(self._states))
        ).columns
    
    @cache
    def _filename(self, token, condition, state) -> str:
        assert token in self._tokens
        assert condition in self._conditions
        assert state in self._states

        return os.path.join(
            self._source,
            condition,
            state,
            token + '.csv'
        )
    
    @cache
    def _filename_state(self, token, condition, state) -> str:
        assert token in self._tokens
        assert condition in self._conditions
        assert state in self._states

        return os.path.join(
            self._source,
            condition,
            state,
            token + '.csv'
        )
    
    @cache
    def get(self, token:str, condition:str, state:str) -> pd.DataFrame:
        return pd.read_csv(
            self._filename(token, condition, state)
        ).assign(token=token,condition=condition,state=state)
    
    def get_recording(self, token:str, condition:str) -> pd.DataFrame:
        return pd.concat(
            self.get(token, condition, s)
            for s in self._states
        )
    
    def get_windows(self, token:str, condition:str, state:str) -> tuple[pd.DataFrame]:
        return self._segment(self.get(token, condition, state))
    
    def get_full_set(self):
        return pd.concat(
            self.get(t,c,s)
            for t in self._tokens
            for c in self._conditions
            for s in self._states
        )
    
    def get_domain(self, condition:str, state:str) -> pd.DataFrame:
        return pd.concat(
            self.get(t, condition, state)
            for t in self._tokens
        )