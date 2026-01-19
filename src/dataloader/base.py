import abc
import pandas as pd
class DataLoaderBase(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abc.abstractmethod
    def load(self, *args, **kwargs) -> pd.DataFrame:
        pass