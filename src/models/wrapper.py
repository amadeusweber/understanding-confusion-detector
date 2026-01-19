import numpy as np

class FlatWrapper:
    def __init__(self, model) -> None:
        self._model = model
    
    def _flatten(self, arr):
        return arr.reshape(arr.shape[0], -1)
    
    def fit(self, X,  y, *args, **kwargs):
        return self._model.fit(self._flatten(X), y, *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return self._model.predict(self._flatten(X), *args, **kwargs)