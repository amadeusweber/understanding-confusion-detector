import numpy as np

FUNCTIONALS = (
    # mean
    lambda x: np.mean(x, axis=1),
    # standard deiviation
    lambda x: np.std(x, axis=1),
    # blink frames count
    lambda x: np.expand_dims(np.sum(x[...,-1], axis=1), axis=1)
)

class FlatWrapper:
    def __init__(self, model) -> None:
        self._model = model
    
    def _flatten(self, arr):
        return arr.reshape(arr.shape[0], -1)
    
    def fit(self, X,  y, *args, **kwargs):
        return self._model.fit(self._flatten(X), y, *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return self._model.predict(self._flatten(X), *args, **kwargs)
    
class AvgWrapper:
    def __init__(self, model) -> None:
        self._model = model
    
    def _avg(self, arr):
        return arr.mean(axis=1)
    
    def fit(self, X,  y, *args, **kwargs):
        return self._model.fit(self._avg(X), y, *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return self._model.predict(self._avg(X), *args, **kwargs)
    
class FunctionalWrapper:
    def __init__(self, model, functionals=FUNCTIONALS) -> None:
        self._model = model
        self._functionals = functionals
    
    def _to_functional(self, arr):
        assert len(arr.shape) == 3
        n, t, f = arr.shape

        if t > 1:
            # In window
            return np.hstack(tuple(
                f(arr)
                for f in self._functionals
            ))
        return arr[:,0]
    
    def fit(self, X,  y, *args, **kwargs):
        return self._model.fit(self._to_functional(X), y, *args, **kwargs)
    
    def predict(self, X, *args, **kwargs):
        return self._model.predict(self._to_functional(X), *args, **kwargs)