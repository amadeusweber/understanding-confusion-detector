import os
import numpy as np
from sklearn.model_selection import train_test_split

DATASETS = ['train', 'test', 'validation']

def resample(x, y, t, class_dist=(0.2, 0.1)):
    # get target distribution (no, understanding, confusion)
    ## Assume now multi-class samples
    dist = np.array([*class_dist] + [1-sum(class_dist)])

    xs = list()
    ys = list()
    ts = list()
    for token in set(t):
        selection = t == token
        yt = y[selection]
        yt = (yt.mean(axis=1) > 0.5).astype(float)
        n = len(yt)
        # get actual distribution
        y_class_samples = yt.sum(axis=0)
        y_samples = np.array([*y_class_samples] + [n - sum(y_class_samples)])
        y_dist = y_samples / n

        # compare actual and target distributions
        # find number of total samples to math minimum coverage
        target = np.argmin((y_dist - dist)/dist)
        total_samples = y_samples[target] / dist[target]

        # number of samples per class and no class
        n_samples = (total_samples * dist).astype(int)

        # append class samples
        for c in range(len(class_dist)):
            mask = yt[:,c].astype(bool)
            samples = np.random.choice(sum(mask), size=n_samples[c], replace=False)
            xs.append(x[selection][mask][samples])
            ys.append(y[selection][mask][samples])
            ts.append(t[selection][mask][samples])
        
        # append no class samples
        mask = np.logical_not(yt.any(axis=1))
        samples = np.random.choice(sum(mask), size=n_samples[-1], replace=False)
        xs.append(x[selection][mask][samples])
        ys.append(y[selection][mask][samples])
        ts.append(t[selection][mask][samples])

    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ts)

class DataProvider:
    def __init__(self, dir_source:str, conditions, durations, labels) -> None:
        self._durations = set(durations)
        self._conditions = set(conditions)
        self._labels = tuple(labels)

        self._data = dict()
        for c in conditions:
            temp = self._data[c] = dict()
            for d in self._durations:
                temp[d] = dict()
                for s in DATASETS:
                    data = np.load(
                        os.path.join(dir_source, f"{s}_{c}_{d}s.npz"),
                        allow_pickle=True
                    )
                    x = np.nan_to_num(data['x'])
                    y = data['y']
                    t = data['t']
                    temp[d][s] = x, y, t

    @property
    def durations(self):
        return set(self._durations)
    
    @property
    def conditions(self):
        return set(self._conditions)
    
    @property
    def class_labels(self):
        return tuple(self._labels)
    
    def _get(self, condition, duration):
        assert condition in self._conditions
        assert duration in self._durations
        return self._data[condition][duration]

    def get_train(self, condition, duration, agg_window:bool=True):
        x,y,t = self._get(condition, duration)['train']
        if agg_window:
            y = (y.mean(axis=1) > 0.5).astype(float)
        return x, y
    
    def get_eval(self, condition, duration, agg_window:bool=True):
        x,y,t = self._get(condition, duration)['test']
        if agg_window:
                y = (y.mean(axis=1) > 0.5).astype(float)
        return x, y
    
    def get_validation(self, condition, duration, agg_window:bool=True):
        x,y,t = self._get(condition, duration)['validation']
        if agg_window:
            y = (y.mean(axis=1) > 0.5).astype(float)
        return x, y
    
class DataProviderInterParticipant(DataProvider):
    def __init__(self, dir_source:str, conditions, durations, labels, test_size=.3, sets=['train', 'test'], class_dist=(0.3, 0.15), downsampling_sf=True) -> None:
        super().__init__(dir_source, conditions, durations, labels)

        self._sampled_data = dict()
        for c in self.conditions:
            self._sampled_data[c] = dict()
            for d in self.durations:
                self._sampled_data[d] = dict()
                x, y, t = self._combine(c, d, sets)
                if downsampling_sf and d == 0:
                    # sample every second frame for single frame data
                    x = x[::2]
                    y = y[::2]
                    t = t[::2]
                x, y, t = resample(x, y, t, class_dist)
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                self._sampled_data[c][d] = {
                    'train': (x_train, y_train, None),
                    'test': (x_test, y_test, None),
                    'validation': (None, None, None)
                }


    def _combine(self, condition, duration, sets=['train', 'test']):
        source = super()._get(condition, duration)
        x, y, t = map(np.concat, zip(*tuple(source[s] for s in sets)))
        return x, y, t

    def _get(self, condition, duration):
        assert condition in self._conditions
        assert duration in self._durations
        return self._sampled_data[condition][duration]