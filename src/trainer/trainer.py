import numpy as np
from trainer.data import DataProvider

class Trainer:
    def __init__(self, data:DataProvider, named_metrics) -> None:
        self._data = data
        self._model = None
        self._metrics_names, self._metrics_func = zip(*named_metrics)

    @classmethod
    def _score(cls, y_true, y_pred, metric):
        total = metric(y_true, y_pred)
        indiv = map(
            lambda ys: metric(ys[0], ys[1]),
            zip(y_true.T, y_pred.T)
        )

        return total, *indiv

    @property
    def conditions(self):
        return tuple(self._data.conditions)
    
    @property
    def durations(self):
        return tuple(self._data.durations)
    
    @property
    def labels_duration(self):
        return self.durations
    
    @property
    def labels_class(self):
        return self._data.class_labels
    
    @property
    def labels_condition(self):
        return self.conditions
    
    @property
    def labels_metric(self):
        return tuple(self._metrics_names)
    
    @property
    def labels_eval(self):
        return tuple(['TOTAL'] + list(self.labels_class))
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model


    def train(self, condition, duration):
        self._model.fit(*self._data.get_train(condition, duration))
        return self
    
    

    def eval(self, condition, duration):
        x_eval, y_eval = self._data.get_eval(condition, duration)
        y_pred = self._model.predict(x_eval)

        return tuple(
            self._score(y_eval, y_pred, metric)
            for metric in self._metrics_func
        )

    def eval_in_domain(self):
        # shape (conditions, durations, metrics, classes+1)
        return np.array(tuple(
            tuple(
                self.train(c, d).eval(c, d)
                for d in self.durations
            )
            for c in self.conditions
        ))
    
    def eval_cross_domain(self):
        assert len(self.conditions) == 2

        # shape (2, durations, metrics, classes+1)
        return np.array(tuple(
            tuple(
                self.train(c_train, d).eval(c_eval, d)
                for d in self.durations
            )
            for (c_train, c_eval) in (
                self.conditions,
                reversed(self.conditions)
            )
        ))

    def eval_in_and_cross_domain(self):
        results = list()
        for d in self.durations:
            intermediate = list()
            for c in self.conditions:
                self.train(c, d)
                intermediate.append(tuple(
                    self.eval(ce, d)
                    for ce in self.conditions
                ))
            results.append(intermediate)
        return np.array(results)
    
    def eval_all_domains(self):
        result = list()
        for d in self.durations:
            x_train, y_train = map(np.concat, zip(*(
                self._data.get_train(c, d)
                for c in self.conditions
            )))
            self._model.fit(x_train, y_train)

            evaluation = [
                self.eval(c, d)
                for c in self.conditions
            ]

            x_eval, y_eval = map(np.concat, zip(*(
                self._data.get_eval(c, d)
                for c in self.conditions
            )))
            y_pred = self._model.predict(x_eval)
            evaluation.append(
                tuple(
                    self._score(y_eval, y_pred, metric)
                    for metric in self._metrics_func
                )
            )

            result.append(evaluation)

        return np.array(result)
    
    def eval_in_domain_cross_validation(self, domain, sets=['train', 'test']):
        results = dict()
        for d in self.durations:
            sample = self._data._get(domain, d)
            x, y, t = map(np.concat, zip(*tuple(sample[s] for s in sets)))
            y = (y.mean(axis=1) > 0.5).astype(float)
            for token in set(t):
                if not token in results:
                    results[token] = dict()
                print(token)
                test_mask = (t == token)
                train_mask = np.logical_not(test_mask)
                self._model.fit(x[train_mask], y[train_mask])
                y_pred = self._model.predict(x[test_mask])
                results[token][d] = tuple(
                    self._score(y[test_mask], y_pred, metric)
                    for metric in self._metrics_func
                )

        return results

