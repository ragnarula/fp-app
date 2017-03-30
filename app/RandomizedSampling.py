import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


class RandomizedSamplingParams:
    def __init__(self):
        self.lower = 0.1
        self.upper = 0.8
        self.step = 0.1
        self.n = 10


class RandomizedSampling:

    def __init__(self, params):
        self.params = params

    def search_train(self, data, c, pipe, params):
        sample_size = int(c * len(data.index))
        results = []

        search = GridSearchCV(pipe, params, n_jobs=-1, verbose=1, scoring='f1_macro', cv=10)

        for i in xrange(self.params.n):
            print("Run {} of {} with {}% of data".format(i + 1, self.params.n, c))
            rows = np.random.choice(data.index.values, sample_size)

            train = data.ix[rows]
            test = data.ix[~data.index.isin(train.index)]

            indexes = train.axes[0]
            X = train.drop(['Class'], axis=1)
            Y = train['Class']
            classifier = search.fit(X, Y)

            x = test.drop(['Class'], axis=1)
            y = test['Class']

            y_pred = classifier.predict(x)

            res = {
                'indexes': indexes,
                'classifier': classifier,
                'y': y,
                'y_pred': y_pred,
                'N': self.params.n,
                'c': c
            }
            results.append(res)
        return results

    @staticmethod
    def metric_from_result(res, metric_func, **kwargs):
        return metric_func(res['y'], res['y_pred'], **kwargs)

    def run_experiment(self, data, pipe, params):

        cs = np.arange(self.params.lower, self.params.upper, self.params.step)
        rows = []
        for c in cs:
            run_res = self.search_train(data, c, pipe, params)
            for r in run_res:
                cm = self.metric_from_result(r, confusion_matrix)

                r_dict = {
                    "accuracy": self.metric_from_result(r, accuracy_score),
                    "f1": self.metric_from_result(r, f1_score, average='macro'),
                    "precision": self.metric_from_result(r, precision_score, average='macro'),
                    "recall": self.metric_from_result(r, recall_score, average='macro'),
                    "c": c,
                    "best_params": str(r['classifier'].best_params_)
                }

                for i in range(len(cm)):
                    for j in range(len(cm[0])):
                        key = "cm_{}{}".format(i, j)
                        value = cm[i][j]
                        r_dict.update({key: value})

                rows.append(r_dict)
        return pd.DataFrame(rows)
