import pandas as pd
from RandomizedSampling import RandomizedSampling, RandomizedSamplingParams
from sklearn.pipeline import Pipeline
from sklearn import preprocessing, svm
from sklearn.naive_bayes import GaussianNB

import datetime

svc_params = [
    {
        'svm__kernel': ['rbf'],
        'svm__gamma': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))],
        'svm__C': [x ** y for x, y in zip([2] * 31, range(-15, 16, 1))]
    }
]

nb_params = [
    {
        'nb__priors': [None]
    }
]

params_test = [
    {
        'svm__kernel': ['rbf'],
        'svm__gamma': [1],
        'svm__C': [1]
    }
]

normalize_svm_pipeline = Pipeline(steps=[('normalize', preprocessing.Normalizer()), ('svm', svm.SVC())])
normalize_nb_pipeline = Pipeline(steps=[('normalize', preprocessing.Normalizer()), ('nb', GaussianNB())])

def main():
    rawdata = pd.read_csv("../csv/expanded_dataset_v1.csv", encoding='utf8')
    experiment_params = RandomizedSamplingParams()
    default_experiment = RandomizedSampling(experiment_params)
    # result = default_experiment.run_experiment(rawdata, normalize_svm_pipeline, svc_params)
    result = default_experiment.run_experiment(rawdata, normalize_nb_pipeline, nb_params)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    print(result)
    result.to_csv("nb_results-{}.csv".format(time))


if __name__ == "__main__":
    main()

