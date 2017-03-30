from sklearn.pipeline import Pipeline
from sklearn import preprocessing, svm

normalize_svm_pipeline = Pipeline(steps=[('normalize', preprocessing.Normalizer()), ('svm', svm.SVC())])
