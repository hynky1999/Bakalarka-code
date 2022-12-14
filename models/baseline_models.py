from pathlib import Path
from typing import List
from postprocessing_utils import Gender
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np


class ServerModel:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)


class CategoryModel:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, features, labels):
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)


# Ccan't use TransfomerRegressor because it doesn't support classifictation output
class AuthorModel:
    def __init__(self):
        self.model = MultiOutputClassifier(LogisticRegression())
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def fit(self, features, labels: List[List[str]]):
        labels = self.binarizer.fit_transform(labels)
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.binarizer.inverse_transform(self.model.predict(features))


class GenderTransformer:
    def transform(self, X):
        return np.array(
            [
                [
                    sum(gender == Gender.MAN for gender in genders),
                    sum(gender == Gender.WOMAN for gender in genders),
                ]
                for genders in X
            ]
        )


class AuthorGenderModel:
    def __init__(self):
        self.model = MultiOutputRegressor(LogisticRegression())
        self.transfomer = GenderTransformer()

    def fit(self, features, labels):
        labels = self.transfomer.transform(labels)
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)


class DateModel:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, features, labels):
        labels = np.array(labels)
        self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)
