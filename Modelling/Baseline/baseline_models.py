from pathlib import Path
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import pickle


class Model:
    def __init__(self):
        raise NotImplementedError

    def fit(self, features, labels):
        raise NotImplementedError

    def predict(self, features):
        raise NotImplementedError

    def score(self, features, labels):
        raise NotImplementedError

    def save(self, path: Path):
        raise NotImplementedError

    def load(self, path: Path):
        raise NotImplementedError

    def set_params(self, **params):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError


class PickleModel(Model):
    def __init__(self, model):
        self.model = model

    def fit(self, features, labels):
        self.model = self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.model.predict(features)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)

        return self

    def score(self, features, labels):
        return self.model.score(features, labels)

    def set_params(self, **params):
        self.model.set_params(**params)

    def get_params(self):
        return self.model.get_params()


class ServerModel(PickleModel):
    def __init__(self):
        model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=8)
        super().__init__(model)

    def score(self, features, labels):
        return classification_report(labels, self.predict(features))


class CategoryModel(PickleModel):
    def __init__(self):
        model = LogisticRegression(
            max_iter=1000, verbose=1, multi_class="multinomial", n_jobs=8
        )
        super().__init__(model)

    def score(self, features, labels):
        return classification_report(labels, self.predict(features))

    def predict(self, features):
        return super().predict(features)

    def fit(self, features, labels):
        return super().fit(features, labels)


# Ccan't use TransfomerRegressor because it doesn't support classifictation output
class AuthorModel(Model):
    def __init__(self):
        self.model = MultiOutputClassifier(
            LogisticRegression(multi_class="multinomial"), vebose=1, n_jobs=8
        )
        self.binarizer = MultiLabelBinarizer(sparse_output=True)

    def fit(self, features, labels):
        labels = self.binarizer.fit_transform(labels)
        self.model = self.model.fit(features, labels)
        return self

    def predict(self, features):
        return self.binarizer.inverse_transform(self.model.predict(features))

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
            pickle.dump(self.binarizer, f)


class AuthorGenderModel(PickleModel):
    def __init__(self):
        model = MultiOutputRegressor(
            LogisticRegression(max_iter=1000, verbose=1, n_jobs=8)
        )
        super().__init__(model)

    def score(self, features, labels):
        res = self.predict(features)
        labels = np.array(labels)
        return f"Men:\n{classification_report(labels[:, 0], res[:, 0])}\nWomen:\n{classification_report(labels[:, 1], res[:, 1])}"


class AuthorGenderModelSimple(PickleModel):
    def __init__(self):
        model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=8)
        super().__init__(model)

    def score(self, features, labels):
        return classification_report(labels, self.predict(features))


class DayModel(PickleModel):
    def __init__(self):
        model = LogisticRegression(max_iter=1000, verbose=1, n_jobs=8)
        super().__init__(model)

    def score(self, features, labels):
        return classification_report(labels, self.predict(features))
