from functools import reduce
import itertools
from typing import Callable, List
from postprocessing_utils import Gender
from preprocess_utils import load_jsonb
import numpy as np
import scipy.sparse as sp


class Dataset:
    def __init__(self, X: sp.csr_matrix, y):
        self.X = X
        self.y = y


# Currently works as in-memory dataset
class ArticleTFIDFDataset(Dataset):
    def __init__(
        self,
        X: sp.csr_matrix,
        a_file: str,
        transform_X: Callable | None = None,
        transform_Y: Callable | None = None,
        filter: Callable | None = None,
        start: int = 0,
        end: int = None,
    ):
        if end is not None:
            y = itertools.islice(load_jsonb(a_file), start, end)
        else:
            y = load_jsonb(a_file)

        if transform_X:
            X = transform_X(X)

        if transform_Y:
            y = transform_Y(y)

        if filter:
            X, y = filter(X, y)

        super().__init__(X, list(y))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def remove_y_None(X, y):
    y = list(y)
    indexes = [i for i, y_i in enumerate(y) if y_i is not None]
    return X[indexes], [y[i] for i in indexes]


def filter_by_set(y_set: set):
    def _keep_top_unique(X, y):
        # Needs to remove None values from y
        indexes = [i for i, y_i in enumerate(y) if y_i in y_set]
        return X[indexes], [y[i] for i in indexes]

    return _keep_top_unique


def get_top_unique(y, top=50):
    y_unique = list(set(y))
    y_unique.sort(key=y.count, reverse=True)
    return y_unique[:top]


def select_col(col_name: str):
    def _select_col(y):
        return [y[col_name] for y in y]

    return _select_col


def gender_transform(y):
    return np.array(
        [
            [
                sum(gender == Gender.MAN for gender in genders),
                sum(gender == Gender.WOMAN for gender in genders),
            ]
            if genders is not None
            else None
            for genders in y
        ]
    )


def get_gender(genders: List[Gender] | None):
    if genders is None:
        return None

    if all(gender == Gender.MAN for gender in genders):
        return Gender.MAN
    elif all(gender == Gender.WOMAN for gender in genders):
        return Gender.WOMAN
    return Gender.MIXED


def gender_type_transform(y):
    return np.array([get_gender(genders) for genders in y])


def sequence_transform(seq: list[Callable]):
    def _sequence_transform(y):
        return reduce(lambda x, f: f(x), seq, y)

    return _sequence_transform


def sequence_filter(seq: list[Callable]):
    def _sequence_filter(X, y):
        for f in seq:
            X, y = f(X, y)
        return X, y

    return _sequence_filter
