import numpy as np
from pathlib import Path
from typing import List, Callable, Any
import pickle
import json
from sklearn.base import BaseEstimator
import logging

# Set logging output to file


class PickledTransform(BaseEstimator):
    """
    For some reason, doesn't work with parallel processing
    but if I use the multiprocessing library as backend, it works
    """

    def __init__(
        self,
        transformer,
        cache_path,
        hash_keys,
        fitted_doc_hash=None,
    ):
        self.transformer = transformer
        self.hash_keys = hash_keys
        self.fitted_doc_hash = fitted_doc_hash
        self.cache_path = cache_path
        self.cache = MemoryCache(cache_path)

    def _hash_vec_docs(self, docs):
        params = self.transformer.get_params()
        hashed_params = {k: params[k] for k in self.hash_keys}
        params_str = json.dumps(hashed_params, sort_keys=True)
        doc_str = "".join([str(d)[0] for d in docs])
        class_str = self.transformer.__class__.__name__
        hsh = hash(params_str + doc_str + class_str)
        logging.info(
            f"""
        Hashing with params: {params_str}
        and docs: {doc_str}
        and class: {class_str}
        result: {hsh}
        """
        )
        return str(hsh)

    def fit(self, X, y=None):
        hsh = self._hash_vec_docs(X)
        self.fitted_doc_hash = hsh
        self.transformer = self.cache(
            lambda: self.transformer.fit(X, y), Path(hsh) / "fit.pkl"
        )
        return self

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        if self.fitted_doc_hash is None:
            raise ValueError("Must fit before transform")

        transformed = self.cache(
            lambda: self.transformer.transform(X),
            Path(self.fitted_doc_hash) / f"{self._hash_vec_docs(X)}.pkl",
        )

        params = self.transformer.get_params()
        hashed_params = {k: params[k] for k in self.hash_keys}

        return transformed


class MemoryCache:
    def __init__(self, cache_path: Path | None):
        self.cache_path = cache_path

    def __call__(self, fc: Callable[[], Any], name: Path | str):
        if self.cache_path == None:
            return fc()

        file = self.cache_path / name
        if file.exists():
            return pickle.load(file.open("rb"))

        res = fc()
        if not file.parent.exists():
            file.parent.mkdir(parents=True)

        pickle.dump(res, file.open("wb"))
        return res

    def get(self, name: Path | str):
        if self.cache_path == None:
            raise ValueError("Cache path not set")

        file = self.cache_path / name
        if not file.exists():
            raise ValueError(f"File {file} not found")

        return pickle.load(file.open("rb"))
