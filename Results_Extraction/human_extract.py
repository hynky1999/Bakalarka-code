from pathlib import Path
from sklearn.metrics import cohen_kappa_score, f1_score
from itertools import combinations
from functools import cache
from datasets import load_dataset
import numpy as np

import pandas as pd

excel_name_map = {
    "server": "Predikce server",
    "authors_cum_gender": "Predikce pohlaví",
    "day_of_week": "Predikce den v týdnu",
    "category": "Predikce kategorie"
}

def get_excel_features(path: Path):
    cs = pd.read_csv(path, skiprows=3, header=0)
    genders = cs['gender.1'].dropna().to_list()
    server = cs["server.1"].dropna().to_list()
    day_of_week = cs["day of week"].dropna().to_list()
    category = cs["category.1"].dropna().to_list()
    return {
        "authors_cum_gender": genders,
        "server": server,
        "day_of_week": day_of_week,
        "category": category
    }

def get_prediction(df, features):
    def process(preds, features):
        preds = preds.to_list()[:-1]
        preds = [features.index(pred) for pred in preds]
        return preds
    
    return {name: process(df[excel_name_map[name]], features[name]) for name in excel_name_map.keys()}

def get_kappa(predictions1, predictions2, split="test_human"):
    labels = {task: get_labels(task, split) for task in predictions1.keys()}
    return {name: cohen_kappa_score(predictions1[name], predictions2[name], labels=labels[name]) for name in predictions1.keys()}

@cache
def get_true(task, dst_path="hynky/czech_news_dataset", split="test_human"):
    dst = load_dataset(dst_path, split=split)
    dst = dst.filter(lambda x: x[task] != 0)
    dst = dst.map(lambda x: {task: x[task] - 1})
    return dst[task]

def get_f1(predictions, true, split="test_human", average="macro"):
    labels = {task: get_labels(task, split) for task in predictions.keys()}
    print(labels)
    return {name: f1_score(true[name], predictions[name], labels=labels[name], average=average) for name in predictions.keys()}



def get_humans_predictions(root_path: Path, true_table_name: str, features):
    predictions = {}
    for file in root_path.glob("*.csv"):
        if file.name == true_table_name:
            continue
        df = pd.read_csv(file, skiprows=11, header=0)
        name = file.name.split("-")[-1].split(".")[0].strip()
        preds = get_prediction(df, features)
        predictions[name] = preds

    return predictions

def get_preds_inter_agreements(predictions: dict, features: dict):
    rows = []
    names = []
    for name1, name2 in combinations(predictions.keys(), 2):
        names.append(f"{name1} vs {name2}")
        rows.append(get_kappa(predictions[name1], predictions[name2], features))
    return pd.DataFrame(rows, index=names, columns=features.keys())

def get_preds_f1(predictions: dict, true:dict, split="test_human"):
    rows = []
    names = []
    tasks = []
    for name in predictions.keys():
        names.append(name)
        f1_macro, f1_micro = get_f1(predictions[name], true, split=split), get_f1(predictions[name], true, split=split, average="micro")
        # sort by features
        f1_macro = {k: f1_macro[k] for k in predictions[name].keys()}
        f1_micro = {k: f1_micro[k] for k in predictions[name].keys()}
        tasks = list(f1_macro.keys())
        combined_f1 = list(f1_macro.values())
        for i,f in enumerate(f1_micro.values()):
            combined_f1.insert(i*2+1, f)
        rows.append(combined_f1)
    return pd.DataFrame(rows, index=names, columns=pd.MultiIndex.from_product([tasks, [f"{split}/f1_macro", f"{split}/f1_micro"]]))


@cache
def get_labels(task, split):
    dst = load_dataset("hynky/czech_news_dataset", split=split)
    unq = np.unique(dst[task]).tolist()
    if 0 in unq:
        unq.remove(0)
    return [u - 1 for u in unq]

