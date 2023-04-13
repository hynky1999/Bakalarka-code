import pandas as pd
from sklearn.metrics import f1_score
import re
end_re = re.compile(r'(Pond|Úte|Stř|Čtv|Pát|Sob|Ned)')
translate_day = {
    "Pondělí": "Pond",
    "Pond": "Pond",
    "Úterý": "Úte",
    "Úte": "Úte",
    "Středa": "Stř",
    "Stř": "Stř",
    "Čtvrtek": "Čtv",
    "Čtv": "Čtv",
    "Pátek": "Pát",
    "Pát": "Pát",
    "Sob": "Sob",
    "Ned": "Ned",
    "Sobota": "Sob",
    "Neděle": "Ned",
}
cols = ["server", "category", "authors_cum_gender", "day_of_week"]


def split_to_task(text):
    end = end_re.search(text)
    text =text.replace("Smíšené pohlaví", "Smíšené")
    text = text[:end.end()]
    server, *category, gender, day = text.strip().split(' ')
    category = ' '.join(category)
    day = translate_day[day]
    return server, category, gender, day



def get_predictions(path):
    results_df = pd.read_json(path, lines=True, orient="records")
    predicted = pd.DataFrame(results_df["results"].apply(split_to_task).to_list(), columns=cols)
    return predicted

def get_expected(path):
    results_df = pd.read_json(path, lines=True, orient="records")
    expected = pd.DataFrame(results_df["answer"].apply(split_to_task).to_list(), columns=cols)

    return expected





