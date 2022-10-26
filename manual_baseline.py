import re
import json
import numpy as np
import os
import csv
import nltk

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        f = json.load(file)
        return f


def load_ms_macro(filepath):
    return_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            return_dict["MARCO_"+line[0]] = line[1]
            # TODO: Remove break used for testing purposes
            if line[0] == "1000":
                break
    return return_dict


def preprocess(doc):
    # Removes all non alpha-numerical characters, performs stopword removal and KStemming
    return [
        term
        for term in re.sub(r"[^\w]|_", " ", doc).lower().split()
        if term not in STOPWORDS
    ]


if __name__ == "__main__":
    # manual_test_data = load_json(os.path.normpath('2020/2020_manual_evaluation_topics_v1.0.json'))

    # Import and pre-process marco collection
    macro_collection = load_ms_macro(os.path.normpath('data/MS Macro collection.tsv'))
    for index in macro_collection:
        macro_collection[index] = preprocess(macro_collection[index])
    print(macro_collection)

    # TODO: Import and pre-process wiki collection
