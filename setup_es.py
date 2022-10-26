from elasticsearch import Elasticsearch
from krovetzstemmer import Stemmer as Kstemmer
import re
import os
import csv
import nltk


nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


INDEX_NAME = "collection"

INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "id": {
                "type": "text",
                "term_vector": "no",
                "analyzer": "english",
            },
            "data": {
                "type": "text",
                "term_vector": "yes",
                "analyzer": "english",
            },
        }
    }
}


def reset_index(es: Elasticsearch):
    """Reset Index"""
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)


def bulk_index(es, collection):
    """Iterate over artists, and index those that are of the
    right type.

    Args:
        es: Elasticsearch instance.
        artists: Dictionary with artist names and their properties.
    """
    for col in collection:
        body = {"id": col, "data": ' '.join(collection[col])}
        es.index(index=INDEX_NAME, doc_type="_doc", id=col, body=body)


def preprocess(doc):
    # Removes all non alpha-numerical characters, performs stopword removal and K-stemming
    stemmer = Kstemmer()
    return [stemmer.stem(term) for term in re.sub('[^0-9a-zA-Z]+', " ", doc).lower().split() if term not in STOPWORDS]


def load_ms_macro(filepath):
    return_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            return_dict["MARCO_"+line[0]] = line[1]
    return return_dict


if __name__ == "__main__":
    """Index artists"""
    es = Elasticsearch()
    es.info()

    # Import and pre-process marco collection
    macro_collection = load_ms_macro(os.path.normpath('data/MS Macro collection split.tsv'))
    for index in macro_collection:
        macro_collection[index] = preprocess(macro_collection[index])

    reset_index(es)
    bulk_index(es, macro_collection)
    print(es.get(index=INDEX_NAME, id='MARCO_5499807'))
