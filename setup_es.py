from elasticsearch import Elasticsearch
from krovetzstemmer import Stemmer as Kstemmer
import re
import os
import csv
import nltk
from trec_car import read_data
from typing import List


nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))


INDEX_NAME = "prosjektdbfull"

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


def reset_index(es: Elasticsearch) -> None:
    """
    Deletes and recreates index from elasticsearch
    Args:
        es: elasticsearch client
    """
    if es.indices.exists(INDEX_NAME):
        es.indices.delete(index=INDEX_NAME)

    es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)


def preprocess(text: str) -> List[str]:
    """
    Removes all non alpha-numerical characters, performs stopword removal and K-stemming on a string.
    Args:
        text: String of text to be preprocessed
    Returns:
        A list of the words in the pre-processed string.
    """
    # Removes all non alpha-numerical characters, performs stopword removal and K-stemming
    stemmer = Kstemmer()
    return [stemmer.stem(term) for term in re.sub('[^0-9a-zA-Z]+', " ", text).lower().split() if term not in STOPWORDS]


def load_ms_macro_to_es(file_path: str, es: Elasticsearch) -> None:
    """
    Reads the ms macro TSV file and adds each document to Elasticsearch
    Args:
        file_path: Filepath to MS Macro collection.tsv
        es: elasticsearch client
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            body = {"id": "MARCO_"+line[0], "data": ' '.join(preprocess(line[1]))}
            es.index(index=INDEX_NAME, doc_type="_doc", id="MARCO_"+line[0], body=body)
    return


def load_trec_car_to_es(file_path: str, es: Elasticsearch) -> None:
    """
    Reads the CAR cbor file and adds each document to Elasticsearch
    Args:
        file_path: Filepath to dedup.articles-paragraphs.cbor
        es: elasticsearch client
    """
    for par in read_data.iter_paragraphs(open(file_path, "rb")):
        totText = ""
        for p in par.bodies:
            totText += p.get_text() # text, can be strung togherther to get full body
        body = {"id": "CAR_"+par.para_id, "data": ' '.join(preprocess(totText))}
        es.index(index=INDEX_NAME, doc_type="_doc", id="CAR_"+par.para_id, body=body)
    return


def main():
    es = Elasticsearch()
    # reset_index(es)

    # Import, pre-process and add marco collection to es database
    load_ms_macro_to_es(os.path.normpath('data/MS Macro collection.tsv'), es)

    # Import and pre-process wiki collection
    load_trec_car_to_es(os.path.normpath("data/dedup.articles-paragraphs.cbor"), es)


if __name__ == "__main__":
    print("Remove comment mark from line under to build database")
    # main()

