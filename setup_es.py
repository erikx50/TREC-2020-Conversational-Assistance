from elasticsearch import Elasticsearch
from krovetzstemmer import Stemmer as Kstemmer
import re
import os
import csv
import nltk
from trec_car import read_data


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


def preprocess(doc):
    # Removes all non alpha-numerical characters, performs stopword removal and K-stemming
    stemmer = Kstemmer()
    return [stemmer.stem(term) for term in re.sub('[^0-9a-zA-Z]+', " ", doc).lower().split() if term not in STOPWORDS]


def load_ms_macro_to_es(filepath, es):
    with open(filepath, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            body = {"id": "MARCO_"+line[0], "data": ' '.join(preprocess(line[1]))}
            es.index(index=INDEX_NAME, doc_type="_doc", id="MARCO_"+line[0], body=body)
    return


def load_trec_car_to_es(filepath, es):
    for par in read_data.iter_paragraphs(open(filepath, "rb")):
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
    # load_ms_macro_to_es(os.path.normpath('data/MS Macro collection.tsv'), es)

    # Import and pre-process wiki collection
    # load_trec_car_to_es(os.path.normpath("data/dedup.articles-paragraphs.cbor"), es)


if __name__ == "__main__":
    print("Remove comment mark from line under to build database")
    # main()

