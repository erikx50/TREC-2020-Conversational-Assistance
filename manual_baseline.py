import json
import os
import nltk
import cbor2 as cbor
from elasticsearch import Elasticsearch

nltk.download("stopwords")
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        f = json.load(file)
        return f


def load_cbor(filepath):
    with open(filepath, 'r', encoding='ISO-8859-1') as file:
        f = cbor.load(file)
        return f


if __name__ == "__main__":
    # Import test data
    manual_test_data = load_json(os.path.normpath('2020/2020_manual_evaluation_topics_v1.0.json'))

    # Import and pre-process wiki collection
    # wiki_collection = load_cbor(os.path.normpath('data/dedup.articles-paragraphs.cbor'))

    # Score the whole collection
    es = Elasticsearch()
    print(es.get(index="collection", id='MARCO_5499807'))
