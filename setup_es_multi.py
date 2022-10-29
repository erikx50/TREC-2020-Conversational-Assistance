from xmlrpc.server import list_public_methods
from elasticsearch import Elasticsearch
from krovetzstemmer import Stemmer as Kstemmer
import re
import os
import csv
import nltk
from trec_car import read_data
import ray


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
    i = 0
    data: list = []
    blockers: list = []

    with open(filepath, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            i += 1
            data.append(line)
            if len(data) >= 500000:
                last = toIndxMacro.remote(data)
                blockers.append(last)
                data = []
            if i % 500000 == 0:
                print(f"Diveded {i:,} macro Documents")
        if len(data) > 0: # leftover data 
            r = toIndxMacro.remote(data)
            blockers.append(r)

    print(f"Diveded up a total of {i:,} car documents")
    print("WAITING")
    for b in blockers:
        ray.get(b)
    print("ALL MACRO HAS RETURNED")
    return

@ray.remote
def toIndxMacro(data):
    i = 0
    es = Elasticsearch()
    for line in data:
        i += 1
        body = {"id": "MARCO_"+line[0], "data": ' '.join(preprocess(line[1]))}
        es.index(index=INDEX_NAME, doc_type="_doc", id="MARCO_"+line[0], body=body)
        if i % 100000 == 0:
            print(f"@{i:,} Documents")
    print("ONE PROCESS DONE")
        

@ray.remote
def toIndxCar(data):
    i = 0
    es = Elasticsearch()
    for par in data:
        i += 1
        totText = ""
        for p in par.bodies:
            totText += p.get_text() # text, can be strung togherther to get full body
        body = {"id": "CAR_"+par.para_id, "data": ' '.join(preprocess(totText))}
        es.index(index=INDEX_NAME, doc_type="_doc", id="CAR_"+par.para_id, body=body)
        if i % 100000 == 0:
            print(f"@{i:,} Documents")
    print("ONE PROCESS DONE")

def load_trec_car_to_es(filepath, es):
    i = 0
    count = 0
    data: list = []
    blockers: list = []

    for par in read_data.iter_paragraphs(open(filepath, "rb")):
        i += 1
        count += 1
        data.append(par)
        if len(data) >= 500000:
            last = toIndxCar.remote(data)
            blockers.append(last)
            data = []
        if i % 500000 == 0:
            print(f"Diveded {i:,} car Documents")
        
        # There are too many documents, need to split it
        # into numcpus chunks at a time.
        if count >= os.cpu_count():
            for b in blockers: # Block til done
                ray.get(b)
            blockers = []
            count = 0
    if len(data) > 0: # leftover data 
        r = toIndxCar.remote(data)
        blockers.append(r)
        
    print(f"Diveded up a total of {i:,} car documents")
    print("WAITING")

    print("ALL CAR HAS RETURNED")
    return


def main():
    ray.init(num_gpus=0)
    es = Elasticsearch()
    reset_index(es)

    # Import, pre-process and add marco collection to es database
    #load_ms_macro_to_es(os.path.normpath('data/MS Macro collection.tsv'), es)

    # Import and pre-process wiki collection
    load_trec_car_to_es(os.path.normpath("data/dedup.articles-paragraphs.cbor"), es)


if __name__ == "__main__":
    #print("Remove comment mark from line under to build database")
    main()

