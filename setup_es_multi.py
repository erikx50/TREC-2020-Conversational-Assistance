from elasticsearch import Elasticsearch
from krovetzstemmer import Stemmer as Kstemmer
from setup_es import reset_index, preprocess
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


CARBATCHSIZE = 100000


def load_ms_macro_to_es(filepath: str, es: Elasticsearch) -> None:
    """
    Loads the macro collection at filepath into the elastisearch connection on the es client.
    It will divide the workload over the amount of CPUs available in chunks of 500k documents.
    Args:
        filepath: path to tsv file containing macro collection
        es: elasticsearch client
    """
    i = 0
    count = 0
    data: list = []
    blockers: list = []

    with open(filepath, 'r', encoding='utf-8') as file:
        tsv_file = csv.reader(file, delimiter="\t")
        for line in tsv_file:
            i += 1
            data.append(line)
            if len(data) >= 500000:
                count += 1
                last = toIndxMacro.remote(data)
                blockers.append(last)
                data = []
            if i % 500000 == 0:
                print(f"Diveded {i:,} macro Documents")
            if count >= os.cpu_count(): # limt paralells to cpu cores
                # May cause an error if above.
                print("Waiting for one macro batch")
                for b in blockers: # Block til done
                    ray.get(b)
                blockers = []
                count = 0
                print("One macro batch, starting a new")
        if len(data) > 0: # leftover data 
            print(f"Waiting for last batch. Total to be {i:,} documents")
            r = toIndxMacro.remote(data)
            blockers.append(r)

    for b in blockers:
        ray.get(b) # If any leftover has to be fully processed
    print(f"Diveded up a total of {i:,} macro documents")
    print("ALL MACRO HAS RETURNED")
    return

@ray.remote
def toIndxMacro(data: list) -> None:
    """
    Preprocesses the documents and index them to local elastisearch instance.
    Args:
        data: List of raw lines from tsv file of macro collection
    """
    i = 0
    es = Elasticsearch()
    for line in data:
        i += 1
        body = {"id": "MARCO_"+line[0], "data": ' '.join(preprocess(line[1]))}
        es.index(index=INDEX_NAME, doc_type="_doc", id="MARCO_"+line[0], body=body, request_timeout=60)
        if i % 100000 == 0:
            print(f"@{i:,} Documents")
    print("ONE PROCESS DONE")
        

@ray.remote
def toIndxCar(data: list):
    """
    Preprocesses the documents and index them to local elastisearch instance.
    Args:
        data: List of line entries from trec car collection (wiki)
    """
    print("New car process started")
    i = 0
    es = Elasticsearch()
    for par in data:
        i += 1
        totText = ""
        for p in par.bodies:
            totText += p.get_text() # text, can be strung togherther to get full body
        body = {"id": "CAR_"+par.para_id, "data": ' '.join(preprocess(totText))}
        es.index(index=INDEX_NAME, doc_type="_doc", id="CAR_"+par.para_id, body=body, request_timeout=120)
        if i % 100000 == 0:
            print(f"@{i:,} Documents")
    es.close()
    print("ONE PROCESS DONE")


def load_trec_car_to_es(filepath: str, es:Elasticsearch) -> None:
    """
    Loads the trec car collection at filepath into the elastisearch connection on the es client.
    It will divide the workload over the amount of CPUs available in chunks of global
    variable CARBATCHSIZE due to memory limits.
    Args:
        filepath: path to cbor file containing trec car (wiki) collection
        es: elasticsearch client
    """
    i = 0
    count = 0
    data: list = []
    blockers: list = []

    for par in read_data.iter_paragraphs(open(filepath, "rb")):
        i += 1
        data.append(par)
        if len(data) >= CARBATCHSIZE:
            count += 1
            last = toIndxCar.remote(data)
            blockers.append(last)
            data = []
        if i % CARBATCHSIZE == 0:
            print(f"Diveded {i:,} car Documents")
        
        # There are too many documents, need to split it
        # into numcpus chunks at a time.
        if count >= os.cpu_count():
            print("Waiting for one car batch")
            for b in blockers: # Block til done
                ray.get(b)
            blockers = []
            count = 0
            print("One car batch, starting a new")
    if len(data) > 0: # leftover data 
        print(f"Waiting for last batch. Total to be {i:,} documents")
        r = toIndxCar.remote(data)
        blockers.append(r)
    
    for b in blockers:
        ray.get(b) # If any leftover has to be fully processed
    print(f"Diveded up a total of {i:,} car documents")
    print("ALL CAR HAS RETURNED")
    return


def main() -> None:
    """
    Will pre-process and add Macro collection and wiki collection to
    local running Elastisearch instance at index of global variable: INDEX_NAME
    """
    ray.init(num_gpus=0, num_cpus=os.cpu_count())
    es = Elasticsearch()
    reset_index(es)

    # Import, pre-process and add marco collection to es database
    load_ms_macro_to_es(os.path.normpath('data/MS Macro collection.tsv'), es)

    # Import and pre-process wiki collection
    load_trec_car_to_es(os.path.normpath("data/dedup.articles-paragraphs.cbor"), es)


if __name__ == "__main__":
    #print("Remove comment mark from line under to build database")
    main()

