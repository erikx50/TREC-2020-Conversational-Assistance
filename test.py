from elasticsearch import Elasticsearch

INDEX_NAME = "collection"

if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch(timeout=120)
    print(es.get(index="collection", id='MARCO_3700000')['_source'])
