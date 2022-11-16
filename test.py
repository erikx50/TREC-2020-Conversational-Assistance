from elasticsearch import Elasticsearch

INDEX_NAME = "prosjektdbfull"

if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch(timeout=120)
    r = es.count(q='*', index=INDEX_NAME, request_timeout=60)
    c = r['count']
    print('Num docs: ' + str(c))
