import json
import os
from setup_es import preprocess
from elasticsearch import Elasticsearch

INDEX_NAME = "collection"


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        f = json.load(file)
        return f


def baseline_retrieval(es, index_name, query, k):
    query_list = []
    for term in query:
        query_list.append({"match": {"data": term}},)
    q = {
        "query": {
            "bool": {
                 "should": query_list
                    }
                }
        }

    res = es.search(index=index_name, body=q, size=k)       # Receives top k candidates in descending order
    return_list = []
    for hits in res['hits']['hits']:
        return_list.append(hits['_id'])
    return return_list


def run_baseline_retrieval(es, eval_topics, index_name, k):
    result_dict = {}
    for topic in eval_topics:
        for turn in topic['turn']:
            query = preprocess(turn['manual_rewritten_utterance'])
            result_dict[str(topic['number']) + '_' + str(turn['number'])] = baseline_retrieval(es, index_name, query, k)
    return result_dict


if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch()

    # Import train data
    manual_train_data = load_json(os.path.normpath('2020/2020_manual_evaluation_topics_v1.0.json'))

    # Run baseline retrieval
    result = run_baseline_retrieval(es, manual_train_data, INDEX_NAME, 500)
    print(result)

    # TODO: Remove code under
    #print(es.get(index="collection", id='MARCO_335000')['_source'])
