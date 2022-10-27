import json
import os
from setup_es import preprocess
from elasticsearch import Elasticsearch

INDEX_NAME = "collection"


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        f = json.load(file)
        return f


def write_results(path, result):
    with open(path, "w") as file:
        for id in result:
            counter = 1
            for doc in result[id]:
                file.write(str(id) + ' ' + 'Q0' + ' ' + str(doc) + ' ' + str(counter) + ' ' + str(result[id][doc]) + '\n')
                counter += 1


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
    return_dict = {}
    for hits in res['hits']['hits']:
        return_dict[hits['_id']] = hits['_score']
    return return_dict


def run_baseline_retrieval(es, eval_topics, index_name, k):
    result_dict = {}
    for topic in eval_topics:
        for turn in topic['turn']:
            query = preprocess(turn['manual_rewritten_utterance'])
            result_dict[str(topic['number']) + '_' + str(turn['number'])] = baseline_retrieval(es, index_name, query, k)
    return result_dict


if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch(timeout=120)
    # Import train data
    manual_train_data = load_json(os.path.normpath('2020/2020_manual_evaluation_topics_v1.0.json'))

    # Run baseline retrieval
    result = run_baseline_retrieval(es, manual_train_data, INDEX_NAME, 500)
    print(result['81_1'])

    # Write result to file
    write_results(os.path.normpath('results/manual_results.txt'), result)

    # TODO: Remove code under
    # print(es.get(index="collection", id='MARCO_570000')['_source'])
