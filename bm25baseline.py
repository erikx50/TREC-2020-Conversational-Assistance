import json
import os
from setup_es import preprocess
from elasticsearch import Elasticsearch
from typing import Dict, List, Union

INDEX_NAME = "prosjektdbfull"


def load_json(file_path: str) -> json:
    """
    Reads data from a json file.
    Args:
        file_path: The path to the json file.
    Returns:
        The content of the json file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        f = json.load(file)
        return f


def write_results(file_path: str, result: Dict[str, Dict[str, float]], utterance_type: str) -> None:
    """
    Writes a txt file in TREC format.
    Args:
        file_path: The path of where to write the file..
        result: Dictionary containing the topic+turn as key and dictionary of top k documents and their score as value
        utterance_type: Manual or Automatic depending on what utterances we want to use
    """
    with open(file_path, "w") as file:
        for id in result:
            counter = 1
            for doc in result[id]:
                file.write(str(id) + ' ' + 'Q0' + ' ' + str(doc) + ' ' + str(counter) + ' ' + str(result[id][doc]) + ' ' + utterance_type + '\n')
                counter += 1


def baseline_retrieval(es, index_name: str, query: List[str], k: int) -> Dict[str, float]:
    """
    Performs BM25 baseline retrieval on index.
    Args:
        es: elasticsearch client
        index_name: The elastic search index where the retrieval is performed.
        query: A list of split up query strings.
        k: Number of documents to return.

    Returns:
        A dictionary where the ID of the document is the key and the score of the document is the value.
        This is done i decending order where the highest score is first in the dictionary.
    """

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


def run_baseline_retrieval(es, utterance_type: str, json_file: json, index_name: str, k: int) -> Union[Dict[str, Dict[str, float]], None]:
    if utterance_type == 'Manual':
        utterance = 'manual_rewritten_utterance'
    elif utterance_type == 'Automatic':
        utterance = 'automatic_rewritten_utterance'
    else:
        return

    result_dict = {}
    for topic in json_file:
        for turn in topic['turn']:
            query = preprocess(turn[utterance])
            result_dict[str(topic['number']) + '_' + str(turn['number'])] = baseline_retrieval(es, index_name, query, k)
    return result_dict


def main(es, utterance_type: str, source_path: str, write_path: str) -> None:
    # Load JSON file
    data = load_json(os.path.normpath(source_path))

    # Get baseline results
    results = run_baseline_retrieval(es, utterance_type, data, INDEX_NAME, 500)

    # Write to file
    write_results(os.path.normpath(write_path), results, utterance_type)


if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch(timeout=120)

    # Run baseline retrieval on manual evaluation topics
    main(es, 'Manual', '2020/2020_manual_evaluation_topics_v1.0.json', 'results/bm25_manual_results.txt')

    # Run baseline retrieval on automatic evaluation topics
    main(es, 'Automatic', '2020/2020_automatic_evaluation_topics_v1.0.json', 'results/bm25_automatic_results.txt')

