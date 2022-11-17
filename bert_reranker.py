from sentence_transformers import CrossEncoder, SentenceTransformer, util
from bm25baseline import baseline_retrieval, load_json
from setup_es import preprocess
from elasticsearch import Elasticsearch
import os
from tqdm import tqdm
from typing import Dict, List, Union


INDEX_NAME = "prosjektdbfull"


def write_bert_results(file_path: str, result: Dict[str, List[str]], utterance_type: str):
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
                file.write(str(id) + ' ' + 'Q0' + ' ' + str(doc[0]) + ' ' + str(counter) + ' ' + str(doc[1]) + ' ' + 'BERT_' + utterance_type + '\n')
                counter += 1


def bert_re_ranker(es: Elasticsearch, utterance_type: str, json_path: str, index_name: str, k: int) -> Union[Dict[str, List[str]], None]:
    """
    Performs bm25 baseline retrieval before re-ranking the result using BERT.
    Args:
        es: elasticsearch client
        utterance_type: Manual or Automatic depending on what utterances we want to use
        json_path: Path to the json evaluation_topics file
        index_name: The elastic search index where the retrieval is performed.
        k: Number of documents to return.

    Returns:
        A dictionary containing the topic-number_turn-number as key and a list containing document score pairs as value.
    """

    data = load_json(os.path.normpath(json_path))
    model = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2', max_length=512)

    if utterance_type == 'Manual':
        utterance = 'manual_rewritten_utterance'
    elif utterance_type == 'Automatic':
        utterance = 'automatic_rewritten_utterance'
    else:
        return

    result_dict = {}
    for topic in tqdm(data):
        for turn in tqdm(topic['turn']):
            query = preprocess(turn[utterance])
            docs = [doc for doc in baseline_retrieval(es, index_name, query, k*5)]

            query_passage_pairs = [[" ".join(query), es.get(index=INDEX_NAME, id=doc_id)["_source"]["data"]] for doc_id in docs]
            scores = model.predict(query_passage_pairs)

            turn_results = {}
            for index in range(len(scores)):
                turn_results[docs[index]] = scores[index]
            turn_results = sorted(turn_results.items(), key=lambda x: x[1], reverse=True)
            result_dict[str(topic['number']) + '_' + str(turn['number'])] = turn_results[0:500]
    return result_dict


def main(es: Elasticsearch, utterance_type: str, source_path: str, write_path: str) -> None:
    """
    Performs bert re-ranking and writes to file.
    Args:
        es: elasticsearch client
        utterance_type: Manual or Automatic depending on what utterances we want to use
        source_path: The path of the evaluation_topics json file.
        write_path: Where to write the result txt file
    """
    # Get BERT results
    result = bert_re_ranker(es, utterance_type, source_path, INDEX_NAME, 500)

    # Write to file
    write_bert_results(write_path, result, utterance_type)


if __name__ == "__main__":
    # Initiate elastic search client
    es = Elasticsearch(timeout=120)

    # Run BERT re-ranking on manual evaluation topics
    #main(es, 'Manual', '2020/2020_manual_evaluation_topics_v1.0.json', 'results/BERT_manual_results.txt')

    # Run BERT re-ranking on automatic evaluation topics
    main(es, 'Automatic', '2020/2020_automatic_evaluation_topics_v1.0.json', 'results/BERT_automatic_results.txt')



