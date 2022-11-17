import os
import torch
import numpy as np
from tqdm import tqdm
from setup_es import preprocess
from elasticsearch import Elasticsearch
from bm25baseline import baseline_retrieval, load_json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from scipy.special import softmax
from typing import Dict, List, Union

#Model: https://huggingface.co/castorini/monot5-base-msmarco

INDEX_NAME = "prosjektdbfull"
model = T5ForConditionalGeneration.from_pretrained("castorini/monot5-base-msmarco")
tokenizer = T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco")



def calcScore(es: Elasticsearch, docID: str, queryString: str) -> float:
    """
    Scores the query on given document using T5 with model and tokenizer from 
    https://huggingface.co/castorini/monot5-base-msmarco

    Args:
        es: elasticsearch client
        docID: Id of document to be scored against
        queryString: raw query string to score document against
    Returns:
        score: float number representing the documents score
    """
    d = es.get(index=INDEX_NAME, id=docID)
    doc = tokenizer(d['_source']['data'], return_tensors='pt')
    query = tokenizer(queryString, return_tensors="pt")

    score = 0
    with torch.no_grad():
        o = model(input_ids=doc.input_ids, labels=query.input_ids)
        logs = o[1][0]
        smTransform = softmax(logs.numpy(), axis=1)
        for i, b in enumerate(query.input_ids[0]):
            score += np.log10(smTransform[i][b])
    
    return score


def reranker(es: Elasticsearch, utterance_type: str, json_path: str, index_name: str, k: int) -> Union[Dict[str, List[str]], None]:
    """
        Args:
        es: elasticsearch client
        utterance_type: Manual or Automatic depending on what utterances we want to use
        json_path: Path to the json evaluation_topics file
        index_name: The elastic search index where the retrieval is performed.
        k: Number of returned elastisearch documents to score each query on.

    Returns:
        A dictionary containing the topic-number_turn-number as key and a list containing document score pairs as value.
    """
    data = load_json(os.path.normpath(json_path))
    
    utterance: str
    if utterance_type == 'Manual':
        utterance = 'manual_rewritten_utterance'
    elif utterance_type == 'Automatic':
        utterance = 'automatic_rewritten_utterance'
    else:
        return

    result_dict = {}
    for topic in tqdm(data):
        for turn in topic['turn']:
            queryst = turn[utterance]
            query = preprocess(queryst)
            docs = [doc for doc in baseline_retrieval(es, index_name, query, k)]

            scores = []
            for d in docs:
                scores.append(calcScore(es, d, queryst))

            turn_results = {}
            for index in range(len(scores)):
                turn_results[docs[index]] = scores[index]
            turn_results = sorted(turn_results.items(), key=lambda x: x[1], reverse=True)
            result_dict[str(topic['number']) + '_' + str(turn['number'])] = turn_results
    return result_dict



def saveToFile(result_dict: Dict[str, Dict[str, float]], filepath: str) -> None:
    """
    Writes a txt file in TREC format.
    Args:
        file_path: The path of where to write the file..
        result: Dictionary containing the topic+turn as key and dictionary of top k documents and their score as value
        utterance_type: Manual or Automatic depending on what utterances we want to use
    """
    with open(filepath, "w") as file:
        for id in result_dict:
            counter = 1
            for doc in result_dict[id]:
                file.write(str(id) + ' ' + 'Q0' + ' ' + str(doc[0]) + ' ' + str(counter) + ' ' + str(doc[1]) + ' ' + 'BERT_' + "manual_rewritten_utterance" + '\n')
                counter += 1



def main(utterance_type: str, source_path: str, write_path: str) -> None:
    """
    Performs T5 re-ranking and writes the result to a corresponding file.
    Args:
        es: elasticsearch client
        utterance_type: Manual or Automatic depending on what utterances we want to use
        source_path: The path of the evaluation_topics json file.
        write_path: Where to write the result txt file
    """
    es = Elasticsearch(timeout=120)

    dcCount = input("How many documents per query: ")
    dcCount = int(dcCount)
    res = reranker(es, utterance_type, source_path, INDEX_NAME, dcCount)
    saveToFile(res, write_path)


if __name__ == "__main__":
    mode = input("Do you want to run manual (m) or automatic (a): ")

    if mode.lower() == "m" or mode.lower() == "manual":
        main('Manual', '2020/2020_manual_evaluation_topics_v1.0.json', 'results/T5_manual_results.txt')
    elif mode.lower() == "a" or mode.lower() == "automatic":
        main('Automatic', '2020/2020_automatic_evaluation_topics_v1.0.json', 'results/T5_automatic_results.txt')