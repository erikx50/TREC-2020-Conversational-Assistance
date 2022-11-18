import os
import ir_measures
from ir_measures import nDCG, MAP, MRR, R


def evaluate(result_path: str, qrels_path: str) -> None:
    """
    Evaluates the TREC results based on the qrels file. Recall, Mean Average Precision, Mean Reciprocal Rank and
    Normalized Discounted Cumulative Gain @3 and @1000.
    Args:
        result_path: The path to the result file in TREC format to be evaluated.
        qrels_path: The path to the qrels txt file in TREC format.
    """
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(result_path)
    print(ir_measures.calc_aggregate([R(rel=2)@1000, MAP(rel=2)@1000, MRR(rel=2)@1000, nDCG@1000, nDCG@3], qrels, run))


def evaluate_by_turn_depth(result_path: str, qrels_path: str) -> None:
    """
    Print the Normalized Discounted Cumulative Gain @3 from the TREC results based on the qrels file for every turn
    up to turn 10.
    Args:
        result_path: The path to the result file in TREC format to be evaluated.
        qrels_path: The path to the qrels txt file in TREC format.
    """
    # Load run and qrels
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(result_path)

    # Add the qrels and runs of every turn up to turn 10 into dictionaries
    qrels_depth = {}
    for qrel in qrels:
        turn = qrel[0].split('_')[1]
        if int(turn) <= 10:
            if turn not in qrels_depth:
                qrels_depth[turn] = [qrel]
            else:
                qrels_depth[turn].append(qrel)

    run_depth = {}
    for scored_doc in run:
        turn = scored_doc[0].split('_')[1]
        if int(turn) <= 10:
            if turn not in run_depth:
                run_depth[turn] = [scored_doc]
            else:
                run_depth[turn].append(scored_doc)

    # Calculate NDCG@3 for every turn
    for turn in qrels_depth:
        print('Turn: ' + turn + ' - ' + str(ir_measures.calc_aggregate([nDCG@3], qrels_depth[turn], run_depth[turn])))


def evaluate_by_topic(result_path: str, qrels_path: str) -> None:
    """
    Print the Normalized Discounted Cumulative Gain @3 from the TREC results based on the qrels file for every topic.
    Args:
        result_path: The path to the result file in TREC format to be evaluated.
        qrels_path: The path to the qrels txt file in TREC format.
    """
    # Load run and qrels
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(result_path)

    # Add the qrels and runs of every topic into dictionaries
    qrels_topic = {}
    for qrel in qrels:
        topic = qrel[0].split('_')[0]
        if topic not in qrels_topic:
            qrels_topic[topic] = [qrel]
        else:
            qrels_topic[topic].append(qrel)

    run_topic = {}
    for scored_doc in run:
        topic = scored_doc[0].split('_')[0]
        if topic not in run_topic:
            run_topic[topic] = [scored_doc]
        else:
            run_topic[topic].append(scored_doc)

    # Calculate NDCG@3 for every topic
    for topic in qrels_topic:
        print('Topic: ' + topic + ' - ' + str(ir_measures.calc_aggregate([nDCG@3], qrels_topic[topic], run_topic[topic])))


if __name__ == "__main__":
    print("Manual Evaluation Topics")
    print("BM25 manual evaluation topics")
    evaluate(os.path.normpath('results/bm25_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker manual evaluation topics")
    evaluate(os.path.normpath('results/BERT_reranker_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker manual evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/BERT_reranker_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT ranker manual evaluation topics")
    evaluate(os.path.normpath('results/BERT_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT ranker manual evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/BERT_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("T5 ranker manual evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/T5_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("\n")
    print('Automatic Evaluation Topics')
    print("BM25 automatic evaluation topics")
    evaluate(os.path.normpath('results/bm25_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker automatic evaluation topics")
    evaluate(os.path.normpath('results/BERT_reranker_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker automatic evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/BERT_reranker_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT ranker automatic evaluation topics")
    evaluate(os.path.normpath('results/BERT_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT ranker automatic evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/BERT_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("T5 ranker automatic evaluation topics by turn depth")
    evaluate_by_turn_depth(os.path.normpath('results/T5_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

