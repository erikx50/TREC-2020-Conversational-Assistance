import os
import ir_measures
from ir_measures import nDCG, MAP, MRR, R


def evaluate(result_path, qrels_path):
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(result_path)
    print(ir_measures.calc_aggregate([R(rel=2)@1000, MAP(rel=2)@1000, MRR(rel=2)@1000, nDCG@1000, nDCG@3], qrels, run))


if __name__ == "__main__":
    print("Manual Evaluation Topics")
    print("BM25 manual evaluation topics")
    evaluate(os.path.normpath('results/bm25_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker manual evaluation topics")
    evaluate(os.path.normpath('results/BERT_manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("\n")
    print('Automatic Evaluation Topics')
    print("BM25 automatic evaluation topics")
    evaluate(os.path.normpath('results/bm25_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))

    print("BERT re-ranker automatic evaluation topics")
    evaluate(os.path.normpath('results/BERT_automatic_results.txt'), os.path.normpath('2020/2020qrels.txt'))
