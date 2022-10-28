import os
import ir_measures
from ir_measures import nDCG, MAP, MRR, R


def evaluate(result_path, qrels_path):
    qrels = ir_measures.read_trec_qrels(qrels_path)
    run = ir_measures.read_trec_run(result_path)
    print(ir_measures.calc_aggregate([R@1000, MAP@1000, MRR@1000, nDCG@1000, nDCG@3], qrels, run))


if __name__ == "__main__":
    evaluate(os.path.normpath('results/manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

