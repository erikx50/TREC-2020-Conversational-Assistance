from trectools import TrecRun, TrecQrel, TrecEval
import os


def evaluate(result_path, qrels_path):
    run = TrecRun(result_path)
    qrels = TrecQrel(qrels_path)
    eval = TrecEval(run, qrels)
    print('Evaluation of run')
    print('Recall: ', )
    print('MAP: ' + str(eval.get_map(depth=1000)))
    print('MRR: ' + str(eval.get_reciprocal_rank(depth=1000)))
    print('NDCG: ' + str(eval.get_ndcg(depth=1000)))
    print('NDCG@3: ' + str(eval.get_ndcg(depth=3)))



if __name__ == "__main__":
    evaluate(os.path.normpath('results/manual_results.txt'), os.path.normpath('2020/2020qrels.txt'))

