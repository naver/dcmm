
'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import argparse
import json

import pytrec_eval
import scipy.stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path")
    parser.add_argument("--run1_file_path")
    parser.add_argument("--run2_file_path")
    parser.add_argument("--measure")  # should match pytrec_eval measures !
    args = parser.parse_args()

    with open(args.qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(args.run1_file_path) as reader:
        run1 = json.load(reader)
    with open(args.run2_file_path) as reader:
        run2 = json.load(reader)
    assert len(run2) == len(run1)

    if args.measure.startswith("P_"):
        arg = "P"
    elif args.measure.startswith("ndcg_"):
        arg = "ndcg_cut"
    elif args.measure.startswith("map_"):
        arg = "map_cut"
    else:
        arg = args.measure
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {arg})
    results_1 = evaluator.evaluate(run1)
    print(results_1)
    results_2 = evaluator.evaluate(run2)
    query_ids = list(results_1.keys())
    scores_1 = [
        results_1[query_id][args.measure] for query_id in query_ids]
    print(args.measure, "for run 1", sum(scores_1) / len(scores_1))
    scores_2 = [
        results_2[query_id][args.measure] for query_id in query_ids]
    print(args.measure, "for run 2", sum(scores_2) / len(scores_2))
    print("~~ TEST ~~")
    print(scipy.stats.ttest_rel(scores_1, scores_2))
