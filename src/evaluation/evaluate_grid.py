'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from tabulate import tabulate

from src.utils.utils import evaluate

"""
evaluation script for a grid of experiments 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path")
    parser.add_argument("--exps_dir")

    args = parser.parse_args()
    best = ("0", {"ndcg": 0})
    metrics = ["map", "ndcg", "ndcg_cut_5", "ndcg_cut_20", "P_5", "P_20"]
    res = defaultdict(list)
    val_perf = {}
    with open(args.qrel_file_path) as handler:
        qrel = json.load(handler)
    for i, exp in enumerate(os.listdir(args.exps_dir)):
        if os.path.isdir(os.path.join(args.exps_dir, exp)):
            run_file_path = os.path.join(args.exps_dir, exp, "test_evaluation", "run.json")
            with open(run_file_path) as handler:
                run = json.load(handler)
            exp_res = evaluate(qrel_file=qrel,
                               run_file=run,
                               out_path=os.path.join(args.exps_dir, exp, "test_evaluation",
                                                     "test_metrics.json"))
            # extract val perf:
            ckpt_file_path = os.path.join(args.exps_dir, exp, "saved",
                                          os.listdir(os.path.join(args.exps_dir, exp, "saved"))[
                                              0])  # only one ckpt file !
            ckpt = torch.load(ckpt_file_path, map_location=torch.device("cpu"))
            val_perf[exp] = ckpt["val_perf"]
            for k, v in exp_res.items():
                res[k].append(v)
            if exp_res["ndcg"] > best[1]["ndcg"]:
                best = (exp, exp_res)
            res["exp_id"].append(exp)
            print("+ done evaluating exp {}".format(i))
    print("== BEST config for {}: {}".format(args.exps_dir, best[0]))
    df = pd.DataFrame(res)
    df = df.set_index("exp_id")
    # sort by validation:
    val_perf = sorted(val_perf.items(), key=lambda kv: kv[1], reverse=True)
    sorted_exp_ids, sorted_val_metric = zip(*val_perf)
    sorted_df = df.loc[list(sorted_exp_ids)]  # test df sorted by validation perf
    out = sorted_df.assign(best_val=list(sorted_val_metric))
    out["exp_id"] = out.index
    out = out.reindex(columns=metrics + ["best_val", "exp_id"])
    rounding_args = {col: 3 for col in metrics + ["best_val"]}  # all the numeric columns
    out = out.round(rounding_args)
    out.to_csv(path_or_buf=os.path.join(args.exps_dir, "sorted_val_exp_metrics.csv"),
               index=False,
               sep=",")
    with open(os.path.join(args.exps_dir, "best_results.md"), "w") as handler:
        handler.write(tabulate(out.drop(columns=["exp_id"]).head(5), tablefmt="pipe", headers="keys"))
