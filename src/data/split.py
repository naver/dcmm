'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import argparse
import math
import os
import pickle
import random
from shutil import copyfile

import pandas as pd

random.seed(12345)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path")
    parser.add_argument("--test_path")
    parser.add_argument("--out_dir")
    parser.add_argument("--val_ratio", type=float, default=0.8)
    args = parser.parse_args()

    data = pd.read_csv(args.train_path, sep=",", header=0)
    q_groups = data.groupby(["query_id"])
    q_groups = [df.reset_index(drop=True) for _, df in q_groups]  # list of dfs (one per query)
    print("number of queries in total:", len(q_groups))
    q_ids = [int(df.iloc[0].query_id) for df in q_groups]  # gives the query id for each group in q_groups
    # shuffling:
    indexes = list(range(len(q_ids)))
    random.shuffle(indexes)  # in place
    q_groups = [q_groups[i] for i in indexes]
    q_ids = [q_ids[i] for i in indexes]
    # builds the directories:
    out_dir = args.out_dir
    os.mkdir(out_dir)
    os.mkdir(out_dir + "/train")
    os.mkdir(out_dir + "/train/val")
    os.mkdir(out_dir + "/train/val/raw/")
    os.mkdir(out_dir + "/train/train")
    os.mkdir(out_dir + "/train/train/raw")
    os.mkdir(out_dir + "/test")
    os.mkdir(out_dir + "/test/raw")
    n = math.ceil(len(q_ids) * args.val_ratio)
    # TRAIN:
    train = q_groups[:n]
    train_query_ids = q_ids[:n]
    train_df = pd.concat(train, ignore_index=True)
    train_df.to_csv(out_dir + "/train/train/raw/interactions.csv", index=False)
    print("number of queries in train:", len(train_query_ids))
    # VAL:
    val = q_groups[n:]
    val_query_ids = q_ids[n:]
    val_df = pd.concat(val, ignore_index=True)
    val_df.to_csv(out_dir + "/train/val/raw/interactions.csv", index=False)
    print("number of queries in val:", len(val_query_ids))
    # TEST: 
    copyfile(args.test_path, out_dir + "/test/raw/interactions.csv")
    split = {"train": train_query_ids, "val": val_query_ids}
    pickle.dump(split, open(os.path.join(args.out_dir, "train_val_split.pkl"), "wb"))
