'''
DCMM
Copyright 2020-present NAVER Corp.
CC BY-NC 4.0
'''
import json
import os

import torch

"""
module implementing the prediction part of the ranking models. Loads a model and outputs the ranking for some test data.
"""


class Tester:
    def __init__(self, config, model):
        """
        config: config dict
        model: model instantiation
        """
        self.model = model
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        best = os.listdir(os.path.join(config["checkpoint_dir"], "saved"))[0]  # => because we only keep
        # a single config (the best)
        if self.device == torch.device("cuda"):
            checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "saved", best))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            print("restore model on GPU at {}".format(os.path.join(config["checkpoint_dir"], "saved", best)))
        else:  # CPU
            checkpoint = torch.load(os.path.join(config["checkpoint_dir"], "saved", best),
                                    map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print("restore model on CPU at {}".format(os.path.join(config["checkpoint_dir"], "saved", best)))
        self.model.eval()  # => put in eval mode

    def test(self, test_data, res_dir):
        """
        """
        raise NotImplementedError


class RankingTester(Tester):

    def test(self, test_data, res_dir):
        """
        test_data: a dataset object (and not a dataloader !)
        res_dir: dir to store results
        """
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        res = {}
        with open(os.path.join(test_data.root, "mapping_query_ids.json")) as handler:
            mapping_query_ids = json.load(handler)
        with open(os.path.join(test_data.root, "mapping_img_ids.json")) as handler:
            mapping_img_ids = json.load(handler)
        reverse_mapping_img_ids = {v: k for k, v in mapping_img_ids.items()}
        with torch.no_grad():  # (the model has already been put in eval mode at init)
            for i in range(len(test_data)):
                # we iterate on the dataset, element by element == query by query
                query = test_data[i]
                for k in query.keys:
                    query[k] = query[k].to(self.device)
                    # => move all the tensors to device
                scores = self.model(query).tolist()
                run = {}
                for sc, image_id in zip(scores, query.img_ids.tolist()):
                    run[reverse_mapping_img_ids[image_id]] = sc
                res[mapping_query_ids[str(query.query_id.item())]] = run
        with open(os.path.join(res_dir, "run.json"), "w") as handler:
            json.dump(res, handler)
        print("evaluation done...")
        return res
