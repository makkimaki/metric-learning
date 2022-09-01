import argparse
import sys
from tkinter import Image
from turtle import distance

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from typing import Tuple, Any, List

from pathlib import Path
from torchvision import datasets, transforms

from src.dataset import WafermapTripletDataset
from src.models import Net, Embedder


def parse_arguments():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

class ImageRetrieval(object):
    def __init__(self,
                 inference_model = None,
                 dataset_class = None,
                 cite_data_path: Path = None,
                 applydb_data_path: Path = None,
                 cite_wafer_id: str = None,
                 apply_wafer_id: str = None,
                 load_index: bool = False,
                 resize_image_size: int = None,
                 ) -> None:
        """_summary_

        Args:
            inference_model (_type_, optional): _description_. Defaults to None.
            dataset_class (_type_, optional): _description_. Defaults to None.
            cite_data_path (Path, optional): _description_. Defaults to None.
            applydb_data_path (Path, optional): _description_. Defaults to None.
            cite_wafer_id (str, optional): _description_. Defaults to None.
            apply_wafer_id (str, optional): _description_. Defaults to None.
            load_index (bool, optional): _description_. Defaults to False.
            resize_image_size (int, optional): _description_. Defaults to None.
        """
        self._inference_model = inference_model
        self._dataset_class = dataset_class

        if self.applydb_data_path is not None:
            self._init_dataset()
            assert self.cite_wafer_id in list(self.df_cite["waferId"])
        else:
            self.test_dataset = self._dataset_class
    
    def search():

    def retrieve():

    def train_knn():

    def _init_dataset():

    def preprocess_image():


if __name__ == "__main__":
    args = parse_arguments()

    # load the models(Net, Embedder)
    trunk = Net()
    trunk.load_state_dict(torch.load(args.saved_trunk_path))
    trunk.cpu()

    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=0.7)
    inference_model = InferenceModel(trunk=trunk,
                                     embedder=embedder,
                                     match_finder=match_finder,
                                     data_device="cpu"
                                     )
    dataset_class = WafermapTripletDataset
    query_wafer_id = args.cite_wafer_id

    image_retrieval = ImageRetrieval(inference_model = inference_model,
                                     dataset_class = dataset_class,
                                     cite_data_path = args.cite_data_path,
                                     applydb_data_path = args.applydb_data_path,
                                     cite_wafer_id = args.cite_wafer_id,
                                     apply_wafer_id = args.apply_wafer_id,
                                     load_index = args.load_index,
                                     resized_image_size = args.resized_image_size,
                                     )
    distances_list, indices_list = image_retrieval.search(query_wafer_id=query_wafer_id, 
                                                          top_k=args.top_k,
                                                          )
    df_result = image_retrieval.retrieve(indices_list)
    check_statement = f"{args.apply_wafer_id}" in list(df_result["waferId"])
    print(f"Wafer ID: {query_wafer_id} EXISTS?: {check_statement}", f"(Score={[1 if check_statement else 0][0]}")


