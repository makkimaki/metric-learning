from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WafermapTripletDataset(Dataset):
    def __init__(self,
                 dataset_path: str = "",
                 column_name_apply_wafermap: str = "apply_wafermap",
                 column_name_cite_wafermap: str = "cite_wafermap",
                 column_name_label: str = "label",
                 resize_image_size: int = 28,
                 transforms: Any = None,
                 phase: str = "train",
                 ) -> None:
        super().__init__()
        # self.__dict__.update(locals())
        self.dataset_path = dataset_path
        self.column_name_apply_wafermap = column_name_apply_wafermap
        self.column_name_cite_wafermap = column_name_cite_wafermap
        self.column_name_label = column_name_label
        self.resize_image_size = resize_image_size
        self._init_dataset()
        self.transforms = transforms
        self.phase = phase
        self.targets = self.labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.column_name_cite_wafermap is not None:
            apply_wafermap, cite_wafermap, label \
                = self.apply_wafermap[idx], \
                self.cite_wafermap[idx], \
                self.labels[idx]
            cite_wafermap = cite_wafermap.to(torch.float32)

        else:
            apply_wafermap, label = self.apply_wafermap[idx], self.labels[idx]

        if self.transforms:
            apply_wafermap = self.transforms(apply_wafermap, self.phase)
            cite_wafermap = self.transforms(cite_wafermap, self.phase)
        else:
            pass

        apply_wafermap = apply_wafermap.to(torch.float32)

        return (apply_wafermap, label)

    def _init_dataset(self):
        self.df = pd.read_pickle(self.dataset_path).reset_index()
        self.labels = self.df[self.column_name_label]

        self.df[self.column_name_apply_wafermap] = \
            self.df[self.column_name_apply_wafermap].apply(lambda x: cv2.resize(x, (self.resize_image_size, self.resize_image_size)))
        self.df[self.column_name_apply_wafermap] = \
            self.df[self.column_name_apply_wafermap].apply(lambda x: np.repeat(x[..., np.newaxis], 1, -1))
        self.apply_wafermap = self.df[self.column_name_apply_wafermap].apply(lambda x: x.transpose((2, 1, 0)))
        self.apply_wafermap = self.apply_wafermap + 1e-6
        self.apply_wafermap = self.apply_wafermap.apply(lambda x: torch.from_numpy(x))

        if self.column_name_cite_wafermap is not None:
            self.df[self.column_name_cite_wafermap] = \
                self.df[self.column_name_cite_wafermap].apply(lambda x: cv2.resize(x, (self.resize_image_size, self.resize_image_size)))
            self.df[self.column_name_cite_wafermap] = \
                self.df[self.column_name_cite_wafermap].apply(lambda x: np.repeat(x[..., np.newaxis], 1, -1))
            self.cite_wafermap = self.df[self.column_name_cite_wafermap].apply(lambda x: x.transpose((2, 1, 0)))
            self.cite_wafermap = self.cite_wafermap + 1e-6
            self.cite_wafermap = self.cite_wafermap.apply(lambda x: torch.from_numpy(x))
        else:
            pass