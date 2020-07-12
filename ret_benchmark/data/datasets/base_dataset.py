# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict

from torch.utils.data import Dataset
from ret_benchmark.utils.img_reader import read_image


class BaseDataSet(Dataset):
    """
    Basic Dataset read image path from img_source
    img_source: list of img_path and label
    """

    def __init__(self, img_source, transforms=None, mode="RGB", **kwargs):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label


class RetailDataset(BaseDataSet):
    def __init__(self, img_source, label_map_path="", transforms=None, mode="RGB", **kwargs):
        """This class is derived from BaseDataSet, with only a few difference in dealing with
        data paths and labels.

        Args:
            img_source (str): path to the data indices flat file, each row contains abs_path and label string.
            label_map (str): path to the label map file, each row is a unique label
        """
        with open(label_map_path) as f: 
            self.label_map = [l.strip() for l in f]
        super().__init__(img_source, transforms=transforms, mode=mode, **kwargs)
        self.root = ""

    def _build_label_index_dict(self):
        """Will transform label string into naturally incrementing IDs (int)
        """
        # _mapping = sorted(set(self.label_list))
        self.label_list = [self.label_map.index(l) for l in self.label_list]
        return super()._build_label_index_dict()
