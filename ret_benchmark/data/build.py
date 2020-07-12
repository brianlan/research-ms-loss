# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.
import importlib

from torch.utils.data import DataLoader

from .collate_batch import collate_fn
from .datasets import BaseDataSet
from .samplers import RandomIdentitySampler
from .transforms import build_transforms


def build_data(cfg, img_source, is_train=True):
    transforms = build_transforms(cfg, is_train=is_train)
    dataset_class = getattr(importlib.import_module("ret_benchmark.data.datasets"), cfg.DATA.DATASET_CLASS)
    if is_train:
        dataset = dataset_class(img_source, transforms=transforms, mode=cfg.INPUT.MODE, label_map_path=cfg.DATA.LABEL_MAP_PATH)
        sampler = RandomIdentitySampler(dataset=dataset,
                                        batch_size=cfg.DATA.TRAIN_BATCHSIZE,
                                        num_instances=cfg.DATA.NUM_INSTANCES,
                                        max_iters=cfg.SOLVER.MAX_ITERS
                                        )
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 batch_sampler=sampler,
                                 num_workers=cfg.DATA.NUM_WORKERS,
                                 pin_memory=True
                                 )
    else:
        dataset = dataset_class(img_source, transforms=transforms, mode=cfg.INPUT.MODE, label_map_path=cfg.DATA.LABEL_MAP_PATH)
        data_loader = DataLoader(dataset,
                                 collate_fn=collate_fn,
                                 shuffle=False,
                                 batch_size=cfg.DATA.TEST_BATCHSIZE,
                                 num_workers=cfg.DATA.NUM_WORKERS
                                 )
    return data_loader
