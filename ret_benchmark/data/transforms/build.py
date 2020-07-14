# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torchvision.transforms as T

from .customized import PadShorter


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    transforms = []
    if cfg.DATA.TRANSFORM.PAD_SHORTER:
        transforms.append(PadShorter())
    if is_train:
        transforms.extend(
            [
                T.Resize(size=cfg.INPUT.ORIGIN_SIZE),
                T.RandomResizedCrop(
                    scale=cfg.INPUT.CROP_SCALE, size=cfg.INPUT.CROP_SIZE
                ),
                T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
                T.ToTensor(),
                normalize_transform,
            ]
        )

    else:
        transforms.extend(
            [
                T.Resize(size=cfg.INPUT.ORIGIN_SIZE),
                T.CenterCrop(cfg.INPUT.CROP_SIZE),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    return T.Compose(transforms)
