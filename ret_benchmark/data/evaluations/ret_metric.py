# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import numpy as np


class RetMetric(object):
    def __init__(self, feats, labels):

        if len(feats) == 2 and type(feats) == list:
            """
            feats = [query_feats, gallery_feats]
            labels = [query_labels, gallery_labels]
            """
            self.is_equal_query = False

            self.query_feats, self.gallery_feats = feats
            self.query_labels, self.gallery_labels = labels

        else:
            self.is_equal_query = True
            self.gallery_feats = self.query_feats = feats
            self.gallery_labels = self.query_labels = labels

        self.sim_mat = np.matmul(self.query_feats, np.transpose(self.gallery_feats))

    def recall_k(self, k=1):
        m = len(self.sim_mat)

        match_counter = 0

        for i in range(m):
            pos_sim = self.sim_mat[i][self.gallery_labels == self.query_labels[i]]
            neg_sim = self.sim_mat[i][self.gallery_labels != self.query_labels[i]]

            thresh = np.sort(pos_sim)[-2] if self.is_equal_query else np.max(pos_sim)

            if np.sum(neg_sim > thresh) < k:
                match_counter += 1
        return float(match_counter) / m
