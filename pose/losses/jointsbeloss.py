from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class JointsBELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsBELoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def __str__(self):
        return 'JointsBELoss'

    def forward(self, output, target, target_weight):
        return self.criterion(output, target)
