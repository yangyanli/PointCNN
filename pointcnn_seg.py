from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pointcnn import PointCNN


class Net(PointCNN):
    def __init__(self, points, features, num_class, is_training, setting):
        PointCNN.__init__(self, points, features, num_class, is_training, setting, 'segmentation')
