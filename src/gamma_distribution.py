from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import argparse
import yaml
import os
import sys
import cv2
import time
from tqdm import tqdm
tqdm.monitor_interval = 0

from caffe2.python import (
    workspace,
    model_helper,
    core, brew,
    optimizer,
    net_drawer
)
from caffe2.proto import caffe2_pb2

from model_utils import load_init_net

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



GPU_ID = 3
# INIT_NET_PB = "/home/zhibin/wangxiao/workshop/visual-tasks/Bi-Attention-Slimming/experiments/cars196/FINETUNE-bat_slimming-190410-174044/snapshot/bat_slimming_cars196_init_net-best.pb"
# LAMBDA = 0.0

# INIT_NET_PB = "/home/zhibin/wangxiao/workshop/visual-tasks/Bi-Attention-Slimming/experiments/cars196/FINETUNE-bat_slimming-190411-004731/snapshot/bat_slimming_cars196_init_net-best.pb"
# LAMBDA = 1e-5

INIT_NET_PB = "/home/zhibin/wangxiao/workshop/visual-tasks/Bi-Attention-Slimming/experiments/cars196/FINETUNE-bat_slimming-190411-074804/snapshot/bat_slimming_cars196_init_net-best.pb"
LAMBDA = 1e-4

def normalize_array(arr):
    V_MAX = np.max(arr)
    V_MIN = np.min(arr)
    arr_norm = np.array([(x - V_MIN) / (V_MAX - V_MIN) for x in arr])
    return arr_norm


def normalize_array2(arr):
    arr_abs = np.abs(arr)
    V_MAX = np.max(arr_abs)
    arr_norm = np.array([x / V_MAX for x in arr_abs])
    return arr_norm


def plot_distribution(X):
    plt.figure(1, figsize=(12, 8))
    plt.title('scale distribution')
    plt.hist(X, bins=np.linspace(0, 1, num=100))
    plt.draw()
    plt.savefig('./scale_distribution_{}.jpg'.format(LAMBDA))


if __name__ == "__main__":
    # initialization
    workspace.ResetWorkspace()

    # set device
    device_opt = caffe2_pb2.DeviceOption()
    device_opt.device_type = caffe2_pb2.CUDA
    device_opt.cuda_gpu_id = GPU_ID

    # load init net
    load_init_net(INIT_NET_PB, device_opt)

    # gamma statistics
    bn_scales = np.array([])
    for blob in workspace.Blobs():
        if blob.endswith('_bn_s'):
            tmp_scale = workspace.FetchBlob(blob)
            bn_scales = np.concatenate((bn_scales, tmp_scale))

    # bn_scales = normalize_array(bn_scales)
    bn_scales = normalize_array2(bn_scales)
    plot_distribution(bn_scales)


    '''
    print("="*100)
    print("bn_scales length: {}".format(len(bn_scales)))
    print("max of bn_scales:{}".format(np.max(bn_scales)))
    print("min of bn_scales:{}".format(np.min(bn_scales)))
    print("bn_scales:{}".format(bn_scales))
    '''







