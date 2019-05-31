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

from model_utils import load_init_net, prune_init_net
from model_utils import add_input, add_model, add_loss, add_training_operators
from model_utils import add_accuracy
from experiment import Experiment



def function_log(func):
    ''' print basic running info of function '''
    def wrapper(*args, **kwargs):
        print("[INFO] start running {} ...".format(func.__name__))
        ret = func(*args, **kwargs)
        print("[INFO] finish running {} ...\n".format(func.__name__))
        return ret
    return wrapper


def file_log(run):
    ''' print running info of the script'''
    def wrapper(*args, **kwargs):
        print("[INFO] start running {} ...".format(__file__))
        beg_time = time.time()
        ret = run(*args, **kwargs)
        end_time = time.time()
        print("[INFO] finish running {}, total_time is {:.3f}s".format(
            __file__, end_time - beg_time))
        return ret
    return wrapper


def parse_args():
    # load config file
    config_parser = argparse.ArgumentParser(
        description='Imagenet model-finetune config parser',
    )
    config_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help = 'config file'
    )
    args = config_parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
        config['config_path'] = os.path.join(os.getcwd(), args.config)
    return config


@function_log
def initialize(config):
    '''
    1. do the sanity check for path
    2. initialize workspace, e.g: add some CONST VALUE into config
    '''
    # 1. sanity check
    if not os.path.exists(config['root_dir']):
        raise ValueError("Root directory does not exist!")
    if not os.path.exists(config['network']['init_net']):
        raise ValueError("Pretrained init_net does not exist")

    # 2. initialze workspace
    workspace.ResetWorkspace(config['root_dir'])


@function_log
def build_validation_model(config):
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    # build model
    with core.DeviceScope(device_opt):
        validation_model = model_helper.ModelHelper(
            name = '{}_validation_model'.format(config['name']),
            init_params=False,
        )
        data, label = add_input(validation_model, config, is_test=True)
        pred = add_model(validation_model, config, data, is_test=True)
        loss = add_loss(validation_model, config, pred, label)
        add_accuracy(validation_model)

    # init workspace for validation net
    workspace.RunNetOnce(validation_model.param_init_net)

    # load pretrained network params
    load_init_net(config['network']['init_net'], device_opt)

    workspace.CreateNet(validation_model.net)
    return validation_model


@file_log
def run_main(config):
    ''' running MAMC training & validation'''
    # init model
    initialize(config)
    validation_model= build_validation_model(config)

    total_iter_time = 0
    for test_iter in range(config['solver']['test_iterations']):
        single_iter_beg = time.time()
        workspace.RunNet(validation_model.net)
        single_iter_end = time.time()

        single_iter_time = single_iter_end - single_iter_beg
        one_forward_time = single_iter_time / config['training_data']['input_transform']['batch_size']
        print("one_forward_time of inference is: {:.3f}s".format(one_forward_time))

        total_iter_time += single_iter_time
    avg_forward_time = total_iter_time / (config['solver']['test_iterations'] * config['training_data']['input_transform']['batch_size'])
    print("average one_forward_time of inference is: {:.3f}s".format(one_forward_time))


if __name__ == '__main__':
    config = parse_args()
    run_main(config)




