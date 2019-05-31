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
from model_utils import add_input, add_model, add_loss, add_training_operators
from model_utils import add_accuracy



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

    # load pre-trained network parameters
    load_init_net(config['network']['init_net'], device_opt)
    # if config['solver']['percent'] is not None:
    #     prune_init_net(config, device_opt)

    # create net
    workspace.CreateNet(validation_model.net)
    return validation_model


def get_weigts_map(config):
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    init_net_proto = caffe2_pb2.NetDef()
    init_net_pb = config['network']['init_net']
    weights_map = {}
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())
        for op in init_net_proto.op:
            _weight_shape = np.asarray(op.arg[0].ints)
            weights_map[op.output[0]] = _weight_shape

    return weights_map


def DEBUG_get_network_statistics(model, weights_map):
    # calculate params & flops
    total_params = 0.0
    total_flops = 0.0

    C_out = C_in = 3
    H_sum = H_out = H_in = 224
    W_sum = W_out = W_in = 224
    for op in model.net.Proto().op:
        print("="*100)
        print("op type: {}".format(op.type))
        print("op output: {}".format(op.output[0]))

        # deal with attention branch structure
        if len(op.input) > 0 and  op.input[0] == "res5_2_branch2c_bn":
            print("attention branch head: {}".format(op.type))
            print("H_sum: {}".format(H_sum))
            print("W_sum: {}".format(W_sum))
            H_in = H_sum
            W_in = W_sum

        # operators that adds on to params * flops
        if op.type == "Conv":
            k = float(op.arg[0].i) # kernel
            p = float(op.arg[2].i) # pad
            s = float(op.arg[4].i) # stride
            output_weight_shape = weights_map['{}_w'.format(op.output[0])]

            C_out = output_weight_shape[0]
            C_in = output_weight_shape[1]
            if op.output[0].endswith("_0_branch2a"):
                H_in = H_sum
                W_in = W_sum
            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1

            conv_params = C_out * (C_in * k * k + 1)
            conv_flops = H_out * W_out * conv_params
            print("k: {}".format(k))
            print("p: {}".format(p))
            print("s: {}".format(s))
            print("C_in: {}".format(C_in))
            print("C_out: {}".format(C_out))
            print("conv_params: {}, conv_flops: {}".format(conv_params, conv_flops))

            total_params += conv_params
            total_flops += conv_flops
        elif op.type == "SpatialBN":
            output_weight_shape = weights_map['{}_s'.format(op.output[0])]
            C_out = output_weight_shape[0]

            bn_params = C_out * 2
            bn_flops = H_in * W_in * C_out
            print("C_out: {}".format(C_out))
            print("bn_params: {}, bn_flops: {}".format(bn_params, bn_flops))

            total_params += bn_params
            total_flops += bn_flops
        elif op.type == "FC":
            output_weight_shape = weights_map['{}_w'.format(op.output[0])]
            C_out = output_weight_shape[0]
            C_in = output_weight_shape[1]

            fc_params = C_in * C_out
            fc_flops = fc_params
            print("C_in: {}".format(C_in))
            print("C_out: {}".format(C_out))
            print("fc_params: {}, fc_flops: {}".format(fc_params, fc_flops))

            total_params += fc_params
            total_flops += fc_flops
        # operators that DOES NOT add on to params * flops
        elif op.type == "MaxPool":
            k = float(op.arg[0].i) # kernel
            p = float(op.arg[2].i) # pad
            s = float(op.arg[4].i) # stride

            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1
            # treat first maxpool like "sum"
            H_sum = H_out
            W_sum = W_out
            print("k: {}".format(k))
            print("p: {}".format(p))
            print("s: {}".format(s))
            print("C_in: {}".format(C_in))
            print("C_out: {}".format(C_out))
        elif op.type == "AveragePool":
            k = float(op.arg[0].i) # kernel
            p = 0
            s = float(op.arg[1].i) # stride

            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1
            print("k: {}".format(k))
            print("p: {}".format(p))
            print("s: {}".format(s))
            print("C_in: {}".format(C_in))
            print("C_out: {}".format(C_out))
        # element-wise sum for resnet-bottleneck
        elif op.type == "Sum":
            H_sum = H_in
            W_sum = W_in
        # deal with mask mulplication for SE excition
        elif op.type == "Mul" and "excitation" in op.output[0]:
            print("here, the op is: {}".format(op.type))
            H_in = H_out = H_sum
            W_in = W_out = W_sum

        # loop iterator
        C_in = C_out
        H_in = H_out
        W_in = W_out
        print("feature map shape: (N, {}, {}, {})\n".format(C_out, H_out, W_out))

    return total_params, total_flops


def get_network_statistics(model, weights_map):
    print("+"*100)
    print("computing the params and FLOPs of network")
    # calculate params & flops
    total_params = 0.0
    total_flops = 0.0

    C_out = C_in = 3
    H_sum = H_out = H_in = 224
    W_sum = W_out = W_in = 224
    for op in model.net.Proto().op:
        print("="*100)
        # deal with attention branch structure
        if len(op.input) > 0 and  op.input[0] == "res5_2_branch2c_bn":
            H_in = H_sum
            W_in = W_sum

        # operators that adds on to params * flops
        if op.type == "Conv":
            k = float(op.arg[0].i) # kernel
            p = float(op.arg[2].i) # pad
            s = float(op.arg[4].i) # stride
            output_weight_shape = weights_map['{}_w'.format(op.output[0])]

            C_out = output_weight_shape[0]
            C_in = output_weight_shape[1]
            if op.output[0].endswith("_0_branch2a"):
                H_in = H_sum
                W_in = W_sum
            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1

            conv_params = C_out * (C_in * k * k + 1)
            conv_flops = H_out * W_out * conv_params
            print("conv_params: {}, conv_flops: {}".format(conv_params, conv_flops))

            total_params += conv_params
            total_flops += conv_flops
        elif op.type == "SpatialBN":
            output_weight_shape = weights_map['{}_s'.format(op.output[0])]
            C_out = output_weight_shape[0]

            bn_params = C_out * 2
            bn_flops = H_in * W_in * C_out
            print("bn_params: {}, bn_flops: {}".format(bn_params, bn_flops))

            total_params += bn_params
            total_flops += bn_flops
        elif op.type == "FC":
            output_weight_shape = weights_map['{}_w'.format(op.output[0])]
            C_out = output_weight_shape[0]
            C_in = output_weight_shape[1]

            fc_params = C_in * C_out
            fc_flops = fc_params
            print("fc_params: {}, fc_flops: {}".format(fc_params, fc_flops))

            total_params += fc_params
            total_flops += fc_flops
        # operators that DOES NOT add on to params * flops
        elif op.type == "MaxPool":
            k = float(op.arg[0].i) # kernel
            p = float(op.arg[2].i) # pad
            s = float(op.arg[4].i) # stride

            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1
            # treat first maxpool like "sum"
            H_sum = H_out
            W_sum = W_out
        elif op.type == "AveragePool":
            k = float(op.arg[0].i) # kernel
            p = 0
            s = float(op.arg[1].i) # stride

            H_out = (H_in + 2 * p - k) // s + 1
            W_out = (W_in + 2 * p - k) // s + 1
        # element-wise sum for resnet-bottleneck
        elif op.type == "Sum":
            H_sum = H_in
            W_sum = W_in
        # deal with mask mulplication for SE excition
        elif op.type == "Mul" and "excitation" in op.output[0]:
            H_in = H_out = H_sum
            W_in = W_out = W_sum

        # loop iterator
        C_in = C_out
        H_in = H_out
        W_in = W_out
        print("feature map shape: (N, {}, {}, {})\n".format(C_out, H_out, W_out))

    return total_params, total_flops


def get_pruned_channels_info(config):
    # set device
    device_opt = caffe2_pb2.DeviceOption()
    if config['gpu_id'] is not None:
        device_opt.device_type = caffe2_pb2.CUDA
        device_opt.cuda_gpu_id = config['gpu_id']

    # fetch all bn blobs
    bn_names = []
    bn_scales = []
    bn_bias = []
    for blob in workspace.Blobs():
        if blob.endswith('_bn_s'):
            name = blob[:-2]
            bn_names.append(name)
            bn_scales.append(workspace.FetchBlob(blob))
            # bias must be with scale for 'spatial_bn'
            assert(workspace.HasBlob(name + '_b'))
            bn_bias.append(workspace.FetchBlob(name + '_b'))

    # compute global threshold
    total_channels = 0
    for bs in bn_scales:
        total_channels += bs.shape[0]

    index = 0
    bn = np.zeros(total_channels)
    for bs in bn_scales:
        size = bs.shape[0]
        bn[index : (index + size)] = np.abs(bs)
        index += size

    sorted_bn = np.sort(bn)
    threshold_index = int(total_channels * config['solver']['percent'])
    threshold = sorted_bn[threshold_index]

    # zero out pruned channel
    pruned_channels = 0
    pruned_channels_info = {}
    for bname, bs, bb in zip(bn_names, bn_scales, bn_bias):
        # get mask with np.abs()!
        bs_abs = np.abs(bs)
        mask = (bs_abs > threshold).astype(np.float32)
        pruned_channels += mask.shape[0] - np.sum(mask)
        print("bn layer name: {} \t layer total channels: {} \t remaining"\
              "channels: {} \t pruned rate: {:.2f}%".format(
                  bname, mask.shape[0], np.sum(mask),
                  100 * float(mask.shape[0] - np.sum(mask)) / mask.shape[0]))
        pruned_channels_info['{}_s'.format(bname)] = [mask.shape[0], np.sum(mask)]

    print("total pruned rate: {}".format(pruned_channels / float(total_channels)))
    return pruned_channels_info


def update_weights_map(model, weights_map, pruned_channels_info):
    C_out = C_in = 3
    for op in model.net.Proto().op:
        print("="*100)
        # operators that adds on to params * flops
        if op.type == "Conv":
            # this bn layers should not change channel numbers because of the
            # bottle neck structure
            if op.output[0] == "conv1" or op.output[0].endswith("_branch1") or op.output[0].endswith("_branch2c"):
                output_weight_shape = weights_map['{}_w'.format(op.output[0])]
                C_out = output_weight_shape[0]
                print("op type: {}".format(op.type))
                print("in channels: {}, out channels: {}\n".format(C_in, C_out))
                continue

            # prune the channel of 'conv'
            output_weight_shape = weights_map['{}_w'.format(op.output[0])]
            old_out = output_weight_shape[0]
            old_in = output_weight_shape[1]

            if C_in != old_in:
                print("C_in: {}, old_in: {}".format(C_in, old_in))
                assert(C_in <= old_in)
                weights_map['{}_w'.format(op.output[0])][1] = C_in

            C_out = old_out
        elif op.type == "SpatialBN":
            # this bn layers should not change channel numbers because of the
            # bottle neck structure
            if op.output[0] == "res_conv1_bn" or op.output[0].endswith("_branch1_bn") or op.output[0].endswith("_branch2c_bn"):
                output_weight_shape = weights_map['{}_s'.format(op.output[0])]
                C_out = output_weight_shape[0]
                print("op type: {}".format(op.type))
                print("in channels: {}, out channels: {}\n".format(C_in, C_out))
                continue

            # prune the channel of 'bn'
            bn_name = op.output[0]
            old_bn_shape = weights_map['{}_s'.format(op.output[0])]
            assert(pruned_channels_info['{}_s'.format(bn_name)][0] == old_bn_shape[0])

            pruned_bn_channel = pruned_channels_info['{}_s'.format(bn_name)][1]
            weights_map['{}_s'.format(bn_name)][0] = pruned_bn_channel
            weights_map['{}_b'.format(bn_name)][0] = pruned_bn_channel
            weights_map['{}_rm'.format(bn_name)][0] = pruned_bn_channel
            weights_map['{}_riv'.format(bn_name)][0] = pruned_bn_channel

            # prune the channel of last conv's output channel 'bn'
            assert(len(bn_name) > 3)
            conv_name = bn_name[:-3]
            if weights_map.has_key('{}_w'.format(conv_name)):
                weights_map['{}_w'.format(conv_name)][0] = pruned_bn_channel
            if weights_map.has_key('{}_b'.format(conv_name)):
                weights_map['{}_b'.format(conv_name)][0] = pruned_bn_channel

            C_out = pruned_bn_channel

        # loop iterator
        print("op type: {}".format(op.type))
        print("in channels: {}, out channels: {}\n".format(C_in, C_out))
        C_in = C_out

    return weights_map


@file_log
def run_main(config):
    ''' running MAMC training & validation'''
    # init model
    initialize(config)

    # build model
    validation_model = build_validation_model(config) # with init net loaded

    # get information of network's  weights
    weights_map = get_weigts_map(config)

    # get params and flops
    # params, flops = get_network_statistics(validation_model, weights_map)

    # get pruned params and flops
    if config['solver']['percent'] is not None:
        print("*"*100)
        print("net work pruning...")
        print("*"*100)
        pruned_channels_info = get_pruned_channels_info(config)
        weights_map = update_weights_map(validation_model, weights_map, pruned_channels_info)
        print("\n\n\n")
        print("new weights_map:\n")
        for k, v in weights_map.items():
            print("{} : {}".format(k, v))
    params, flops = get_network_statistics(validation_model, weights_map)

    print("params: {}, FLOPs: {}".format(params, flops))


    '''
    # print network op structure
    op_set = set()
    for op in validation_model.net.Proto().op:
        print("="*100)
        print(op.type)
        op_set.add(op.type)
        print(op.input)
        print(op.output)
        print()
    print("op_set: {}".format(op_set))
    '''

    '''
    # print network conv structure
    conv_cnts = 0
    for op in validation_model.net.Proto().op:
        if op.type == "Conv":
            conv_cnts += 1
            print("="*100)
            print(op.input)
            print(op.output)
            # kernel
            print(op.arg[0])
            print(op.arg[0].name)
            print(op.arg[0].i)

            # pad
            print(op.arg[2])
            print(op.arg[2].name)
            print(op.arg[2].i)

            # stride
            print(op.arg[4])
            print(op.arg[4].name)
            print(op.arg[4].i)
            print()
    print("total conv ops: {}".format(conv_cnts))
    '''


if __name__ == '__main__':
    config = parse_args()
    run_main(config)



