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
    net_drawer,
    scope,
)
from caffe2.proto import caffe2_pb2

from add_resnet50_l1_sparsified_model import add_resnet50, add_resnet50_core
from caffe2.python.optimizer import (
    Optimizer,
    _get_param_to_device,
    get_param_device,
)



# configs
SPARSE_SCALE = 0.00001
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.01
MOMENTUM = 0.9
MAX_ITER = 10000

class L1NormBuilder(Optimizer):
    def __init__(self, sparse_scale):
        self.sparse_scale = sparse_scale

    def _run(self, net, param_init_net, param_info):
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)

        ONE = param_init_net.ConstantFill(
            [],
            "ONE_{}_{}".format(dev.device_type, dev.cuda_gpu_id),
            shape=[1],
            value=1.0
        )
        SS = param_init_net.ConstantFill(
            [], "SS_{}_{}".format(dev.device_type, dev.cuda_gpu_id),
            shape=[1], value=self.sparse_scale
        )

        if isinstance(param_info.grad, core.GradientSlice):
            raise ValueError(
                "Weight decay does not yet support sparse gradients")
        else:
            param_sign = net.Sign(
                [param_info.blob],
                ['{}_sign'.format(param_info.blob)],
            )
            net.WeightedSum(
                [param_info.grad, ONE, param_sign, SS],
                param_info.grad,
            )

def _build_l1_bn(
    model,
    optimizer,
    weights_only=False,
    use_param_info_optim=True,
    max_gradient_norm=None,
    allow_lr_injection=False,
):
    param_to_device = _get_param_to_device(model)

    # Validate there are no duplicate params
    model.Validate()

    params = []
    for param_info in model.GetOptimizationParamInfo():
        if weights_only and param_info.blob not in model.weights:
            continue
        # add L1 norm for spatial bn
        if param_info.name.endswith('bn_s'):
            params.append(param_info)

    lr_multiplier = None
    if max_gradient_norm is not None:
        lr_multiplier = _calc_norm_ratio(
            model,
            params,
            'norm_clipped_grad_update',
            param_to_device,
            max_gradient_norm,
        )

    if allow_lr_injection:
        if not model.net.BlobIsDefined(_LEARNING_RATE_INJECTION):
            lr_injection = model.param_init_net.ConstantFill(
                [],
                _LEARNING_RATE_INJECTION,
                shape=[1],
                value=1.0,
            )
        else:
            lr_injection = _LEARNING_RATE_INJECTION

        if lr_multiplier is None:
            lr_multiplier = lr_injection
        else:
            lr_multiplier = model.net.Mul(
                [lr_multiplier, lr_injection],
                'lr_multiplier',
                broadcast=1,
            )
    optimizer.add_lr_multiplier(lr_multiplier)

    for param_info in params:
        param_name = str(param_info.blob)

        device = get_param_device(param_name, param_info.grad, param_to_device)

        with core.DeviceScope(device):
            if param_info.optimizer and use_param_info_optim:
                param_info.optimizer(model.net, model.param_init_net, param_info)
            else:
                optimizer(model.net, model.param_init_net, param_info)
    return optimizer


def add_l1_normalization_bn(model, sparse_scale):
    _build_l1_bn(
        model,
        L1NormBuilder(sparse_scale=sparse_scale),
        # WeightDecayBuilder(sparse_scale=sparse_scale),
        weights_only=True,
        use_param_info_optim=False,
    )


if __name__ == "__main__":
    # add model
    model = model_helper.ModelHelper('foo')

    # add input
    data = model.param_init_net.GivenTensorFill(
        [],
        ['data'],
        shape=[1, 3, 224, 224],
        values=np.random.randn(1, 3, 224, 224),
    )
    label = model.param_init_net.ConstantFill([], ['label'], shape=[1], value=1)

    # add model
    pred = add_resnet50(model, data)

    # add loss
    softmax, loss = model.net.SoftmaxWithLoss(
        [pred, label],
        ['softmax', 'loss'],
    )

    # add training operator
    model.AddGradientOperators([loss])

    # add optimizer
    optimizer.add_weight_decay(model, WEIGHT_DECAY)
    add_l1_normalization_bn(model, SPARSE_SCALE)
    optimizer.build_multi_precision_sgd(
        model,
        base_learning_rate = LEARNING_RATE,
        momentum = MOMENTUM,
        nesterov = 1,
        policy = 'poly',
        power = 1.,
        max_iter = MAX_ITER,
    )

    # initialization
    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)
    print("hello foo")

    # ================= DEBUG PRINT =======================
    # print(model.net.Proto())

    # print(model.param_init_net.Proto())

    # i = 0
    # for param in model.param_to_grad:
    #     print("{} : {}".format(i , param))
    #     i += 1

    # i = 1
    # for param in model.params:
    #     print("{} : {}".format(i , param))
    #     i += 1

    i = 1
    for blob in workspace.Blobs():
         print("{} : {}".format(i , blob))
         i += 1













