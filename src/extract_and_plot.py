import logging
import time
import os
from shutil import copyfile
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



# hard coded params
LOG_FILE_PATH = "/home/zhibin/wangxiao/workshop/visual-tasks/Bi-Attention-Sli"\
    "mming/experiments/cars196/FINETUNE-bat_slimming-190410-174044/output.log"
FONT_SIZE = 30
LEGEND_FONT_SIZE = 23


def extract_loss(f):
    avg_train_loss = []
    avg_val_loss = []
    for current_epoch in range(1, 1+100):
        epoch_train_loss = 0.
        iter_counts = 0
        avg_epoch_val_loss = 0.
        for line in f:
            if "epoch" in line:
                loss_index = line.rfind("loss") # rfind
                loss_index += len("loss: ")
                loss_value_str = line[loss_index : loss_index + 6]
                epoch_train_loss += float(loss_value_str)
                iter_counts += 1
            elif "avg_loss" in line:
                avg_loss_index = line.rfind("avg_loss") # rfind
                avg_loss_index += len("avg_loss: ")
                avg_loss_value_str = line[avg_loss_index : avg_loss_index + 6]
                avg_epoch_val_loss = float(avg_loss_value_str)
                break
        avg_train_loss.append(epoch_train_loss / iter_counts)
        avg_val_loss.append(avg_epoch_val_loss)

    assert(len(avg_train_loss) == 100)
    assert(len(avg_val_loss) == 100)
    return avg_train_loss, avg_val_loss


def extract_acc(f):
    avg_train_acc = []
    avg_val_acc = []
    for current_epoch in range(1, 1+100):
        epoch_train_acc = 0.
        iter_counts = 0
        avg_epoch_val_acc = 0.
        for line in f:
            if "epoch" in line:
                acc_index = line.find("accuracy") # find
                acc_index += len("accuracy: ")
                acc_value_str = line[acc_index : acc_index + 6]
                epoch_train_acc += float(acc_value_str)
                iter_counts += 1
            elif "avg_acc" in line:
                avg_acc_index = line.find("avg_acc") # find
                avg_acc_index += len("avg_acc: ")
                avg_acc_value_str = line[avg_acc_index : avg_acc_index + 6]
                avg_epoch_val_acc = float(avg_acc_value_str)
                break
        avg_train_acc.append(epoch_train_acc / iter_counts)
        avg_val_acc.append(avg_epoch_val_acc)

    assert(len(avg_train_acc) == 100)
    assert(len(avg_val_acc) == 100)
    return avg_train_acc, avg_val_acc


def add_chart(chart_name, xlabel='', ylabel=''):
    plt.figure(chart_name, figsize=(12, 8))
    plt.title(chart_name, fontsize=FONT_SIZE)
    plt.xlabel(xlabel, fontsize=FONT_SIZE)
    plt.ylabel(ylabel, fontsize=FONT_SIZE)

    return chart_name


def add_plot(chart_name, epoch_results, curve_format='b.-', curve_type='debug'):
    # switch figure
    plt.figure(chart_name)
    # plot
    fig = plt.plot(
        range(1, 1+len(epoch_results)),
        epoch_results,
        curve_format,
        label='{}_{}'.format(curve_type, chart_name),
        # linewidth=4,
        # markersize=9,
    )
    # label the curve type on chart
    if len(epoch_results) == 1:  # remove duplicated legends
        plt.legend()
    # set axis notation font & size
    # legend
    font_legend = {
        'weight' : 'normal',
        'size' : LEGEND_FONT_SIZE,
    }
    legend = plt.legend(prop=font_legend)
    # coordination axis
    plt.tick_params(labelsize=LEGEND_FONT_SIZE)

    # save chart
    plt.draw()
    plt.savefig('{}.png'.format(chart_name))


if __name__ == "__main__":
    with open(LOG_FILE_PATH, 'r') as f:
        train_loss, val_loss = extract_loss(f)
        # plot
        chart_loss = add_chart('loss', xlabel='epochs', ylabel='loss')
        add_plot(chart_loss, train_loss, 'b+--', 'training')
        add_plot(chart_loss, val_loss, 'g+--', 'testing')


    with open(LOG_FILE_PATH, 'r') as f:
        train_acc, val_acc = extract_acc(f)
        # plot
        chart_acc = add_chart('accuracy', xlabel='epochs', ylabel='accuracy')
        add_plot(chart_acc, train_acc, 'r.--', 'training')
        add_plot(chart_acc, val_acc, 'c.--', 'testing')

    print("[INFO] loss and accuracy's extraction-plotting over...")







