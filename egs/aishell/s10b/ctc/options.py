#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import argparse
import os


def _str2bool(v):
    '''
    This function is modified from
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    elif v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _set_training_args(parser):
    parser.add_argument('--train.labels-scp',
                        dest='labels_scp',
                        help='filename of labels.scp',
                        type=str)

    parser.add_argument('--train.num-epochs',
                        dest='num_epochs',
                        help='number of epochs to train',
                        type=int)

    parser.add_argument('--train.lr',
                        dest='learning_rate',
                        help='learning rate',
                        type=float)

    parser.add_argument('--train.l2-regularize',
                        dest='l2_regularize',
                        help='l2 regularize',
                        type=float)

    # TODO(fangjun): add validation feats_scp


def _check_training_args(args):
    assert os.path.isfile(args.labels_scp)

    assert args.num_epochs > 0
    assert args.learning_rate > 0
    assert args.l2_regularize >= 0

    if args.checkpoint:
        assert os.path.exists(args.checkpoint)


def _check_args(args):
    if args.is_training:
        _check_training_args(args)

    assert os.path.isdir(args.dir)
    assert os.path.isfile(args.feats_scp)

    assert args.batch_size > 0
    assert args.device_id >= 0

    assert args.input_dim > 0
    assert args.output_dim > 0
    assert args.num_layers > 0
    assert args.hidden_dim > 0
    assert args.proj_dim > 0

    assert args.log_level in ['debug', 'info', 'warning']


def get_args():
    parser = argparse.ArgumentParser(
        description='chain training in PyTorch with kaldi pybind')

    _set_training_args(parser)

    parser.add_argument('--is-training',
                        dest='is_training',
                        help='true for training, false for inference',
                        required=True,
                        type=_str2bool)

    parser.add_argument('--dir',
                        help='dir to save results. The user has to '
                        'create it before calling this script.',
                        required=True,
                        type=str)

    parser.add_argument('--feats-scp',
                        dest='feats_scp',
                        help='filename of feats.scp',
                        required=True,
                        type=str)

    parser.add_argument('--device-id',
                        dest='device_id',
                        help='GPU device id',
                        required=True,
                        type=int)

    parser.add_argument('--batch-size',
                        dest='batch_size',
                        help='batch size used in training and inference',
                        required=True,
                        type=int)

    parser.add_argument('--input-dim',
                        dest='input_dim',
                        help='input dimension of the network',
                        required=True,
                        type=int)

    parser.add_argument('--output-dim',
                        dest='output_dim',
                        help='output dimension of the network',
                        required=True,
                        type=int)

    parser.add_argument('--num-layers',
                        dest='num_layers',
                        help="number of LSTM layers in the network",
                        required=True,
                        type=int)

    parser.add_argument('--hidden-dim',
                        dest='hidden_dim',
                        help="dimension of the LSTM cell state",
                        required=True,
                        type=int)

    parser.add_argument(
        '--proj-dim',
        dest='proj_dim',
        help="dimension of the affine layer after every LSTM layer",
        required=True,
        type=int)

    parser.add_argument('--log-level',
                        dest='log_level',
                        help='log level. valid values: debug, info, warning',
                        type=str,
                        default='info')

    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='filename of the checkpoint, required for inference',
        type=str)

    args = parser.parse_args()

    _check_args(args)

    return args
