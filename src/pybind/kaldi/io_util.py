#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import kaldi_pybind


class FileNotOpenException(Exception):
    pass


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L153
def read_vec_int(rxfilename):
    '''Read an int32 vector from an rxfilename.

    It can be used to read alignment information from `ali.scp`

    Args:
        rxfilename: filename to read.
    Returns:
        A list of int.
    '''
    ki = kaldi_pybind.Input()
    is_opened, = ki.Open(rxfilename, read_header=False)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    holder = kaldi_pybind.IntVectorHolder()
    holder.Read(ki.Stream())
    v = holder.Value().copy()
    ki.Close()

    return v


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L256
def read_vec_flt(rxfilename):
    '''Read a kaldi::Vector<float> from an rxfilename

    Args:
        rxfilename: filename to read.
    Returns:
        an object of kaldi.FloatVector.
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    v = kaldi_pybind.FloatVector()
    v.Read(ki.Stream(), is_binary)
    ki.Close()

    return v


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L376
def read_mat(rxfilename):
    '''Read a kaldi::Matrix<float> from an rxfilename

    Args:
        rxfilename: filename to read
    Returns:
       an object of kaldi.FloatMatrix
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    mat = kaldi_pybind.FloatMatrix()
    mat.Read(ki.Stream(), is_binary)
    ki.Close()

    return mat


def write_mat(mat, wxfilename, binary=True):
    '''Write a kaldi.FloatMatrix to wxfilename

    Args:
        mat: an object of kaldi.FloatMatrix
        wxfilename: filename to save the matrix
        binary: true to save the matrix in binary format;
                false to text format.
    '''
    if binary:
        write_header = True
    else:
        write_header = False

    output = kaldi_pybind.Output(filename=wxfilename, binary=binary, write_header=write_header)

    assert output.IsOpen(), 'Failed to create {}'.format(wxfilename)

    mat.Write(output.Stream(), binary=binary)


def read_transition_model(rxfilename):
    '''Read binary transition model from an rxfilename.
    Args:
        rxfilename: filename to read
    Returns:
        Return a `TransitionModel`
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    trans_model = kaldi_pybind.TransitionModel()
    trans_model.Read(ki.Stream(), is_binary)

    ki.Close()

    return trans_model


def read_nnet3_model(rxfilename):
    '''Read nnet model from an rxfilename.
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    nnet = kaldi_pybind.nnet3.Nnet()
    nnet.Read(ki.Stream(), is_binary)

    ki.Close()

    return nnet


def read_tree(rxfilename):
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    ctx_dep = kaldi_pybind.ContextDependency()
    ctx_dep.Read(ki.Stream(), is_binary)

    ki.Close()
    return ctx_dep
