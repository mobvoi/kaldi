#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import chain
from kaldi import nnet3


class TestNnetChainExample(unittest.TestCase):

    def test_example_merging_config(self):
        merging_config = nnet3.ExampleMergingConfig()
        print(merging_config)


if __name__ == '__main__':
    unittest.main()
