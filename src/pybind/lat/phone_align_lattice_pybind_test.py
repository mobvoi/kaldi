#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestPhoneAlignLatticeOptions(unittest.TestCase):

    def test(self):
        opts = kaldi.PhoneAlignLatticeOptions()
        print(opts)


if __name__ == '__main__':
    unittest.main()
