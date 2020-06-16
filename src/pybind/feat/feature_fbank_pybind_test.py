#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestFeatureFbank(unittest.TestCase):
    def test_fbank_options(self):
        opts = kaldi.FbankOptions()
        self.assertEqual(opts.use_energy, False)
        self.assertEqual(opts.energy_floor, 0)
        self.assertEqual(opts.raw_energy, True)
        self.assertEqual(opts.htk_compat, False)
        self.assertEqual(opts.use_log_fbank, True)
        self.assertEqual(opts.use_power, True)


if __name__ == '__main__':
    unittest.main()
