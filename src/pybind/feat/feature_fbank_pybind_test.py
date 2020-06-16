#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

import kaldi
from kaldi import SequentialWaveReader


class TestFeatureFbank(unittest.TestCase):

    def test_fbank_options(self):
        opts = kaldi.FbankOptions()
        self.assertEqual(opts.use_energy, False)
        self.assertEqual(opts.energy_floor, 0)
        self.assertEqual(opts.raw_energy, True)
        self.assertEqual(opts.htk_compat, False)
        self.assertEqual(opts.use_log_fbank, True)
        self.assertEqual(opts.use_power, True)

    def test_fbank(self):
        opts = kaldi.FbankOptions()
        opts.frame_opts.dither = 0
        computer = kaldi.Fbank(opts)

        reader = SequentialWaveReader('ark:wav.ark')

        golden_reader = kaldi.RandomAccessMatrixReader('ark:fbank.ark')
        for key, value in reader:
            feat = computer.ComputeFeatures(value.Data().Row(0), value.SampFreq(), 1.0)
            self.assertTrue(golden_reader.HasKey(key))

            golden_feat = golden_reader.Value(key)
            np.testing.assert_almost_equal(feat.numpy(), golden_feat.numpy(), decimal=5)


if __name__ == '__main__':
    unittest.main()
