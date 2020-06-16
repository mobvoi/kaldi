#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestMelComputations(unittest.TestCase):
    def test_mel_banks_options(self):
        opts = kaldi.MelBanksOptions()
        self.assertEqual(opts.num_bins, 25)
        self.assertEqual(opts.low_freq, 20)
        self.assertEqual(opts.high_freq, 0)
        self.assertEqual(opts.vtln_low, 100)
        self.assertEqual(opts.vtln_high, -500)
        self.assertEqual(opts.debug_mel, False)
        self.assertEqual(opts.htk_mode, False)

        opts = kaldi.MelBanksOptions(23)
        self.assertEqual(opts.num_bins, 23)


if __name__ == '__main__':
    unittest.main()
