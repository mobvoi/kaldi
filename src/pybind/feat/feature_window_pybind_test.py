#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestFeatureWindow(unittest.TestCase):
    def test_frame_extraction_options(self):
        opts = kaldi.FrameExtractionOptions()
        self.assertEqual(opts.samp_freq, 16000)
        self.assertEqual(opts.frame_shift_ms, 10)
        self.assertEqual(opts.frame_length_ms, 25)
        self.assertAlmostEqual(opts.preemph_coeff, 0.97)
        self.assertEqual(opts.remove_dc_offset, True)
        self.assertEqual(opts.window_type, 'povey')
        self.assertEqual(opts.round_to_power_of_two, True)
        self.assertAlmostEqual(opts.blackman_coeff, 0.42)
        self.assertEqual(opts.snip_edges, True)
        self.assertEqual(opts.allow_downsample, False)
        self.assertEqual(opts.allow_upsample, False)
        self.assertEqual(opts.max_feature_vectors, -1)

        opts.samp_freq = 48000
        self.assertEqual(opts.samp_freq, 48000)


if __name__ == '__main__':
    unittest.main()
