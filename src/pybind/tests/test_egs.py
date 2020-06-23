#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

import kaldi
from kaldi import chain


class TestEgs(unittest.TestCase):

    def test(self):

        fbank_opts = kaldi.FbankOptions()
        fbank_opts.frame_opts.dither = 0
        fbank_computer = kaldi.Fbank(fbank_opts)

        phone_align_opts = kaldi.PhoneAlignLatticeOptions()
        phone_align_opts.replace_output_symbols = True

        lat_trans_mdl = kaldi.read_transition_model('test_data/lat_trans.mdl')

        wav_reader = kaldi.SequentialWaveReader('scp:test_data/wav.scp')

        lat_reader = kaldi.RandomAccessCompactLatticeReader('scp:test_data/lat.scp')
        chain_tree = kaldi.read_tree('test_data/tree')
        chain_trans_mdl = kaldi.read_transition_model('test_data/0.trans_mdl')
        print(type(chain_tree))

        sup_opts = chain.SupervisionOptions()
        sup_opts.frame_subsampling_factor = 3
        sup_opts.left_tolerance = 5
        sup_opts.right_tolerance = 5

        for key, value in wav_reader:
            assert key in lat_reader
            lat = lat_reader[key]

            is_ok, phone_lat = kaldi.PhoneAlignLattice(lat, lat_trans_mdl, phone_align_opts)
            assert is_ok

            is_ok, proto_supervision = chain.PhoneLatticeToProtoSupervision(sup_opts, phone_lat)
            assert is_ok

            is_ok, supervision = chain.ProtoSupervisionToSupervision(chain_tree, chain_trans_mdl, proto_supervision,
                                                                     sup_opts.convert_to_pdfs)
            assert is_ok

            feat = fbank_computer.ComputeFeatures(value.Data().Row(0), value.SampFreq(), 1.0)
            print(feat.numpy().shape, type(supervision))


if __name__ == '__main__':
    unittest.main()
