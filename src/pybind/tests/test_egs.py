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
from kaldi import fst
from kaldi import nnet3


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

        srand = 0
        nnet3.srand(srand)

        eg_config = nnet3.ExampleGenerationConfig()
        eg_config.left_context = 29
        eg_config.right_context = 29
        eg_config.num_frames_str = '150,110,90'
        eg_config.frame_subsampling_factor = 3
        eg_config.num_frames_overlap = 0

        eg_config.ComputeDerived()

        utt_splitter = nnet3.UtteranceSplitter(eg_config)

        normalization_fst = fst.ReadFstKaldi('test_data/normalization.fst')
        egs_trans_mdl = kaldi.TransitionModel()

        supervision_length_tolerance = 1
        long_key = False

        for utt_id, wav in wav_reader:
            assert utt_id in lat_reader
            lat = lat_reader[utt_id]

            is_ok, phone_lat = kaldi.PhoneAlignLattice(lat, lat_trans_mdl, phone_align_opts)
            assert is_ok

            is_ok, proto_supervision = chain.PhoneLatticeToProtoSupervision(sup_opts, phone_lat)
            assert is_ok

            is_ok, supervision = chain.ProtoSupervisionToSupervision(chain_tree, chain_trans_mdl, proto_supervision,
                                                                     sup_opts.convert_to_pdfs)
            assert is_ok

            assert supervision.num_sequences == 1

            feat = fbank_computer.ComputeFeatures(wav.Data().Row(0), wav.SampFreq(), 1.0)
            print(feat.numpy().shape, type(supervision))

            num_input_frames = feat.NumRows()
            num_output_frames = supervision.frames_per_sequence

            frame_subsampling_factor = utt_splitter.Config().frame_subsampling_factor

            assert utt_splitter.LengthsMatch(utt_id, num_input_frames, num_output_frames, supervision_length_tolerance)

            if num_input_frames > num_output_frames * frame_subsampling_factor:
                num_input_frames = num_output_frames * frame_subsampling_factor

            chunks = utt_splitter.GetChunksForUtterance(num_input_frames)
            assert chunks, 'The returned chunks should not be an empty list; the wav is too short!'

            sup_splitter = chain.SupervisionSplitter(supervision)
            print(len(chunks))
            for chunk in chunks:
                start_frame_subsampled = chunk.first_frame // frame_subsampling_factor
                num_frames_subsampled = chunk.num_frames // frame_subsampling_factor

                supervision_part = sup_splitter.GetFrameRange(start_frame_subsampled, num_frames_subsampled)
                is_ok = chain.AddWeightToSupervisionFst(normalization_fst, supervision_part)
                assert is_ok

                first_frame = 0

                nnet_chain_eg = nnet3.NnetChainExample()
                nnet_chain_eg.outputs.resize(1)
                output_weights = kaldi.FloatSubVector(chunk.output_weights)

                nnet_supervision = nnet3.NnetChainSupervision('output', supervision_part, output_weights, first_frame,
                                                              frame_subsampling_factor)
                nnet_chain_eg.outputs[0] = nnet_supervision

                nnet_chain_eg.inputs.resize(1)

                tot_input_frames = chunk.left_context + chunk.num_frames + chunk.right_context
                start_frame = chunk.first_frame - chunk.left_context

                input_frames = kaldi.ExtractRowRangeWithPadding(kaldi.GeneralMatrix(feat), start_frame,
                                                                tot_input_frames)

                input_io = nnet3.NnetIo("input", -chunk.left_context, input_frames)
                nnet_chain_eg.inputs[0] = input_io

                if long_key:
                    key = '{utt_id}-{first_frame}-{left_context}-{num_frames}-{right_context}-v1'.format(
                        utt_id=utt_id,
                        first_frame=chunk.first_frame,
                        left_context=chunk.left_context,
                        num_frames=chunk.num_frames,
                        right_context=chunk.right_context)
                else:
                    key = '{}-{}'.format(utt_id, chunk.first_frame)
                # TODO(fangjun): is key needed or is nnet_chain_eg needed?


if __name__ == '__main__':
    unittest.main()
