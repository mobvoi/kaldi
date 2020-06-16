// pybind/feat/feature_fbank_pybind.cc

// Copyright 2020   Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)

// See ../../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "feat/feature_fbank_pybind.h"

#include "feat/feature-fbank.h"

using namespace kaldi;

static void pybind_fbank_options(py::module& m) {
  using PyClass = FbankOptions;
  py::class_<PyClass> pyclass(m, "FbankOptions");
  DEF_INIT();
  DEF_P(frame_opts);
  DEF_P(mel_opts);
  DEF_P(use_energy);
  DEF_P(energy_floor);
  DEF_P(raw_energy);
  DEF_P(htk_compat);
  DEF_P(use_log_fbank);
  DEF_P(use_power);

  pyclass.def("__str__", [](const PyClass& self) {
    std::ostringstream os;

    os << "samp_freq: " << self.frame_opts.samp_freq << "\n";
    os << "frame_shift_ms: " << self.frame_opts.frame_shift_ms << "\n";
    os << "frame_length_ms: " << self.frame_opts.frame_length_ms << "\n";
    os << "dither: " << self.frame_opts.dither << "\n";
    os << "preemph_coeff: " << self.frame_opts.preemph_coeff << "\n";
    os << "remove_dc_offset: " << self.frame_opts.remove_dc_offset << "\n";
    os << "window_type: " << self.frame_opts.window_type << "\n";
    os << "round_to_power_of_two: " << self.frame_opts.round_to_power_of_two
       << "\n";
    os << "blackman_coeff: " << self.frame_opts.blackman_coeff << "\n";
    os << "snip_edges: " << self.frame_opts.snip_edges << "\n";
    os << "allow_downsample: " << self.frame_opts.allow_downsample << "\n";
    os << "allow_upsample: " << self.frame_opts.allow_upsample << "\n";
    os << "max_feature_vectors: " << self.frame_opts.max_feature_vectors
       << "\n";

    os << "num_bins: " << self.mel_opts.num_bins << "\n";
    os << "low_freq: " << self.mel_opts.low_freq << "\n";
    os << "high_freq: " << self.mel_opts.high_freq << "\n";
    os << "vtln_low: " << self.mel_opts.vtln_low << "\n";
    os << "vtln_high: " << self.mel_opts.vtln_high << "\n";
    os << "debug_mel: " << self.mel_opts.debug_mel << "\n";
    os << "htk_mode: " << self.mel_opts.htk_mode << "\n";

    os << "use_energy: " << self.use_energy << "\n";
    os << "energy_floor: " << self.energy_floor << "\n";
    os << "raw_energy: " << self.raw_energy << "\n";
    os << "htk_compat: " << self.htk_compat << "\n";
    os << "use_log_fbank: " << self.use_log_fbank << "\n";
    os << "use_power: " << self.use_power << "\n";
    return os.str();
  });
}

void pybind_feature_fbank(py::module& m) { pybind_fbank_options(m); }
